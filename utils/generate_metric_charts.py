#!/usr/bin/env python3
# generate_metric_charts.py
"""
generate_metric_charts.py
--------------------------------------------------
Create 1920×1080 Apple-Keynote-style line charts (English or Chinese)
for each metric in a 10-year history table, with Buffett's standard lines where applicable.

Usage:
    python generate_metric_charts.py --input metrics.csv \
           --font SourceHanSansSC-Bold.otf \
           --output metrics \
           --theme dark \
           --language chinese \
           --formats png,svg,jpg

Input file format (CSV or Excel):
    Year,Metric1,Metric2,...
    2015,...,...
    2016,...,...

• The first column must be named "Year".
• For bilingual labels, name a column like:
      "Revenue ($B)|收入（十亿美元）"
  (English and Chinese separated by a pipe "|").
• Percent metrics should include a "%" in the header so the script auto-formats the y-axis.
• Ratio metrics can contain "P/" or "/" (e.g. "Debt/Equity").
• Buffett's standard lines are drawn for metrics like P/E, P/B, Debt/Equity, etc.

Produces (examples):
    <output>/1. <metric>.png / .svg / .jpg
    metric_charts.zip       (all images zipped)
--------------------------------------------------
"""

import argparse, os, gc, sys, pandas as pd, matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mtick, matplotlib.font_manager as fm
from zipfile import ZipFile
import re

# ---------- Config (feel free to tweak) ----------
FIGSIZE = (19.2, 10.8)   # inches → 1920×1080 at dpi=100
TITLE_SIZE = 36
LABEL_SIZE = 28
TICK_SIZE = 24
LINE_WIDTH = 3
MARKER_SIZE = 8
ANNOTATION_OFFSET = 20  # Base offset in points for annotations
MIN_DISTANCE = 0.1      # Minimum y-value difference to consider annotations or lines separate
LINE_PROXIMITY_THRESHOLD = 0.5  # Minimum y-distance from lines to avoid overlap
MAX_RECURSION = 10      # Maximum recursion depth to prevent stack overflow

# Buffett's standard thresholds with direction
BUFFETT_STANDARDS = {
    'P/E': {'threshold': 15, 'direction': '≤'},
    'P/B': {'threshold': 1.5, 'direction': '≤'},
    'Debt/Equity': {'threshold': 0.5, 'direction': '<'},
    'Interest Coverage': {'threshold': 3, 'direction': '≥'},
    'Interest Expense/Operating Income': {'threshold': 0.15, 'direction': '≤'},
    'Current Ratio': {'threshold': 1.5, 'direction': '>'},
    'Gross Margin': {'threshold': 40, 'direction': '≳'},
    'Net Profit Margin': {'threshold': 20, 'direction': '≳'},
    'ROE': {'threshold': 15, 'direction': '≥'},
    'ROIC': {'threshold': 12, 'direction': '≥'},
    'ROTC': {'threshold': 12, 'direction': '≥'},
    'ROA': {'threshold': 6, 'direction': '≥'},
    'Depreciation/Gross Profit': {'threshold': 0.1, 'direction': '≤'},
    'FCF/Revenue': {'threshold': 5, 'direction': '>'},
    'Return on Retained Earnings': {'threshold': 12, 'direction': '≥'},
    'Owner’s-earnings/Market Cap': {'threshold': 10, 'direction': '≥'}
}

# Theme configurations
THEMES = {
    'dark': {
        'figure.facecolor': '#1A2B3C',
        'axes.facecolor': '#1A2B3C',
        'axes.edgecolor': '#E0E0E0',
        'axes.labelcolor': '#E0E0E0',
        'xtick.color': '#E0E0E0',
        'ytick.color': '#E0E0E0',
        'line_color': '#40C4FF',
        'avg_line_color': '#FFCA28',
        'buffett_line_color': '#FF5252',
        'latest_annot_color': '#26A69A',
        'max_annot_color': '#66BB6A',
        'min_annot_color': '#EF5350',
        'title_color': '#E0E0E0',
        'spine_color': '#E0E0E0'
    },
    'light': {
        'figure.facecolor': '#F8FAFC',
        'axes.facecolor': '#F8FAFC',
        'axes.edgecolor': '#2D3748',
        'axes.labelcolor': '#2D3748',
        'xtick.color': '#2D3748',
        'ytick.color': '#2D3748',
        'line_color': '#2563EB',
        'avg_line_color': '#F59E0B',
        'buffett_line_color': '#DC2626',
        'latest_annot_color': '#9333EA',
        'max_annot_color': '#16A34A',
        'min_annot_color': '#DB2777',
        'title_color': '#2D3748',
        'spine_color': '#2D3748'
    }
}

# Language-specific labels for annotations and x-axis
LABELS = {
    'english': {'avg': 'Avg','buffett': 'Buffett Standard','latest': 'Latest','max': 'Max','min': 'Min','xlabel': 'Fiscal Year'},
    'chinese': {'avg': '平均值','buffett': '巴菲特标准','latest': '最新','max': '最高','min': '最低','xlabel': '财政年度'}
}

# Language-specific y-axis labels
YAXIS_LABELS = {
    'english': {'%': '%','Ratio (x)': 'Ratio (x)','USD': 'USD','Price (USD)': 'Price (USD)','USD Billions': 'USD Billions','USD Millions': 'USD Millions','USD Per Share': 'USD Per Share','': ''},
    'chinese': {'%': '%','Ratio (x)': '比率 (x)','USD': '美元','Price (USD)': '价格 (美元)','USD Billions': '亿美元','USD Millions': '百万美元','USD Per Share': '每股美元','': ''}
}

# Metric categories for special handling
PRICE_METRICS = ['MarketPriceUSD']
RATIO_METRICS = ['PriceToEarning', 'PriceToBook', 'PEGRatio', 'CurrentRatio', 'DebtToEquityRatio', 'InterestCoverageRatio', 'CapitalExpenditureToDepreciation']
PERCENTAGE_METRICS = ['RevenueGrowthRate', 'NetIncomeGrowthRate', 'BookValueGrowthRate', 'GrossMargin', 'OperatingMargin', 'NetIncomeMargin', 'FreeCashFlowMargin', 'ReturnOnEquity', 'ReturnOnAssets', 'ReturnOnInvestedCapital', 'RetainedEarningsToNetIncome', 'DividendPayoutRatio']

# Comprehensive Chinese translations for metrics
METRIC_TRANSLATIONS = {
    # Core Metrics
    'MarketPriceUSD': '市场价格 (美元)',
    'Revenues': '营业收入',
    'GrossProfit': '毛利润',
    'OperatingIncomeLoss': '营业利润',
    'NetIncomeLoss': '净利润',
    'NetCashFromOperatingActivities': '经营活动现金流',
    'Assets': '总资产',
    'CashAndCashEquivalents': '现金及现金等价物',
    'Liabilities': '总负债',
    'StockholdersEquity': '股东权益',
    # Derived Metrics
    'EarningsPerShare': '每股收益 (EPS)',
    'PriceToEarning': '市盈率 (P/E)',
    'PEGRatio': '市盈率相对盈利增长比率 (PEG)',
    'BookValuePerShare': '每股账面价值',
    'PriceToBook': '市净率 (P/B)',
    'RevenueGrowthRate': '营收增长率',
    'NetIncomeGrowthRate': '净利润增长率',
    'BookValueGrowthRate': '账面价值增长率',
    'GrossMargin': '毛利率',
    'OperatingMargin': '营业利润率',
    'NetIncomeMargin': '净利率',
    'FreeCashFlowMargin': '自由现金流利润率',
    'FreeCashFlow': '自由现金流',
    'OwnerEarnings': '股东盈余',
    'CurrentRatio': '流动比率',
    'DebtToEquityRatio': '债务权益比',
    'InterestCoverageRatio': '利息保障倍数',
    'ReturnOnEquity': '净资产收益率 (ROE)',
    'ReturnOnAssets': '总资产收益率 (ROA)',
    'ReturnOnInvestedCapital': '投入资本回报率 (ROIC)',
    'RetainedEarningsToNetIncome': '留存收益与净利润比',
    'DividendPayoutRatio': '股息支付率',
    'CapitalExpenditureToDepreciation': '资本支出与折旧比'
}

# -------------------------------------------------
def translate_metric_name(metric_name: str, language: str) -> str:
    if language != 'chinese':
        return metric_name
    
    # Use the new comprehensive translation dictionary
    return METRIC_TRANSLATIONS.get(metric_name, metric_name)

def sanitize_filename(filename: str) -> str:
    invalid_chars = r'[\/:*?"<>|%]'
    sanitized = re.sub(invalid_chars, '_', filename)
    sanitized = sanitized.strip().replace('  ', ' ')
    return sanitized

def detect_ylabel(metric_name: str, language: str):
    if 'Per Share' in metric_name:
        return YAXIS_LABELS[language]['USD Per Share']
    if metric_name in PRICE_METRICS:
        return YAXIS_LABELS[language]['Price (USD)']
    if metric_name in RATIO_METRICS:
        return YAXIS_LABELS[language]['Ratio (x)']
    if metric_name in PERCENTAGE_METRICS:
        return YAXIS_LABELS[language]['%']
    if '$' in metric_name or 'USD' in metric_name:
        return YAXIS_LABELS[language]['USD']
    return YAXIS_LABELS[language]['']

def get_buffett_standard(metric_name: str):
    for key, info in BUFFETT_STANDARDS.items():
        if key.lower() in metric_name.lower().replace(' ', ''):
            return key, info['threshold'], info['direction']
    return None, None, None

def load_table(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.xlsx', '.xls']:
        return pd.read_excel(path)
    return pd.read_csv(path)

def adjust_annotation_position(ax, fig, x, y, text, color, offset_x=10, offset_y=0, avoid=None, line_y_positions=None, recursion_depth=0):
    if recursion_depth > MAX_RECURSION:
        return ax.annotate(text, xy=(x, y), xytext=(offset_x, 0), textcoords='offset points',
                           color=color, ha='right', va='center')

    annot = ax.annotate(text, xy=(x, y), xytext=(offset_x, offset_y),
                        textcoords='offset points', color=color, ha='right', va='center')
    renderer = fig.canvas.get_renderer()
    bbox1 = annot.get_window_extent(renderer=renderer)
    y_limits = ax.get_ylim()
    data_y = y + offset_y / 72

    if bbox1.y0 < ax.get_window_extent().y0 or bbox1.y1 > ax.get_window_extent().y1:
        if data_y > y_limits[1]:
            offset_y = int((y_limits[1] - y) * 72 - offset_y)
        elif data_y < y_limits[0]:
            offset_y = int((y_limits[0] - y) * 72 - offset_y)
        annot.remove()
        return adjust_annotation_position(ax, fig, x, y, text, color, offset_x, offset_y, avoid, line_y_positions, recursion_depth + 1)

    if avoid:
        for other_annot in avoid:
            bbox2 = other_annot.get_window_extent(renderer=renderer)
            if bbox1.overlaps(bbox2) or abs(y - other_annot.get_position()[1]) < MIN_DISTANCE:
                offset_y += ANNOTATION_OFFSET * (1 + avoid.index(other_annot))
                annot.remove()
                return adjust_annotation_position(ax, fig, x, y, text, color, offset_x, offset_y, avoid, line_y_positions, recursion_depth + 1)

    if line_y_positions:
        for line_y in line_y_positions:
            if abs(data_y - line_y) < LINE_PROXIMITY_THRESHOLD:
                offset_y += ANNOTATION_OFFSET
                annot.remove()
                return adjust_annotation_position(ax, fig, x, y, text, color, offset_x, offset_y, avoid, line_y_positions, recursion_depth + 1)

    return annot

def main():
    parser = argparse.ArgumentParser(description="Generate key-metric line charts")
    parser.add_argument("--input", required=True, help="CSV or Excel with Year column")
    parser.add_argument("--font", default=None, help="OTF/TTF font for CJK (e.g. SourceHanSansSC-Bold.otf)")
    parser.add_argument("--output", default=None, help="Output directory for images (defaults to <input_filename>)")
    parser.add_argument("--zip", default="metric_charts.zip", help="Name of ZIP archive")
    parser.add_argument("--theme", choices=['dark', 'light'], default='dark', help="Chart theme: dark or light")
    parser.add_argument("--language", choices=['english', 'chinese'], default='english', help="Output language: english or chinese")

    # Multi-format support
    parser.add_argument("--format", dest="img_format",
                        choices=["png", "svg", "jpg", "jpeg", "pdf"],
                        default=None,
                        help="[Deprecated] Single output format; use --formats instead.")
    parser.add_argument("--formats", default=None,
                        help="Comma or space separated list of formats, e.g. 'png,svg,jpg'. Default: png")

    args = parser.parse_args()

    # Output dir default
    if args.output is None:
        args.output = os.path.splitext(os.path.basename(args.input))[0]

    # Font
    if args.font and os.path.exists(args.font):
        fm.fontManager.addfont(args.font)
        plt.rcParams["font.family"] = fm.FontProperties(fname=args.font).get_name()

    df = load_table(args.input)
    if "Year" not in df.columns:
        sys.exit("Input file must have a 'Year' column as first column.")

    years = df["Year"].tolist()

    # Theme
    valid_rcparams = {k: v for k, v in THEMES[args.theme].items()
                      if k in ['figure.facecolor', 'axes.facecolor', 'axes.edgecolor',
                               'axes.labelcolor', 'xtick.color', 'ytick.color']}
    plt.rcParams.update({
        "font.size": LABEL_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        **valid_rcparams
    })

    # Normalize formats
    allowed = {"png", "svg", "jpg", "jpeg", "pdf"}
    if args.formats:
        formats = [f for f in re.split(r"[,\s]+", args.formats.lower()) if f]
    elif args.img_format:
        formats = [args.img_format.lower()]
    else:
        formats = ["png"]

    norm = []
    for f in formats:
        if f not in allowed:
            sys.exit(f"Unsupported format '{f}'. Choose from: {sorted(allowed)}")
        norm.append("jpg" if f == "jpeg" else f)
    formats = norm

    os.makedirs(args.output, exist_ok=True)

    idx = 1
    for col in df.columns:
        if col == "Year":
            continue

        # Split English|Chinese if provided
        if "|" in col:
            eng, chi = col.split("|", 1)
        else:
            eng, chi = col, col
        data = df[col]

        # Title by language
        if args.language == 'chinese':
            title_text = chi.strip() if chi != col else translate_metric_name(eng.strip(), 'chinese')
        else:
            title_text = eng.strip()

        # Clean data
        cleaned_years, cleaned_data = [], []
        for x, y in zip(years, data):
            try:
                x_int = int(x)
                if isinstance(y, str):
                    y = y.replace('%', '').replace('$', '').replace(',', '').strip()
                y_float = float(y)
                if not (y_float != y_float):  # NaN check
                    cleaned_years.append(x_int)
                    cleaned_data.append(y_float)
            except (ValueError, TypeError):
                continue

        if not cleaned_data:
            print(f"⚠️  Skipping '{title_text}' due to no valid data.")
            continue

        unit_suffix = ''
        y_label = detect_ylabel(eng, args.language)

        # Check if it's a monetary value that needs scaling
        if y_label == YAXIS_LABELS[args.language]['USD'] or y_label == '':
            max_val = max(abs(y) for y in cleaned_data) if cleaned_data else 0
            if max_val >= 1000:
                cleaned_data = [y / 1000 for y in cleaned_data]
                unit_suffix = ' (Billions)' if args.language == 'english' else ' (十亿)'
            else:
                unit_suffix = ' (Millions)' if args.language == 'english' else ' (百万)'

        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.plot(cleaned_years, cleaned_data, marker='o',
                color=THEMES[args.theme]['line_color'],
                linewidth=LINE_WIDTH, markersize=MARKER_SIZE)

        annotations = []
        line_y_positions = [y for y in cleaned_data]
        if cleaned_data:
            avg = sum(cleaned_data) / len(cleaned_data)
            ax.axhline(avg, color=THEMES[args.theme]['avg_line_color'], linestyle='--', linewidth=2)
            line_y_positions.append(avg)
            annotations.append(ax.text(0.99, avg, f"{LABELS[args.language]['avg']}: {avg:.2f}",
                                       color=THEMES[args.theme]['avg_line_color'], fontsize=25,
                                       ha='right', va='bottom', transform=ax.get_yaxis_transform()))

            _, buffett_threshold, buffett_direction = get_buffett_standard(eng)
            if buffett_threshold is not None:
                ax.axhline(buffett_threshold, color=THEMES[args.theme]['buffett_line_color'],
                           linestyle=':', linewidth=2)
                line_y_positions.append(buffett_threshold)
                annotations.append(ax.text(0.99, buffett_threshold,
                                           f"{LABELS[args.language]['buffett']} ({buffett_direction}{buffett_threshold:.2f})",
                                           color=THEMES[args.theme]['buffett_line_color'], fontsize=25,
                                           ha='right', va='top', transform=ax.get_yaxis_transform()))

            latest_x, latest_y = cleaned_years[-1], cleaned_data[-1]
            latest_annot = adjust_annotation_position(ax, fig, latest_x, latest_y,
                                                      f"{LABELS[args.language]['latest']}: {latest_y:.2f}",
                                                      THEMES[args.theme]['latest_annot_color'],
                                                      offset_x=10, offset_y=0, avoid=annotations,
                                                      line_y_positions=line_y_positions)
            annotations.append(latest_annot)

            max_idx = cleaned_data.index(max(cleaned_data))
            max_x, max_y = cleaned_years[max_idx], cleaned_data[max_idx]
            offset_y = ANNOTATION_OFFSET if abs(max_y - latest_y) < MIN_DISTANCE else 0
            max_annot = adjust_annotation_position(ax, fig, max_x, max_y,
                                                   f"{LABELS[args.language]['max']}: {max_y:.2f}",
                                                   THEMES[args.theme]['max_annot_color'],
                                                   offset_x=10, offset_y=offset_y, avoid=annotations,
                                                   line_y_positions=line_y_positions)
            annotations.append(max_annot)

            min_idx = cleaned_data.index(min(cleaned_data))
            min_x, min_y = cleaned_years[min_idx], cleaned_data[min_idx]
            offset_y = ANNOTATION_OFFSET if abs(min_y - latest_y) < MIN_DISTANCE or abs(min_y - max_y) < MIN_DISTANCE else 0
            min_annot = adjust_annotation_position(ax, fig, min_x, min_y,
                                                   f"{LABELS[args.language]['min']}: {min_y:.2f}",
                                                   THEMES[args.theme]['min_annot_color'],
                                                   offset_x=10, offset_y=offset_y, avoid=annotations,
                                                   line_y_positions=line_y_positions)
            annotations.append(min_annot)

        ax.set_title(title_text, color=THEMES[args.theme]['title_color'], pad=20)
        ax.set_xlabel(LABELS[args.language]['xlabel'], color=THEMES[args.theme]['axes.labelcolor'])
        ax.set_ylabel(y_label + unit_suffix, color=THEMES[args.theme]['axes.labelcolor'])

        # Percent formatter
        if y_label == YAXIS_LABELS[args.language]['%']:
            ax.yaxis.set_major_formatter(mtick.PercentFormatter())

        ax.tick_params(colors=THEMES[args.theme]['xtick.color'])
        for spine in ax.spines.values():
            spine.set_color(THEMES[args.theme]['spine_color'])
        ax.margins(x=0.05)

        # ---- Save in all requested formats ----
        base_fname = sanitize_filename(f"{idx}. {title_text}")
        for fext in formats:
            fname = base_fname + "." + fext
            rc = {}
            if fext == "svg":
                rc["svg.fonttype"] = "path"   # safest for CJK glyphs
            elif fext == "pdf":
                rc["pdf.fonttype"] = 42
                rc["ps.fonttype"] = 42

            save_kwargs = {"bbox_inches": "tight", "format": fext}
            if fext in {"png", "jpg"}:
                save_kwargs["dpi"] = 100  # 19.2x10.8 in -> 1920x1080 px
            if fext == "jpg":
                # IMPORTANT: JPEG quality must be passed via pil_kwargs
                save_kwargs["pil_kwargs"] = {"quality": 95}

            out_path = os.path.join(args.output, fname)
            with mpl.rc_context(rc):
                fig.savefig(out_path, **save_kwargs)

        plt.close(fig)
        gc.collect()
        idx += 1

    # Zip results
    with ZipFile(args.zip, 'w') as z:
        for f in sorted(os.listdir(args.output)):
            z.write(os.path.join(args.output, f), arcname=f)
    print(f"✅ Charts saved to '{args.output}/' in formats {formats} and zipped into '{args.zip}'")

if __name__ == "__main__":
    main()
