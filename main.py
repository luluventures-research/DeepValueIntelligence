import argparse
from datetime import date as _date, datetime
from pathlib import Path
import os
import re
from typing import Dict, List, Optional, Tuple

from investingagents.graph.trading_graph import InvestingAgentsGraph
from investingagents.default_config import DEFAULT_CONFIG

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run InvestingAgents analysis.")
    parser.add_argument("--ticker", default="NVDA", help="Stock ticker symbol.")
    parser.add_argument(
        "--date",
        default=_date.today().isoformat(),
        help="Trade date in YYYY-MM-DD format (defaults to today).",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "google", "anthropic"],
        default="google",
        help="LLM provider to use.",
    )
    parser.add_argument("--deep-model", default="gemini-3-pro-preview", help="Deep thinking model name.")
    parser.add_argument("--quick-model", default="gemini-3-flash-preview", help="Quick thinking model name.")
    parser.add_argument("--backend-url", default="https://generativelanguage.googleapis.com/v1", help="LLM backend URL.")
    parser.add_argument("--max-debate-rounds", type=int, default=1, help="Number of debate rounds.")
    parser.add_argument(
        "--llm-timeout",
        type=int,
        default=1800,
        help="LLM request timeout in seconds (default: 1800).",
    )
    parser.add_argument(
        "--llm-max-retries",
        type=int,
        default=5,
        help="Max retries for transient LLM errors (default: 5).",
    )
    parser.add_argument(
        "--online-tools",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable online tools (default: enabled).",
    )
    parser.add_argument(
        "--embedding-provider",
        choices=["openai", "google", "ollama", "none"],
        default="",
        help="Embedding provider (default: auto).",
    )
    parser.add_argument(
        "--embedding-model",
        default="",
        help="Embedding model name (provider-specific).",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    parser.add_argument(
        "--generate-thumbnail",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Generate 1920x1080 thumbnails with Gemini (Nano Banana).",
    )
    parser.add_argument(
        "--plot-fundamentals",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate key-metric charts from fundamentals historical table (default: enabled).",
    )
    return parser


def _ensure_string(content) -> str:
    """Convert content to string, extracting only 'text' field from Gemini's format."""
    if content is None:
        return ""
    if isinstance(content, str):
        stripped = content.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                import ast
                parsed = ast.literal_eval(content)
                return _ensure_string(parsed)
            except (ValueError, SyntaxError):
                pass
        return content
    if isinstance(content, dict):
        if content.get("type") == "text" and "text" in content:
            return content["text"]
        return str(content)
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
            elif isinstance(item, str):
                text_parts.append(item)
            else:
                text_parts.append(str(item))
        return "\n\n".join(text_parts)
    return str(content)


def _filter_trading_recommendations(content: str) -> str:
    """
    Filter out explicit BUY/SELL/HOLD recommendations from content for the Intelligence Summary.
    Removes lines containing explicit trading decisions while preserving the analysis.
    """
    import re

    if not content:
        return content

    lines = content.split("\n")
    filtered_lines = []

    skip_patterns = [
        r"^\s*\*{0,2}(FINAL\s+)?(DECISION|RECOMMENDATION|VERDICT)\s*:\s*\*{0,2}\s*(BUY|SELL|HOLD|STRONG\s+BUY|STRONG\s+SELL)",
        r"^\s*\*{0,2}(The\s+)?(recommendation|decision)\s+(is\s+to\s+)?(BUY|SELL|HOLD)",
        r"^\s*\*{0,2}(TRANSACTION\s+PROPOSAL)\s*:\s*\*{0,2}\s*(BUY|SELL|HOLD)",
        r"^\s*-?\s*\*{0,2}Action\s*:\s*\*{0,2}\s*(BUY|SELL|HOLD)",
        r"^\s*>\s*\*{0,2}(DECISION|RECOMMENDATION)\s*:\s*(BUY|SELL|HOLD)",
    ]

    for line in lines:
        skip_line = False
        for pattern in skip_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                skip_line = True
                break
        if not skip_line:
            filtered_lines.append(line)

    return "\n".join(filtered_lines)


def _clean_md_cell(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^\*+|\*+$", "", text)
    return text.strip()


def _extract_historical_metrics_table(report_text: str) -> Optional[Tuple[List[int], Dict[str, List[Optional[float]]]]]:
    if not report_text:
        return None

    lines = report_text.splitlines()
    table_blocks: List[List[str]] = []
    current: List[str] = []
    for line in lines:
        if line.strip().startswith("|"):
            current.append(line.strip())
        elif current:
            table_blocks.append(current)
            current = []
    if current:
        table_blocks.append(current)

    best_years: List[int] = []
    best_series: Dict[str, List[Optional[float]]] = {}

    for block in table_blocks:
        if len(block) < 3:
            continue
        header_cells = [_clean_md_cell(x) for x in block[0].strip("|").split("|")]
        if len(header_cells) < 4:
            continue

        years: List[int] = []
        for cell in header_cells[1:]:
            m = re.search(r"(19|20)\d{2}", cell)
            years.append(int(m.group(0)) if m else -1)

        if sum(1 for y in years if y > 0) < 6:
            continue

        aligned_years = [y for y in years if y > 0]
        if len(aligned_years) < len(years):
            # Keep column alignment with None placeholders when year parsing fails.
            normalized_years = [y if y > 0 else aligned_years[-1] + 1 for y in years]
        else:
            normalized_years = years

        series: Dict[str, List[Optional[float]]] = {}
        for row in block[2:]:
            cells = [_clean_md_cell(x) for x in row.strip("|").split("|")]
            if len(cells) < 2:
                continue
            metric_name = cells[0]
            values = [_parse_metric_value(v) for v in cells[1:]]
            # Normalize length to year columns.
            if len(values) < len(normalized_years):
                values.extend([None] * (len(normalized_years) - len(values)))
            elif len(values) > len(normalized_years):
                values = values[: len(normalized_years)]
            series[metric_name] = values

        if len(series) > len(best_series):
            best_years = normalized_years
            best_series = series

    if not best_years or not best_series:
        return None
    return best_years, best_series


def _parse_metric_value(raw: str) -> Optional[float]:
    text = _clean_md_cell(raw)
    if not text:
        return None

    lower = text.lower()
    if lower in {"n/a", "na", "neg", "-", "--"}:
        return None

    negative = False
    if text.startswith("(") and text.endswith(")"):
        negative = True
        text = text[1:-1]

    text = text.replace(",", "")
    text = text.replace("$", "")
    text = text.replace("x", "")
    text = text.replace("%", "")
    text = text.replace("+", "")
    text = text.strip()

    scale = 1.0
    if text.endswith("T"):
        scale = 1000.0
        text = text[:-1]
    elif text.endswith("B"):
        scale = 1.0
        text = text[:-1]
    elif text.endswith("M"):
        scale = 0.001
        text = text[:-1]
    elif text.endswith("K"):
        scale = 0.000001
        text = text[:-1]

    m = re.search(r"-?\d*\.?\d+", text)
    if not m:
        return None
    value = float(m.group(0)) * scale
    return -value if negative else value


def _normalize_metric_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _pick_key_metrics(series: Dict[str, List[Optional[float]]]) -> List[Tuple[str, List[Optional[float]]]]:
    preferred = [
        ("Revenue", ["revenue", "revenues"]),
        ("Gross Profit", ["grossprofit"]),
        ("Net Income", ["netincome"]),
        ("Free Cash Flow", ["freecashflow"]),
        ("EPS", ["eps", "earningspershare"]),
        ("Operating Margin (%)", ["operatingmargin"]),
        ("Net Margin (%)", ["netmargin", "netincomemargin"]),
        ("ROE (%)", ["roe", "returnonequity"]),
        ("ROIC (%)", ["roic", "returnoninvestedcapital"]),
        ("Debt-to-Equity", ["debttoequity"]),
        ("P/E", ["peratio", "priceearning", "pricetoearning"]),
        ("Market Price", ["marketprice"]),
    ]

    normalized_map = {_normalize_metric_name(k): k for k in series}
    selected: List[Tuple[str, List[Optional[float]]]] = []
    for label, aliases in preferred:
        chosen_key = None
        for norm_name, original in normalized_map.items():
            if any(alias in norm_name for alias in aliases):
                chosen_key = original
                break
        if chosen_key:
            selected.append((label, series[chosen_key]))
        if len(selected) >= 8:
            break
    return selected


def _format_last_value(label: str, value: float) -> str:
    if any(x in label.lower() for x in ["margin", "roe", "roic"]):
        return f"{value:.1f}%"
    if "p/e" in label.lower() or "debt-to-equity" in label.lower():
        return f"{value:.2f}"
    if abs(value) >= 100:
        return f"{value:.0f}"
    return f"{value:.2f}"


def _plot_fundamentals_key_metrics(
    fundamentals_report: str,
    ticker: str,
    analysis_date: str,
    report_dir: Path,
) -> List[str]:
    parsed = _extract_historical_metrics_table(fundamentals_report)
    if not parsed:
        return []
    years, series = parsed
    selected = _pick_key_metrics(series)
    if not selected:
        return []

    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    images_dir = report_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), dpi=160, facecolor="#f8fafc")
    axes_flat = axes.flatten()
    colors = ["#0f766e", "#2563eb", "#0891b2", "#7c3aed", "#ea580c", "#16a34a", "#b45309", "#4f46e5"]

    for idx, ax in enumerate(axes_flat):
        if idx >= len(selected):
            ax.axis("off")
            continue

        label, values = selected[idx]
        x_vals: List[int] = []
        y_vals: List[float] = []
        for x, y in zip(years, values):
            if isinstance(y, (int, float)):
                x_vals.append(x)
                y_vals.append(float(y))
        if len(x_vals) < 2:
            ax.axis("off")
            continue

        color = colors[idx % len(colors)]
        ax.plot(x_vals, y_vals, color=color, linewidth=2.8, marker="o", markersize=4.8)
        ax.fill_between(x_vals, y_vals, [min(y_vals)] * len(y_vals), color=color, alpha=0.08)
        ax.set_title(label, fontsize=13, weight="bold", color="#0f172a")
        ax.grid(alpha=0.25, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelrotation=30, labelsize=9)
        ax.tick_params(axis="y", labelsize=9)

        latest_x = x_vals[-1]
        latest_y = y_vals[-1]
        ax.scatter([latest_x], [latest_y], color=color, s=42, zorder=3)
        ax.annotate(
            _format_last_value(label, latest_y),
            xy=(latest_x, latest_y),
            xytext=(6, 8),
            textcoords="offset points",
            fontsize=9,
            color="#111827",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "#ffffff", "edgecolor": "#d1d5db", "alpha": 0.85},
        )

    fig.suptitle(
        f"{ticker} Fundamentals Key Metrics (10-Year Trend) • {analysis_date}",
        fontsize=18,
        weight="bold",
        color="#0b1324",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    image_name = f"{ticker}_fundamentals_key_metrics_{analysis_date}.png"
    image_path = images_dir / image_name
    fig.savefig(image_path, bbox_inches="tight")
    plt.close(fig)

    return [image_path.resolve().as_posix()]


def _normalize_chart_path_for_report(chart_path: str) -> str:
    """
    Convert chart paths to markdown-friendly report-local paths.
    We store images under report_dir/images, so `images/<name>` is stable.
    """
    if not chart_path:
        return chart_path
    if chart_path.startswith("images/"):
        return chart_path
    p = Path(chart_path)
    if p.is_absolute():
        return f"images/{p.name}"
    return chart_path


def _generate_comprehensive_report(final_state, ticker, analysis_date, fundamentals_chart_paths: Optional[List[str]] = None) -> str:
    """Generate a comprehensive markdown report matching the CLI format."""
    report_lines = []

    # Header
    report_lines.append(f"# {ticker} Deep Value Intelligence")
    report_lines.append("")
    report_lines.append(f"**Analysis Date:** {analysis_date}")
    report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    # Table of Contents
    report_lines.append("## Table of Contents")
    report_lines.append("")
    report_lines.append("1. [Intelligence Summary](#intelligence-summary)")
    report_lines.append("2. [Analyst Team Reports](#analyst-team-reports)")
    report_lines.append("   - [Fundamentals Analysis](#fundamentals-analysis)")
    report_lines.append("   - [Value Analysis (Buffett)](#value-analysis-buffett)")
    report_lines.append("   - [Growth Analysis (Lynch/Druckenmiller/Fisher)](#growth-analysis-lynchdruckenmillerfisher)")
    report_lines.append("   - [Market Analysis](#market-analysis)")
    report_lines.append("   - [Social Sentiment Analysis](#social-sentiment-analysis)")
    report_lines.append("   - [News Analysis](#news-analysis)")
    report_lines.append("3. [Research Team Decision](#research-team-decision)")
    report_lines.append("4. [Trading Team Plan](#trading-team-plan)")
    report_lines.append("5. [Deep Values Strategy](#deep-values-strategy)")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    # Intelligence Summary
    report_lines.append("## Intelligence Summary")
    report_lines.append("")
    if final_state.get("final_trade_decision"):
        decision_content = _ensure_string(final_state["final_trade_decision"])
        filtered_content = _filter_trading_recommendations(decision_content)
        report_lines.append(filtered_content)
    else:
        report_lines.append("*No executive summary available.*")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    # Analyst Team Reports
    report_lines.append("## Analyst Team Reports")
    report_lines.append("")

    report_lines.append("### Fundamentals Analysis")
    report_lines.append("")
    if final_state.get("fundamentals_report"):
        report_lines.append(_ensure_string(final_state["fundamentals_report"]))
    else:
        report_lines.append("*No fundamentals analysis available.*")
    if fundamentals_chart_paths:
        report_lines.append("")
        report_lines.append("#### Fundamentals Key Metrics Charts")
        report_lines.append("")
        for chart_path in fundamentals_chart_paths:
            report_path = _normalize_chart_path_for_report(chart_path)
            report_lines.append(f"![Fundamentals Key Metrics]({report_path})")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    report_lines.append("### Value Analysis (Buffett)")
    report_lines.append("")
    report_lines.append("> *Following Warren Buffett's value investing philosophy: Circle of Competence, Economic Moat, Management Quality, and Margin of Safety.*")
    report_lines.append("")
    if final_state.get("value_report"):
        report_lines.append(_ensure_string(final_state["value_report"]))
    else:
        report_lines.append("*No value analysis available.*")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    report_lines.append("### Growth Analysis (Lynch/Druckenmiller/Fisher)")
    report_lines.append("")
    report_lines.append("> *Following Peter Lynch's GARP, Stanley Druckenmiller's macro-aware approach, and Philip Fisher's scuttlebutt methodology.*")
    report_lines.append("")
    if final_state.get("growth_report"):
        report_lines.append(_ensure_string(final_state["growth_report"]))
    else:
        report_lines.append("*No growth analysis available.*")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    report_lines.append("### Market Analysis")
    report_lines.append("")
    if final_state.get("market_report"):
        report_lines.append(_ensure_string(final_state["market_report"]))
    else:
        report_lines.append("*No market analysis available.*")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    report_lines.append("### Social Sentiment Analysis")
    report_lines.append("")
    if final_state.get("sentiment_report"):
        report_lines.append(_ensure_string(final_state["sentiment_report"]))
    else:
        report_lines.append("*No sentiment analysis available.*")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    report_lines.append("### News Analysis")
    report_lines.append("")
    if final_state.get("news_report"):
        report_lines.append(_ensure_string(final_state["news_report"]))
    else:
        report_lines.append("*No news analysis available.*")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    # Research Team Decision
    report_lines.append("## Research Team Decision")
    report_lines.append("")
    report_lines.append("> *The Research Manager synthesizes the Bull vs Bear debate to provide an AI strategy.*")
    report_lines.append("")
    if final_state.get("investment_plan"):
        report_lines.append(_ensure_string(final_state["investment_plan"]))
    else:
        report_lines.append("*No research team decision available.*")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    # Trading Team Plan
    report_lines.append("## Trading Team Plan")
    report_lines.append("")
    report_lines.append("> *The Trader creates a specific investment plan based on the research team's analysis.*")
    report_lines.append("")
    if final_state.get("trader_investment_plan"):
        report_lines.append(_ensure_string(final_state["trader_investment_plan"]))
    else:
        report_lines.append("*No trading plan available.*")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    # Deep Values Strategy
    report_lines.append("## Deep Values Strategy")
    report_lines.append("")
    report_lines.append("> *The Risk Management Team (Aggressive, Conservative, Neutral) debates and the Portfolio Manager provides the final strategy.*")
    report_lines.append("")
    if final_state.get("final_trade_decision"):
        report_lines.append(_ensure_string(final_state["final_trade_decision"]))
    else:
        report_lines.append("*No AI investing strategy available.*")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    # Footer
    report_lines.append("## Disclaimer")
    report_lines.append("")
    report_lines.append("*This analysis is generated by Deep Value Intelligence, an AI-powered multi-agent trading framework. ")
    report_lines.append("This report is for informational and educational purposes only and does not constitute financial advice. ")
    report_lines.append("Always conduct your own research and consult with qualified financial advisors before making investment decisions.*")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append(f"*Report generated by Deep Value Intelligence on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

    return "\n".join(report_lines)


def main() -> None:
    args = _build_parser().parse_args()

    # Create a custom config
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = args.provider
    config["backend_url"] = args.backend_url
    config["deep_think_llm"] = args.deep_model
    config["quick_think_llm"] = args.quick_model
    config["max_debate_rounds"] = args.max_debate_rounds
    config["llm_timeout"] = args.llm_timeout
    config["llm_max_retries"] = args.llm_max_retries
    config["online_tools"] = args.online_tools
    if args.embedding_provider:
        config["embedding_provider"] = args.embedding_provider
    if args.embedding_model:
        config["embedding_model"] = args.embedding_model

    # Initialize with custom config
    ta = InvestingAgentsGraph(debug=args.debug, config=config)

    # forward propagate
    final_state, decision = ta.propagate(args.ticker, args.date)

    print(decision)

    results_dir = Path(config["results_dir"]) / args.ticker / args.date
    report_dir = results_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    fundamentals_chart_paths: List[str] = []
    if args.plot_fundamentals and final_state.get("fundamentals_report"):
        fundamentals_chart_paths = _plot_fundamentals_key_metrics(
            _ensure_string(final_state["fundamentals_report"]),
            args.ticker,
            args.date,
            report_dir,
        )
        if fundamentals_chart_paths:
            print(f"✅ Fundamentals charts saved: {', '.join(fundamentals_chart_paths)}")

    # Save the final markdown report to the standard log folder (CLI format)
    report_md = _generate_comprehensive_report(
        final_state,
        args.ticker,
        args.date,
        fundamentals_chart_paths=fundamentals_chart_paths,
    )
    report_path = report_dir / f"{args.ticker}_deep_value_intelligence_{args.date}.md"
    report_path.write_text(report_md, encoding="utf-8")

    if args.generate_thumbnail:
        try:
            from utils.thumbnail import generate_thumbnails
            company_label = final_state.get("company_of_interest", args.ticker)
            saved = generate_thumbnails(
                company_label,
                report_dir,
                api_key=config.get("google_api_key"),
                ticker=args.ticker,
                analysis_date=args.date,
            )
            if saved:
                print(f"✅ Thumbnails saved: {', '.join(str(p) for p in saved)}")
        except Exception as exc:
            print(f"⚠️  Thumbnail generation failed: {exc}")


if __name__ == "__main__":
    main()

# Memorize mistakes and reflect
# ta.reflect_and_remember(1000) # parameter is the position returns
