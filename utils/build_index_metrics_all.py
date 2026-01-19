#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_index_metrics_all_v14d.py

What's new in v14d:
- Company CSVs are named:  TICKER-CompanyName.csv  (no extra column added)
- ETF CSVs are named:      TICKER-FundName.csv     (falls back to SEC entity if needed)

Everything else from prior versions remains:
- Universes: --sp500 / --r2000 / --nasdaq100 / --usmarket (+ --include-etfs, --include-otc)
- Latest year uses TTM by default (disable via --no-ttm-for-latest or set --ttm-date)
- Separate ETF folder via --etf-outdir, deep/regular validation, cap ordering, etc.

Reqs:
  pip install pandas requests python-dateutil beautifulsoup4 lxml
Env:
  export SEC_EMAIL="you@example.com"
"""
import os, io, csv, sys, time, json, zipfile, argparse, random, re
from datetime import datetime, timedelta
from dateutil import parser as dtp
from io import StringIO

import requests
import pandas as pd
from bs4 import BeautifulSoup  # pip install beautifulsoup4 lxml

# ---------------- Endpoints ----------------
SEC_TICKER_MAP_URL   = "https://www.sec.gov/files/company_tickers.json"
SEC_FACTS_URL_TMPL   = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SEC_CONCEPT_URL_TMPL = "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/{tax}/{tag}.json"
STOOQ_DL_URL_TMPL    = "https://stooq.com/q/d/l/?s={symbol}&i=d"

SLICKCHARTS_URL_SP500     = "https://www.slickcharts.com/sp500"
SLICKCHARTS_URL_NASDAQ100 = "https://www.slickcharts.com/nasdaq100"
ISHARES_IVV_HOLDINGS      = ("https://www.ishares.com/us/products/239726/ishares-core-sp-500-etf/"
                             "1467271812596.ajax?fileType=csv&fileName=IVV_holdings&dataType=fund")
WIKI_URL_SP500       = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
WIKI_URL_NASDAQ100   = "https://en.wikipedia.org/wiki/Nasdaq-100"
ISHARES_IWM_HOLDINGS = ("https://www.ishares.com/us/products/239710/ishares-russell-2000-etf/"
                        "1467271812596.ajax?fileType=csv&fileName=IWM_holdings&dataType=fund")

NASDAQ_TRADED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt"
OTHER_LISTED_URL  = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

# ---------------- SEC headers ----------------
SEC_EMAIL = os.getenv("SEC_EMAIL", "").strip()
SEC_HEADERS = {
    "User-Agent": f"Index-Metrics-Script ({SEC_EMAIL})" if SEC_EMAIL else "Index-Metrics-Script (no-email-provided)",
    "Accept": "application/json",
}
WEB_HEADERS = {"User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/122 Safari/537.36"}

# ---------------- Cache ----------------
CACHE_DIR = "./cache"
CACHE_SEC = os.path.join(CACHE_DIR, "sec_facts")
CACHE_PRC = os.path.join(CACHE_DIR, "stooq")
os.makedirs(CACHE_SEC, exist_ok=True)
os.makedirs(CACHE_PRC, exist_ok=True)

# Stooq symbol overrides
STOOQ_OVERRIDES = {"BRK.B":"brk-b","BRK.A":"brk-a","BF.B":"bf-b","GOOGL":"googl","GOOG":"goog"}

# ---------------- Company column sets (core + extra) ----------------
CORE_COLS = [
    "Market Price (USD)",
    "Book Value per Share (USD)",
    "Total Number of Shares",
    "Price to Earning ratio (P/E)",
    "Price to Book ratio (P/B)",
    "Total Market Cap (Millions USD)",
    "Total Assets (Millions USD)",
    "Total Liabilities (Millions USD)",
    "Book Value (Millions USD)",
    "Cash and cash equivalents (Millions USD)",
    "Revenue (Millions USD)",
    "Revenue Growth Rate",
    "Gross profit (Millions USD)",
    "Gross Margin (%)",
    "Operating Profit (Millions USD)",
    "Operation Margin (%)",
    "Net Income(Millions USD)",
    "Net Income Growth Rate (%)",
    "Net income margin (%)",
    "Free Cash Flow (Millions USD)",
    "Free Cash Flow Margin (%)",
    "Capital Expenditure (Millions USD)",
    "Maintenance Capital Expenditure (Millions USD)",
    "Owner’s Earnings (Millions USD)",
    "Owner’s Earnings Margin (%)",
    "Retained Earnings (Millions USD)",
    "Retained Earnings Growth Rate (%)",
    "Return on Retained Earnings (%)",
    "Return on Invested Capital (%)",
    "Return on Equity (%)",
    "Return on Assets (%)",
    "Owner’s Earnings (Millions USD)",
    "Owner’s Earnings / Total Market Cap (%)",
    "Shareholder’s Equity (Millions USD)",
    "Shareholder’s Equity / Total Market Cap (%)",
    "Interest Expense / Operating Income (%)",
    "Current Ratio",
    "Debt to equity ratio",
    "Debt to asset ratio",
    "Depreciation / Gross Profit (%)",
]
EXTRA_COLS = [
    "Earnings Per Share – Diluted (USD)",
    "Operating Cash Flow (Millions USD)",
    "Operating Cash Flow Margin (%)",
    "Depreciation & Amortization (Millions USD)",
    "EBITDA (Millions USD)",
    "EBITDA Margin (%)",
    "Effective Tax Rate (%)",
    "Total Debt (Millions USD)",
    "Net Debt (Millions USD)",
    "Enterprise Value (Millions USD)",
    "EV / EBITDA",
    "EV / Sales",
    "Price to Sales ratio (P/S)",
    "Price to Free Cash Flow (P/FCF)",
    "Dividend per Share (USD)",
    "Dividend Yield (%)",
    "Dividend Payout Ratio (%)",
    "Share Repurchases (Millions USD)",
    "Share-Based Compensation (Millions USD)",
    "Research & Development (Millions USD)",
    "Selling, General & Administrative (Millions USD)",
    "Goodwill & Intangibles (Millions USD)",
    "Tangible Book Value (Millions USD)",
    "Tangible Book Value per Share (USD)",
    "Working Capital (Millions USD)",
    "Quick Ratio",
    "Inventory (Millions USD)",
    "Interest Coverage (x)",
    "Net Debt / EBITDA",
    "Debt / EBITDA",
    "Capex / Revenue (%)",
    "Cash Conversion (%)",
    "Share Count Change (% YoY)",
    "Weighted Average Diluted Shares",
]

# ---------------- ETF column set ----------------
ETF_COLS = [
    "Ticker","Fund Name","CIK","Issuer (Entity)","Exchange",
    "Inception Date","CUSIP/ISIN","Benchmark",
    "Total Net Assets (USD)","Shares Outstanding","NAV (USD)","Last Close (USD)",
    "Premium/Discount (%)","30-Day Median Bid/Ask (%)",
    "Expense Ratio (Net, %)","Expense Ratio (Gross, %)",
    "30-Day SEC Yield (%)","Distribution Yield (TTM, %)","Distribution Frequency",
    "1Y Return (NAV, %)","1Y Return (Price, %)",
    "3Y Return (NAV, %)","3Y CAGR (Price, %)",
    "5Y Return (NAV, %)","5Y CAGR (Price, %)",
    "10Y Return (NAV, %)","10Y CAGR (Price, %)","Since Inception Return (NAV, %)",
    "Since Inception CAGR (Price, %)","Number of Holdings","Top 10 Concentration (%)","As Of Date"
]

# ---------------- Tag sets ----------------
TAGSETS = {
    "revenue": ["Revenues","RevenueFromContractWithCustomerExcludingAssessedTax","SalesRevenueNet"],
    "cost_of_revenue": ["CostOfRevenue","CostOfGoodsAndServicesSold","CostOfGoodsSold","CostOfServices"],
    "gross_profit": ["GrossProfit"],
    "oper_income":  ["OperatingIncomeLoss"],
    "net_income":   ["NetIncomeLoss"],
    "assets": ["Assets"],
    "liab":   ["Liabilities"],
    "equity": ["StockholdersEquity","StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest","CommonStockholdersEquity"],
    "cash_eq": ["CashAndCashEquivalentsAtCarryingValue"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities","NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "retained": ["RetainedEarningsAccumulatedDeficit"],
    "eps_diluted": ["EarningsPerShareDiluted","EarningsPerShareBasicAndDiluted"],
    "shares_out": ["CommonStockSharesOutstanding","EntityCommonStockSharesOutstanding"],
    "shares_avg": ["WeightedAverageNumberOfDilutedSharesOutstanding","WeightedAverageNumberOfSharesOutstandingDiluted","WeightedAverageNumberOfSharesOutstandingBasic","WeightedAverageNumberOfSharesOutstanding"],
    "current_assets":["AssetsCurrent"],
    "current_liab": ["LiabilitiesCurrent"],
    "interest_expense":["InterestExpense","InterestExpenseDebt"],
    "pbt": ["IncomeBeforeIncomeTaxes","IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest"],
    "tax_expense": ["IncomeTaxExpenseBenefit"],
    "ddna": ["DepreciationDepletionAndAmortization","DepreciationAndAmortization"],
    "div_per_share":["CommonStockDividendsPerShareDeclared","CommonStockDividendsPerShareCashPaid"],
    "div_cash": ["PaymentsOfDividends","PaymentsOfDividendsCommonStock"],
    "buybacks": ["PaymentsForRepurchaseOfCommonStock"],
    "sbc": ["ShareBasedCompensation"],
    "rnd": ["ResearchAndDevelopmentExpense"],
    "sga": ["SellingGeneralAndAdministrativeExpense"],
    "goodwill": ["Goodwill"],
    "intang_net_ex":["IntangibleAssetsNetExcludingGoodwill","FiniteLivedIntangibleAssetsNet"],
    "intang_net":   ["IntangibleAssetsNet"],
    "short_term_inv":["ShortTermInvestments","MarketableSecuritiesCurrent"],
    "ar_current":   ["AccountsReceivableNetCurrent"],
    "inventory":    ["InventoryNet","Inventory"],
}

# ---------------- Utils ----------------
def log(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
def to_float(x):
    try: return float(x)
    except: return None
def safe_div(a,b):
    try:
        if b in (0,None) or a is None: return None
        return a/b
    except: return None
def pct(a,b): v=safe_div(a,b); return None if v is None else v*100.0
def yoy(curr,prev):
    if curr is None or prev in (None,0): return None
    return (curr - prev)/abs(prev)*100.0
def M(x): return None if x is None else x/1_000_000.0
def as_int(x): return None if x is None else int(round(x))
def safe_name_component(name, maxlen=80):
    # Keep only letters/numbers; drop everything else (spaces, punctuation)
    s = re.sub(r"[^A-Za-z0-9]+", "", (name or ""))
    s = s[:maxlen] if s else "UNKNOWN"
    return s

# ---------------- HTTP/backoff + safe JSON ----------------
def http_get(url, headers=None, timeout=60, max_retries=4, base_sleep=0.6):
    for i in range(max_retries+1):
        r = requests.get(url, headers=headers, timeout=timeout)
        if r.status_code in (429, 503):
            time.sleep(base_sleep * (2**i) + random.uniform(0,0.2)); continue
        if r.status_code == 404:
            return r
        try: r.raise_for_status()
        except Exception:
            if i < max_retries:
                time.sleep(base_sleep * (2**i) + random.uniform(0,0.2)); continue
            raise
        return r
    return r
def http_get_json(url, headers=None, timeout=60, max_retries=4, base_sleep=0.6):
    r = http_get(url, headers=headers, timeout=timeout, max_retries=max_retries, base_sleep=base_sleep)
    if r.status_code == 404: return None
    try: return r.json()
    except Exception: return None

# ---------------- Universe helpers ----------------
def _read_pipe_table(url):
    r = requests.get(url, timeout=60, headers=WEB_HEADERS); r.raise_for_status()
    txt = r.text
    df = pd.read_csv(StringIO(txt), sep="|")
    for col in ("Symbol","ACT Symbol","CQS Symbol","NASDAQ Symbol"):
        if col in df.columns:
            df = df[df[col].astype(str).str.upper() != "FILE CREATION TIME"]
    return df
def _pick_symbol_col(df):
    for col in ("Symbol","ACT Symbol","NASDAQ Symbol","CQS Symbol"):
        if col in df.columns: return col
    return df.columns[0]

def fetch_sp500_tickers_slickcharts():
    try:
        r = requests.get(SLICKCHARTS_URL_SP500, timeout=30, headers=WEB_HEADERS); r.raise_for_status()
        html = r.text
        try:
            dfs = pd.read_html(StringIO(html)); df = dfs[0]
            if "Symbol" in df.columns:
                return [s.upper() for s in df["Symbol"].astype(str).str.strip().tolist() if s]
        except Exception: pass
        m = re.findall(r"/symbol/([A-Za-z\.\-]+)", html)
        if m:
            seen=set(); out=[]
            for s in m:
                s=s.upper()
                if s not in seen: seen.add(s); out.append(s)
            return out
    except Exception: pass
    return None

def fetch_sp500_tickers_ivv(as_of=None):
    url = ISHARES_IVV_HOLDINGS + (f"&asOfDate={as_of}" if as_of else "")
    r = requests.get(url, timeout=60, headers=WEB_HEADERS); r.raise_for_status()
    rows = list(csv.reader(StringIO(r.text)))
    header_idx=None
    for i,row in enumerate(rows):
        if row and row[0].strip().lower()=="ticker":
            header_idx=i; break
    if header_idx is None:
        df = pd.read_csv(StringIO(r.text), skiprows=9, engine="python", on_bad_lines="skip")
    else:
        buf=StringIO(); w=csv.writer(buf)
        for rr in rows[header_idx:]:
            if not rr: break
            w.writerow(rr)
        buf.seek(0)
        df = pd.read_csv(buf, engine="python", on_bad_lines="skip")
    tk_col = "Ticker" if "Ticker" in df.columns else df.columns[0]
    df = df.rename(columns={tk_col:"Ticker"})
    if "Weight (%)" in df.columns:
        df["Weight (%)"] = pd.to_numeric(df["Weight (%)"], errors="coerce")
        df = df.sort_values("Weight (%)", ascending=False)
    out=[]; seen=set()
    for t in df["Ticker"].astype(str).str.strip():
        t=t.upper()
        if t and t!="CASH_USD" and t not in seen:
            seen.add(t); out.append(t)
    return out

def fetch_sp500_tickers_wiki():
    r = requests.get(WIKI_URL_SP500, timeout=30, headers=WEB_HEADERS); r.raise_for_status()
    dfs = pd.read_html(StringIO(r.text)); df = dfs[0]
    col = "Symbol" if "Symbol" in df.columns else df.columns[0]
    return [s.upper() for s in df[col].astype(str).str.strip().tolist() if s]

def fetch_r2000_tickers_from_ishares(as_of=None):
    url = ISHARES_IWM_HOLDINGS + (f"&asOfDate={as_of}" if as_of else "")
    r = requests.get(url, timeout=60, headers=WEB_HEADERS); r.raise_for_status()
    rows = list(csv.reader(StringIO(r.text)))
    header_idx=None
    for i,row in enumerate(rows):
        if row and row[0].strip().lower()=="ticker":
            header_idx=i; break
    if header_idx is None:
        df = pd.read_csv(StringIO(r.text), skiprows=9, engine="python", on_bad_lines="skip")
    else:
        buf=StringIO(); w=csv.writer(buf)
        for rr in rows[header_idx:]:
            if not rr: break
            w.writerow(rr)
        buf.seek(0)
        df = pd.read_csv(buf, engine="python", on_bad_lines="skip")
    tk_col = "Ticker" if "Ticker" in df.columns else df.columns[0]
    df = df.rename(columns={tk_col:"Ticker"})
    if "Asset Class" in df.columns:
        df = df[df["Asset Class"].astype(str).str.upper().str.contains("EQUITY", na=False)]
    if "Weight (%)" in df.columns:
        df["Weight (%)"] = pd.to_numeric(df["Weight (%)"], errors="coerce")
        df = df.sort_values("Weight (%)", ascending=False)
    out=[]; seen=set()
    for t in df["Ticker"].astype(str).str.strip():
        t=t.upper()
        if t and t!="CASH_USD" and t not in seen:
            seen.add(t); out.append(t)
    return out

def fetch_nasdaq100_tickers_slickcharts():
    try:
        r = requests.get(SLICKCHARTS_URL_NASDAQ100, timeout=30, headers=WEB_HEADERS); r.raise_for_status()
        html = r.text
        try:
            dfs = pd.read_html(StringIO(html)); df = dfs[0]
            if "Symbol" in df.columns:
                return [s.upper() for s in df["Symbol"].astype(str).str.strip().tolist() if s]
        except Exception: pass
        m = re.findall(r"/symbol/([A-Za-z\.\-]+)", html)
        if m:
            seen=set(); out=[]
            for s in m:
                s=s.upper()
                if s not in seen: seen.add(s); out.append(s)
            return out
    except Exception: pass
    return None

def fetch_nasdaq100_tickers_wiki():
    r = requests.get(WIKI_URL_NASDAQ100, timeout=30, headers=WEB_HEADERS); r.raise_for_status()
    dfs = pd.read_html(StringIO(r.text))
    for df in dfs:
        cand_col = None
        for c in df.columns:
            lc = str(c).lower()
            if "ticker" in lc or "symbol" in lc:
                cand_col = c; break
        if cand_col is not None:
            syms = [s.upper().strip() for s in df[cand_col].astype(str).tolist()]
            syms = [s for s in syms if s and s != "N/A"]
            if syms: return syms
    return None

def fetch_usmarket_tables(include_etfs=False):
    nas = _read_pipe_table(NASDAQ_TRADED_URL)
    oth = _read_pipe_table(OTHER_LISTED_URL)
    def normalize(df, sym_col):
        df = df.copy()
        df["__SYM__"] = df[sym_col].astype(str).str.strip().str.upper()
        if "ETF" not in df.columns: df["ETF"] = "N"
        return df[["__SYM__","ETF","Listing Exchange"]] if "Listing Exchange" in df.columns else df[["__SYM__","ETF"]]
    nas_scol = _pick_symbol_col(nas)
    oth_scol = _pick_symbol_col(oth)
    nas = normalize(nas, nas_scol)
    oth = normalize(oth, oth_scol)
    comb = pd.concat([nas, oth], ignore_index=True).drop_duplicates("__SYM__", keep="first")
    if not include_etfs:
        comb = comb[comb["ETF"].astype(str).str.upper()=="N"]
    return comb

# ---------------- SEC + Prices ----------------
def fetch_sec_ticker_map(sleep=0.6):
    data = http_get_json(SEC_TICKER_MAP_URL, headers=SEC_HEADERS, timeout=30, base_sleep=sleep)
    if not data: return {}
    return { obj["ticker"].upper(): f"{int(obj['cik_str']):010d}" for _, obj in data.items() }

def fetch_sec_company_facts(cik, sleep=0.6):
    if not cik: return None
    path = os.path.join(CACHE_SEC, f"CIK{cik}.json")
    if os.path.exists(path):
        try:
            with open(path,"r") as f: return json.load(f)
        except Exception: pass
    url = SEC_FACTS_URL_TMPL.format(cik=cik)
    data = http_get_json(url, headers=SEC_HEADERS, timeout=60, base_sleep=sleep)
    if not data: return None
    try:
        with open(path,"w") as f: json.dump(data, f)
    except Exception: pass
    time.sleep(sleep)
    return data

def stooq_symbol(ticker):
    t = ticker.upper()
    base = STOOQ_OVERRIDES.get(t, t).lower()
    return f"{base}.us"

def fetch_stooq_prices(ticker):
    path = os.path.join(CACHE_PRC, f"{ticker.upper()}.csv")
    if os.path.exists(path):
        try: return pd.read_csv(path)
        except Exception: pass
    url = STOOQ_DL_URL_TMPL.format(symbol=stooq_symbol(ticker))
    r = requests.get(url, timeout=30, headers=WEB_HEADERS)
    if r.status_code != 200 or not r.text.strip(): return None
    try:
        df = pd.read_csv(StringIO(r.text))
        df.columns = [c.strip().lower() for c in df.columns]
        df.to_csv(path, index=False)
        return df
    except Exception:
        return None

# ---------------- XBRL helpers ----------------
def _facts_ns(company_facts, ns): return (company_facts or {}).get("facts", {}).get(ns, {})
def pick_fy_fact(company_facts, tags, fy):
    facts = _facts_ns(company_facts, "us-gaap")
    for tag in tags:
        node = facts.get(tag)
        if not node: continue
        for unit, arr in node.get("units", {}).items():
            for it in arr:
                if str(it.get("fy")) == str(fy) and it.get("fp") == "FY" and it.get("form") in ("10-K","20-F"):
                    return (to_float(it.get("val")), it.get("end"))
    return (None, None)

def _collect_instant_candidates(company_facts, tag):
    out=[]
    for ns in ("us-gaap","dei"):
        node = _facts_ns(company_facts, ns).get(tag)
        if not node: continue
        for unit, arr in node.get("units", {}).items():
            for it in arr:
                form = it.get("form")
                if form not in ("10-K","10-Q","20-F"):  continue
                end = it.get("end") or ""
                val = to_float(it.get("val"))
                if end and val is not None:
                    out.append((end, val, form))
    out.sort(key=lambda x: x[0])
    return out

def pick_latest_instant(company_facts, tags, asof=None):
    cands=[]
    for tag in tags:
        cands += _collect_instant_candidates(company_facts, tag)
    if not cands: return (None, None)
    if asof:
        upto = [c for c in cands if c[0] <= asof]
        if upto:
            return upto[-1][1], upto[-1][0]
    return cands[-1][1], cands[-1][0]

def pick_latest_shares_asof(company_facts, asof=None):
    return pick_latest_instant(company_facts, TAGSETS["shares_out"], asof)

def pick_weighted_avg_shares(company_facts, fy):
    facts = _facts_ns(company_facts, "us-gaap")
    for tag in TAGSETS["shares_avg"]:
        node = facts.get(tag)
        if not node: continue
        for unit, arr in node.get("units", {}).items():
            for it in arr:
                if it.get("form") in ("10-K","20-F") and str(it.get("fy")) == str(fy):
                    return to_float(it.get("val"))
    return None

def fetch_sec_company_concept(cik, tax, tag, sleep=0.6):
    url = SEC_CONCEPT_URL_TMPL.format(cik=cik, tax=tax, tag=tag)
    return http_get_json(url, headers=SEC_HEADERS, timeout=60, base_sleep=sleep)

def get_concept_quarter_series(cik, tags, sleep=0.6, prefer_tax="us-gaap"):
    out=[]
    for tag in tags:
        data = fetch_sec_company_concept(cik, prefer_tax, tag, sleep=sleep)
        if not data or "units" not in data: continue
        for unit, arr in data["units"].items():
            for it in arr:
                fp = it.get("fp"); form = it.get("form"); end = it.get("end")
                val = to_float(it.get("val"))
                if form not in ("10-Q","10-K","20-F"): continue
                if fp not in ("Q1","Q2","Q3","Q4"): continue
                if end and val is not None:
                    out.append((end, val))
    out.sort(key=lambda x: x[0])
    return out

def sum_last_n_quarters_from_series(series, n=4, asof=None):
    if not series: return (None, None, 0)
    s = [it for it in series if (asof is None or it[0] <= asof)]
    if not s: return (None, None, 0)
    last = s[-n:] if len(s) >= n else s
    total = sum(v for _, v in last)
    end_date = last[-1][0]
    return (total, end_date, len(last))

def _collect_duration_quarters(company_facts, tags):
    out=[]
    facts = _facts_ns(company_facts, "us-gaap")
    for tag in tags:
        node = facts.get(tag)
        if not node: continue
        for unit, arr in node.get("units", {}).items():
            for it in arr:
                fp = it.get("fp"); form = it.get("form")
                if form not in ("10-Q","10-K","20-F"):  continue
                if fp not in ("Q1","Q2","Q3","Q4"):     continue
                end = it.get("end") or ""; val = to_float(it.get("val"))
                if end and val is not None:
                    out.append((end, val))
    out.sort(key=lambda x: x[0])
    return out

def sum_last_quarters(company_facts, tags, asof=None, need=4):
    q = _collect_duration_quarters(company_facts, tags)
    if asof:
        q = [it for it in q if it[0] <= asof]
    if not q: return (None, None, 0)
    last = q[-need:] if len(q) >= need else q
    return (sum(v for _, v in last), last[-1][0], len(last))

# ---------------- Price helpers ----------------
def nearest_trading_close(prices_df, target_date):
    if prices_df is None or prices_df.empty or not target_date: return None
    df = prices_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    t = pd.to_datetime(target_date)
    prev = df[df["date"] <= t]
    if prev.shape[0] > 0: return float(prev["close"].iloc[-1])
    nxt = df[df["date"] > t]
    if nxt.shape[0] > 0: return float(nxt["close"].iloc[0])
    return float(df["close"].iloc[-1])

def compute_price_returns(prices_df, as_of=None):
    res = {"as_of": None, "r_1y": None, "cagr_3y": None, "cagr_5y": None, "cagr_10y": None, "cagr_si": None}
    if prices_df is None or prices_df.empty: return res
    df = prices_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    if as_of:
        df = df[df["date"] <= pd.to_datetime(as_of)]
        if df.empty: return res
    end_date = df["date"].iloc[-1]; end = float(df["close"].iloc[-1])
    res["as_of"] = end_date.strftime("%Y-%m-%d")
    def find_date(years_back):
        cutoff = end_date - pd.DateOffset(years=years_back)
        prior = df[df["date"] <= cutoff]
        return (float(prior["close"].iloc[-1]), prior["date"].iloc[-1]) if not prior.empty else (None, None)
    p1, d1 = find_date(1)
    if p1 is not None and p1 != 0: res["r_1y"] = (end/p1 - 1.0)*100.0
    for k, yrs in (("cagr_3y",3),("cagr_5y",5),("cagr_10y",10)):
        p, d = find_date(yrs)
        if p is not None and p > 0:
            res[k] = ((end/p)**(1/yrs) - 1.0)*100.0
    start = float(df["close"].iloc[0]); start_date = df["date"].iloc[0]
    yrs = max((end_date - start_date).days/365.25, 0.01)
    if start > 0 and yrs > 0:
        res["cagr_si"] = ((end/start)**(1/yrs) - 1.0)*100.0
    return res

# ---------------- Core computations (FY & TTM) ----------------
def _compute_common_balances_for_instant(company_facts, asof_date):
    assets,_ = pick_latest_instant(company_facts, TAGSETS["assets"], asof=asof_date)
    liab,  _ = pick_latest_instant(company_facts, TAGSETS["liab"],   asof=asof_date)
    equity,_ = pick_latest_instant(company_facts, TAGSETS["equity"], asof=asof_date)
    cash,   _ = pick_latest_instant(company_facts, TAGSETS["cash_eq"], asof=asof_date)
    cur_assets,_ = pick_latest_instant(company_facts, TAGSETS["current_assets"], asof=asof_date)
    cur_liab,  _ = pick_latest_instant(company_facts, TAGSETS["current_liab"],  asof=asof_date)
    re_earn,   _ = pick_latest_instant(company_facts, TAGSETS["retained"],      asof=asof_date)
    goodwill,_ = pick_latest_instant(company_facts, TAGSETS["goodwill"], asof=asof_date)
    int_ex,_   = pick_latest_instant(company_facts, TAGSETS["intang_net_ex"], asof=asof_date)
    int_net,_  = pick_latest_instant(company_facts, TAGSETS["intang_net"], asof=asof_date)
    st_inv,_   = pick_latest_instant(company_facts, TAGSETS["short_term_inv"], asof=asof_date)
    ar,_       = pick_latest_instant(company_facts, TAGSETS["ar_current"], asof=asof_date)
    inventory,_= pick_latest_instant(company_facts, TAGSETS["inventory"], asof=asof_date)

    debt_tags = ["LongTermDebtNoncurrent","LongTermDebtCurrent","DebtCurrent","ShortTermBorrowings","CommercialPaper"]
    total_debt=None; agg=0.0; found=False
    for tag in debt_tags:
        v,_ = pick_latest_instant(company_facts,[tag],asof=asof_date)
        if v is not None: agg += max(0.0, v); found=True
    total_debt = agg if found else None

    gi = None
    if goodwill is not None:
        gi = goodwill + (int_ex if int_ex is not None else (int_net if int_net is not None else 0.0))
    else:
        gi = (int_ex if int_ex is not None else int_net)

    return {
        "assets": assets, "liab": liab, "equity": equity, "cash": cash,
        "current_assets": cur_assets, "current_liab": cur_liab,
        "retained": re_earn, "goodwill_intang": gi,
        "short_term_inv": st_inv, "ar_current": ar, "inventory": inventory,
        "total_debt": total_debt
    }

def compute_company_year(company_facts, prices_df, fy, maint_policy="dda", maint_fraction=0.5):
    revenue,_   = pick_fy_fact(company_facts, TAGSETS["revenue"], fy)
    gross,_     = pick_fy_fact(company_facts, TAGSETS["gross_profit"], fy)
    if gross is None and revenue is not None:
        cor,_   = pick_fy_fact(company_facts, TAGSETS["cost_of_revenue"], fy)
        if cor is not None: gross = revenue - cor
    opinc,_     = pick_fy_fact(company_facts, TAGSETS["oper_income"], fy)
    netinc,endN = pick_fy_fact(company_facts, TAGSETS["net_income"], fy)
    cfo = None
    for tag in TAGSETS["cfo"]:
        v,_ = pick_fy_fact(company_facts, [tag], fy)
        if v is not None: cfo = v; break
    capex,_     = pick_fy_fact(company_facts, TAGSETS["capex"], fy)
    ddna,_      = pick_fy_fact(company_facts, TAGSETS["ddna"], fy)
    eps_dil,_   = pick_fy_fact(company_facts, TAGSETS["eps_diluted"], fy)
    int_exp,_   = pick_fy_fact(company_facts, TAGSETS["interest_expense"], fy)
    pbt,_       = pick_fy_fact(company_facts, TAGSETS["pbt"], fy)
    tax_exp,_   = pick_fy_fact(company_facts, TAGSETS["tax_expense"], fy)
    wad_shares  = pick_weighted_avg_shares(company_facts, fy)

    end_date = endN or f"{fy}-12-31"
    inst = _compute_common_balances_for_instant(company_facts, end_date)
    assets,liab,equity,cash = inst["assets"],inst["liab"],inst["equity"],inst["cash"]
    cur_assets,cur_liab = inst["current_assets"],inst["current_liab"]
    re_earn = inst["retained"]
    short_inv, ar_cur, inventory = inst["short_term_inv"],inst["ar_current"],inst["inventory"]
    total_debt = inst["total_debt"]

    shares_out,_ = pick_latest_shares_asof(company_facts, asof=end_date)
    if shares_out is None:
        shares_out = wad_shares
    price = nearest_trading_close(prices_df, end_date)

    capex_pos = None if capex is None else abs(capex)
    fcf = None if (cfo is None or capex_pos is None) else (cfo - capex_pos)

    ddna_m = M(ddna) if ddna is not None else None
    capex_m = M(capex_pos) if capex_pos is not None else None
    if maint_policy == "dda":
        if ddna_m is None and capex_m is None: maint_capex_m = None
        elif ddna_m is None: maint_capex_m = capex_m
        elif capex_m is None: maint_capex_m = ddna_m
        else: maint_capex_m = min(ddna_m, capex_m)
    elif maint_policy == "fraction":
        maint_capex_m = None if capex_m is None else capex_m * maint_fraction
    else:
        maint_capex_m = None

    bvps = None if (equity is None or shares_out in (None,0)) else (equity / shares_out)
    pe   = None if (price is None or eps_dil in (None,0)) else (price / eps_dil)
    if pe is None and price is not None and shares_out not in (None,0) and netinc not in (None,0):
        pe = (price * shares_out) / netinc
    pb   = None if (price is None or bvps in (None,0)) else (price / bvps)
    mcap = None if (price is None or shares_out is None) else (price * shares_out)
    mcap_m = M(mcap)

    gm_pct   = pct(gross, revenue) if (gross is not None and revenue not in (None,0)) else None
    opm_pct  = pct(opinc, revenue) if (opinc is not None and revenue not in (None,0)) else None
    nim_pct  = pct(netinc, revenue) if (netinc is not None and revenue not in (None,0)) else None
    fcfm_pct = pct(fcf, revenue) if (fcf is not None and revenue not in (None,0)) else None

    owners_earn_m = M(fcf)
    oem_pct = pct(fcf, revenue) if (fcf is not None and revenue not in (None,0)) else None
    oe_over_mcap_pct = pct(fcf, mcap) if (fcf is not None and mcap not in (None,0)) else None

    d_to_e = safe_div(total_debt, equity) if (total_debt is not None and equity not in (None,0)) else None
    d_to_a = safe_div(total_debt, assets) if (total_debt is not None and assets not in (None,0)) else None
    int_over_op_pct = pct(int_exp, opinc) if (int_exp is not None and opinc not in (None,0)) else None
    dep_over_gross_pct = pct(ddna, gross) if (ddna is not None and gross not in (None,0)) else None

    roe_pct = pct(netinc, equity) if (netinc is not None and equity not in (None,0)) else None
    roa_pct = pct(netinc, assets) if (netinc is not None and assets not in (None,0)) else None

    eff_tax_rate = None
    if (pbt not in (None,0)) and (tax_exp is not None):
        eff_tax_rate = max(0.0, min(1.0, tax_exp / pbt))
    nopat = None if opinc is None else opinc * (1 - (eff_tax_rate if eff_tax_rate is not None else 0.21))
    invested_capital = None
    if equity is not None:
        ic = equity + (total_debt if total_debt is not None else 0.0)
        if cash is not None: ic -= cash
        invested_capital = ic
    roic_pct = pct(nopat, invested_capital) if (nopat is not None and invested_capital not in (None,0)) else None

    ebitda = None
    if opinc is not None and ddna is not None:
        ebitda = opinc + ddna
    ebitda_m = M(ebitda) if ebitda is not None else None
    ebitda_margin = pct(ebitda, revenue) if (ebitda is not None and revenue not in (None,0)) else None

    total_debt_m = M(total_debt) if total_debt is not None else None
    net_debt = None if (total_debt is None or cash is None) else (total_debt - cash)
    net_debt_m = M(net_debt) if net_debt is not None else None
    ev = None if mcap is None else (mcap + (total_debt if total_debt is not None else 0.0) - (cash if cash is not None else 0.0))
    ev_m = M(ev) if ev is not None else None

    ev_ebitda = safe_div(ev, ebitda) if (ev is not None and ebitda not in (None,0)) else None
    ev_sales  = safe_div(ev, revenue) if (ev is not None and revenue not in (None,0)) else None
    ps_ratio  = safe_div(mcap, revenue) if (mcap is not None and revenue not in (None,0)) else None
    p_fcf     = safe_div(mcap, fcf) if (mcap is not None and fcf not in (None,0)) else None

    dps,_  = pick_fy_fact(company_facts, TAGSETS["div_per_share"], fy)
    div_cash,_ = pick_fy_fact(company_facts, TAGSETS["div_cash"], fy)
    div_yield = None if (dps is None or price in (None,0)) else (dps/price*100.0)
    payout_ratio = None
    if dps is not None and (eps_dil not in (None,0)):
        payout_ratio = dps/eps_dil*100.0
    elif div_cash is not None and netinc not in (None,0):
        payout_ratio = (div_cash/abs(netinc))*100.0

    buybacks,_ = pick_fy_fact(company_facts, TAGSETS["buybacks"], fy)
    sbc,_      = pick_fy_fact(company_facts, TAGSETS["sbc"], fy)
    rnd,_      = pick_fy_fact(company_facts, TAGSETS["rnd"], fy)
    sga,_      = pick_fy_fact(company_facts, TAGSETS["sga"], fy)

    goodwill_intang = inst["goodwill_intang"]
    goodwill_intang_m = M(goodwill_intang) if goodwill_intang is not None else None
    tangible_book = None if (equity is None or goodwill_intang is None) else (equity - goodwill_intang)
    tbv_m = M(tangible_book) if tangible_book is not None else None
    tbv_ps = None if (tangible_book is None or shares_out in (None,0)) else (tangible_book/shares_out)

    working_cap = None if (cur_assets is None or cur_liab is None) else (cur_assets - cur_liab)
    working_cap_m = M(working_cap) if working_cap is not None else None
    quick_ratio = safe_div(((cash or 0.0) + (short_inv or 0.0) + (ar_cur or 0.0)), cur_liab) if cur_liab not in (None,0) else None
    inventory_m = M(inventory) if inventory is not None else None

    int_cov = None
    if int_exp not in (None,0) and opinc is not None and int_exp > 0:
        int_cov = opinc / int_exp

    net_debt_ebitda = safe_div(net_debt, ebitda) if (net_debt is not None and ebitda not in (None,0)) else None
    debt_ebitda     = safe_div(total_debt, ebitda) if (total_debt is not None and ebitda not in (None,0)) else None
    capex_over_sales_pct = pct(abs(capex) if capex is not None else None, revenue) if (revenue not in (None,0)) else None
    cash_conv_pct = pct(fcf, netinc) if (fcf is not None and netinc not in (None,0)) else None

    met = dict(
        price=price, bvps=bvps, shares=as_int(shares_out),
        pe=pe, pb=pb, mcap_m=mcap_m,
        assets_m=M(assets), liab_m=M(liab), equity_m=M(equity), cash_m=M(cash),
        revenue_m=M(revenue), gross_m=M(gross), gm_pct=gm_pct,
        opinc_m=M(opinc), opm_pct=opm_pct,
        netinc_m=M(netinc), nim_pct=nim_pct,
        fcf_m=M(fcf), fcfm_pct=fcfm_pct,
        capex_m=M(abs(capex)) if capex is not None else None,
        maint_capex_m=maint_capex_m,
        owners_earn_m=owners_earn_m, oem_pct=oem_pct,
        retained_m=M(re_earn),
        roic_pct=roic_pct, roe_pct=roe_pct, roa_pct=roa_pct,
        oe_over_mcap_pct=oe_over_mcap_pct,
        equity_over_mcap_pct=pct(equity, mcap) if (equity is not None and mcap not in (None,0)) else None,
        int_over_op_pct=int_over_op_pct,
        current_ratio=safe_div(cur_assets, cur_liab),
        d_to_e=d_to_e, d_to_a=d_to_a,
        dep_over_gross_pct=dep_over_gross_pct,
        eps_dil=eps_dil,
        cfo_m=M(cfo) if cfo is not None else None,
        cfo_margin=pct(cfo, revenue) if (cfo is not None and revenue not in (None,0)) else None,
        ddna_m=M(ddna) if ddna is not None else None,
        ebitda_m=ebitda_m,
        ebitda_margin=ebitda_margin,
        eff_tax_rate_pct=None if eff_tax_rate is None else eff_tax_rate*100.0,
        total_debt_m=M(total_debt) if total_debt is not None else None,
        net_debt_m=M(net_debt) if net_debt is not None else None,
        ev_m=M(ev) if ev is not None else None,
        ev_ebitda=ev_ebitda,
        ev_sales=ev_sales,
        ps_ratio=ps_ratio,
        p_fcf=p_fcf,
        dps=dps, div_yield_pct=div_yield, payout_ratio_pct=payout_ratio,
        buybacks_m=M(buybacks) if buybacks is not None else None,
        sbc_m=M(sbc) if sbc is not None else None,
        rnd_m=M(rnd) if rnd is not None else None,
        sga_m=M(sga) if sga is not None else None,
        goodwill_intang_m=goodwill_intang_m,
        tbv_m=tbv_m, tbv_ps=tbv_ps,
        working_cap_m=working_cap_m,
        quick_ratio=quick_ratio,
        inventory_m=inventory_m,
        int_coverage=int_cov,
        net_debt_ebitda=net_debt_ebitda,
        debt_ebitda=debt_ebitda,
        capex_over_sales_pct=capex_over_sales_pct,
        cash_conv_pct=cash_conv_pct,
        share_change_pct=None,
        wad_shares=as_int(wad_shares),
        is_ttm=False,
        raw={"revenue":revenue, "net_income":netinc, "retained":re_earn, "shares":shares_out},
        revenue_growth=None, netinc_growth=None, retained_growth=None
    )
    return met

def compute_company_ttm(cik, company_facts, prices_df, maint_policy="dda", maint_fraction=0.5, ttm_date=None, sleep=0.6):
    def ttm_from_facts(tags):
        return sum_last_quarters(company_facts, tags, asof=ttm_date, need=4)

    revenue, endR, nR = ttm_from_facts(TAGSETS["revenue"])
    gross,   endG, nG = ttm_from_facts(TAGSETS["gross_profit"])
    if gross is None and revenue is not None:
        cor, endC, nC = ttm_from_facts(TAGSETS["cost_of_revenue"])
        if cor is not None:
            gross = revenue - cor; endG = endC
    opinc,   endO, nO = ttm_from_facts(TAGSETS["oper_income"])
    netinc,  endN, nN = ttm_from_facts(TAGSETS["net_income"])
    cfo,     endCF,nCF= ttm_from_facts(TAGSETS["cfo"])
    capex,   endX, nX = ttm_from_facts(TAGSETS["capex"])
    ddna,    endD, nD = ttm_from_facts(TAGSETS["ddna"])
    int_exp, endI, nI = ttm_from_facts(TAGSETS["interest_expense"])
    pbt,     endB, nB = ttm_from_facts(TAGSETS["pbt"])
    tax_exp, endT, nT = ttm_from_facts(TAGSETS["tax_expense"])

    def need_concept(v, n): return v is None or n == 0
    if need_concept(revenue, nR):
        ser = get_concept_quarter_series(cik, TAGSETS["revenue"], sleep=sleep)
        revenue, endR, nR = sum_last_n_quarters_from_series(ser, n=4, asof=ttm_date)
    if need_concept(gross, nG) and revenue is not None:
        ser_g = get_concept_quarter_series(cik, TAGSETS["gross_profit"], sleep=sleep)
        g2, e2, n2 = sum_last_n_quarters_from_series(ser_g, n=4, asof=ttm_date)
        if g2 is None:
            ser_cor = get_concept_quarter_series(cik, TAGSETS["cost_of_revenue"], sleep=sleep)
            cor2, e3, n3 = sum_last_n_quarters_from_series(ser_cor, n=4, asof=ttm_date)
            if cor2 is not None:
                g2 = revenue - cor2; e2 = e3; n2 = min(nR, n3) if nR and n3 else (n3 or nR)
        gross, endG, nG = g2, e2, n2
    if need_concept(opinc, nO):
        ser = get_concept_quarter_series(cik, TAGSETS["oper_income"], sleep=sleep)
        opinc, endO, nO = sum_last_n_quarters_from_series(ser, n=4, asof=ttm_date)
    if need_concept(netinc, nN):
        ser = get_concept_quarter_series(cik, TAGSETS["net_income"], sleep=sleep)
        netinc, endN, nN = sum_last_n_quarters_from_series(ser, n=4, asof=ttm_date)
    if need_concept(cfo, nCF):
        ser = get_concept_quarter_series(cik, TAGSETS["cfo"], sleep=sleep)
        cfo, endCF, nCF = sum_last_n_quarters_from_series(ser, n=4, asof=ttm_date)
    if need_concept(capex, nX):
        ser = get_concept_quarter_series(cik, TAGSETS["capex"], sleep=sleep)
        capex, endX, nX = sum_last_n_quarters_from_series(ser, n=4, asof=ttm_date)
    if need_concept(ddna, nD):
        ser = get_concept_quarter_series(cik, TAGSETS["ddna"], sleep=sleep)
        ddna, endD, nD = sum_last_n_quarters_from_series(ser, n=4, asof=ttm_date)
    if need_concept(int_exp, nI):
        ser = get_concept_quarter_series(cik, TAGSETS["interest_expense"], sleep=sleep)
        int_exp, endI, nI = sum_last_n_quarters_from_series(ser, n=4, asof=ttm_date)
    if need_concept(pbt, nB):
        ser = get_concept_quarter_series(cik, TAGSETS["pbt"], sleep=sleep)
        pbt, endB, nB = sum_last_n_quarters_from_series(ser, n=4, asof=ttm_date)
    if need_concept(tax_exp, nT):
        ser = get_concept_quarter_series(cik, TAGSETS["tax_expense"], sleep=sleep)
        tax_exp, endT, nT = sum_last_n_quarters_from_series(ser, n=4, asof=ttm_date)

    end_dates = [d for d in (endR,endG,endO,endN,endCF,endX,endD,endI,endB,endT) if d]
    ttm_end  = sorted(end_dates)[-1] if end_dates else ttm_date

    inst = _compute_common_balances_for_instant(company_facts, ttm_end)
    assets,liab,equity,cash = inst["assets"],inst["liab"],inst["equity"],inst["cash"]
    cur_assets,cur_liab = inst["current_assets"],inst["current_liab"]
    re_earn = inst["retained"]
    short_inv, ar_cur, inventory = inst["short_term_inv"],inst["ar_current"],inst["inventory"]
    total_debt = inst["total_debt"]

    shares_out,_ = pick_latest_shares_asof(company_facts, asof=ttm_end)
    if shares_out is None and ttm_end:
        fy_guess = int(str(ttm_end)[:4]); shares_out = pick_weighted_avg_shares(company_facts, fy_guess)
    price = nearest_trading_close(prices_df, ttm_end)

    capex_pos = None if capex is None else abs(capex)
    fcf = None if (cfo is None or capex_pos is None) else (cfo - capex_pos)

    ddna_m = M(ddna) if ddna is not None else None
    capex_m = M(capex_pos) if capex_pos is not None else None
    if maint_policy == "dda":
        if ddna_m is None and capex_m is None: maint_capex_m = None
        elif ddna_m is None: maint_capex_m = capex_m
        elif capex_m is None: maint_capex_m = ddna_m
        else: maint_capex_m = min(ddna_m, capex_m)
    elif maint_policy == "fraction":
        maint_capex_m = None if capex_m is None else capex_m * maint_fraction
    else:
        maint_capex_m = None

    bvps = None if (equity is None or shares_out in (None,0)) else (equity / shares_out)
    pe=None
    if price is not None and shares_out not in (None,0) and netinc not in (None,0):
        pe = (price * shares_out) / netinc
    pb   = None if (price is None or bvps in (None,0)) else (price / bvps)
    mcap = None if (price is None or shares_out is None) else (price * shares_out)
    mcap_m = M(mcap)

    gm_pct   = pct(gross, revenue) if (gross is not None and revenue not in (None,0)) else None
    opm_pct  = pct(opinc, revenue) if (opinc is not None and revenue not in (None,0)) else None
    nim_pct  = pct(netinc, revenue) if (netinc is not None and revenue not in (None,0)) else None
    fcfm_pct = pct(fcf, revenue) if (fcf is not None and revenue not in (None,0)) else None

    owners_earn_m = M(fcf)
    oem_pct = pct(fcf, revenue) if (fcf is not None and revenue not in (None,0)) else None
    oe_over_mcap_pct = pct(fcf, mcap) if (fcf is not None and mcap not in (None,0)) else None

    d_to_e = safe_div(total_debt, equity) if (total_debt is not None and equity not in (None,0)) else None
    d_to_a = safe_div(total_debt, assets) if (total_debt is not None and assets not in (None,0)) else None
    int_over_op_pct = pct(int_exp, opinc) if (int_exp is not None and opinc not in (None,0)) else None
    dep_over_gross_pct = pct(ddna, gross) if (ddna is not None and gross not in (None,0)) else None

    roe_pct = pct(netinc, equity) if (netinc is not None and equity not in (None,0)) else None
    roa_pct = pct(netinc, assets) if (netinc is not None and assets not in (None,0)) else None

    eff_tax_rate = None
    if (pbt not in (None,0)) and (tax_exp is not None):
        eff_tax_rate = max(0.0, min(1.0, tax_exp / pbt))
    nopat = None if opinc is None else opinc * (1 - (eff_tax_rate if eff_tax_rate is not None else 0.21))
    invested_capital = None
    if equity is not None:
        ic = equity + (total_debt if total_debt is not None else 0.0)
        if cash is not None: ic -= cash
        invested_capital = ic
    roic_pct = pct(nopat, invested_capital) if (nopat is not None and invested_capital not in (None,0)) else None

    ebitda = None
    if opinc is not None and ddna is not None:
        ebitda = opinc + ddna
    ebitda_m = M(ebitda) if ebitda is not None else None
    ebitda_margin = pct(ebitda, revenue) if (ebitda is not None and revenue not in (None,0)) else None

    total_debt_m = M(total_debt) if total_debt is not None else None
    net_debt = None if (total_debt is None or cash is None) else (total_debt - cash)
    net_debt_m = M(net_debt) if net_debt is not None else None
    ev = None if mcap is None else (mcap + (total_debt if total_debt is not None else 0.0) - (cash if cash is not None else 0.0))
    ev_m = M(ev) if ev is not None else None

    ev_ebitda = safe_div(ev, ebitda) if (ev is not None and ebitda not in (None,0)) else None
    ev_sales  = safe_div(ev, revenue) if (ev is not None and revenue not in (None,0)) else None
    ps_ratio  = safe_div(mcap, revenue) if (mcap is not None and revenue not in (None,0)) else None
    p_fcf     = safe_div(mcap, fcf) if (mcap is not None and fcf not in (None,0)) else None

    goodwill_intang = inst["goodwill_intang"]
    goodwill_intang_m = M(goodwill_intang) if goodwill_intang is not None else None
    tangible_book = None if (equity is None or goodwill_intang is None) else (equity - goodwill_intang)
    tbv_m = M(tangible_book) if tangible_book is not None else None
    tbv_ps = None if (tangible_book is None or shares_out in (None,0)) else (tangible_book/shares_out)

    working_cap = None if (cur_assets is None or cur_liab is None) else (cur_assets - cur_liab)
    working_cap_m = M(working_cap) if working_cap is not None else None
    quick_ratio = safe_div(((cash or 0.0) + (short_inv or 0.0) + (ar_cur or 0.0)), cur_liab) if cur_liab not in (None,0) else None
    inventory_m = M(inventory) if inventory is not None else None

    int_cov = None
    if int_exp not in (None,0) and opinc is not None and int_exp > 0:
        int_cov = opinc / int_exp

    net_debt_ebitda = safe_div(net_debt, ebitda) if (net_debt is not None and ebitda not in (None,0)) else None
    debt_ebitda     = safe_div(total_debt, ebitda) if (total_debt is not None and ebitda not in (None,0)) else None
    capex_over_sales_pct = pct(abs(capex) if capex is not None else None, revenue) if (revenue not in (None,0)) else None
    cash_conv_pct = pct(fcf, netinc) if (fcf is not None and netinc not in (None,0)) else None

    met = dict(
        price=price, bvps=bvps, shares=as_int(shares_out),
        pe=pe, pb=pb, mcap_m=mcap_m,
        assets_m=M(assets), liab_m=M(liab), equity_m=M(equity), cash_m=M(cash),
        revenue_m=M(revenue), gross_m=M(gross), gm_pct=gm_pct,
        opinc_m=M(opinc), opm_pct=opm_pct,
        netinc_m=M(netinc), nim_pct=nim_pct,
        fcf_m=M(fcf), fcfm_pct=fcfm_pct,
        capex_m=M(abs(capex)) if capex is not None else None,
        maint_capex_m=maint_capex_m,
        owners_earn_m=M(fcf), oem_pct=oem_pct,
        retained_m=M(re_earn),
        roic_pct=roic_pct, roe_pct=roe_pct, roa_pct=roa_pct,
        oe_over_mcap_pct=oe_over_mcap_pct,
        equity_over_mcap_pct=pct(equity, mcap) if (equity is not None and mcap not in (None,0)) else None,
        int_over_op_pct=int_over_op_pct,
        current_ratio=safe_div(cur_assets, cur_liab),
        d_to_e=safe_div(total_debt, equity) if (total_debt is not None and equity not in (None,0)) else None,
        d_to_a=safe_div(total_debt, assets) if (total_debt is not None and assets not in (None,0)) else None,
        dep_over_gross_pct=dep_over_gross_pct,
        eps_dil=None,
        cfo_m=M(cfo) if cfo is not None else None,
        cfo_margin=pct(cfo, revenue) if (cfo is not None and revenue not in (None,0)) else None,
        ddna_m=M(ddna) if ddna is not None else None,
        ebitda_m=ebitda_m,
        ebitda_margin=ebitda_margin,
        eff_tax_rate_pct=None if eff_tax_rate is None else eff_tax_rate*100.0,
        total_debt_m=M(total_debt) if total_debt is not None else None,
        net_debt_m=M(net_debt) if net_debt is not None else None,
        ev_m=M(ev) if ev is not None else None,
        ev_ebitda=ev_ebitda,
        ev_sales=ev_sales,
        ps_ratio=ps_ratio,
        p_fcf=p_fcf,
        dps=None, div_yield_pct=None, payout_ratio_pct=None,
        buybacks_m=None, sbc_m=None, rnd_m=None, sga_m=None,
        goodwill_intang_m=goodwill_intang_m,
        tbv_m=tbv_m, tbv_ps=tbv_ps,
        working_cap_m=working_cap_m,
        quick_ratio=quick_ratio,
        inventory_m=inventory_m,
        int_coverage=int_cov,
        net_debt_ebitda=net_debt_ebitda,
        debt_ebitda=debt_ebitda,
        capex_over_sales_pct=capex_over_sales_pct,
        cash_conv_pct=cash_conv_pct,
        share_change_pct=None,
        wad_shares=as_int(shares_out),
        is_ttm=True,
        raw={"revenue":revenue, "net_income":netinc, "retained":re_earn, "shares":shares_out},
        revenue_growth=None, netinc_growth=None, retained_growth=None
    )
    return met

# ---------------- Growths ----------------
def fill_growths(row_pairs):
    row_pairs.sort(key=lambda x:x[0])
    prev=None
    for (y, met) in row_pairs:
        raw = met["raw"]
        if prev:
            _, pmet = prev
            praw = pmet["raw"]
            met["revenue_growth"] = yoy(raw["revenue"], praw["revenue"])
            met["netinc_growth"]  = yoy(raw["net_income"], praw["net_income"])
            met["retained_growth"]= yoy(raw["retained"], praw["retained"])
            div=None
            if raw["retained"] is not None and praw["retained"] is not None and raw["net_income"] is not None:
                div = raw["net_income"] - (raw["retained"] - praw["retained"])
            if div is not None and praw["retained"] not in (None,0):
                met["rore_pct"] = (raw["net_income"] - max(div,0)) / abs(praw["retained"]) * 100.0
        else:
            met["revenue_growth"]=met["netinc_growth"]=met["retained_growth"]=None
        prev=(y, met)

# ---------------- Writers & maps ----------------
CORE_MAP = {
    "Market Price (USD)": "price",
    "Book Value per Share (USD)": "bvps",
    "Total Number of Shares": "shares",
    "Price to Earning ratio (P/E)": "pe",
    "Price to Book ratio (P/B)": "pb",
    "Total Market Cap (Millions USD)": "mcap_m",
    "Total Assets (Millions USD)": "assets_m",
    "Total Liabilities (Millions USD)": "liab_m",
    "Book Value (Millions USD)": "equity_m",
    "Cash and cash equivalents (Millions USD)": "cash_m",
    "Revenue (Millions USD)": "revenue_m",
    "Revenue Growth Rate": "revenue_growth",
    "Gross profit (Millions USD)": "gross_m",
    "Gross Margin (%)": "gm_pct",
    "Operating Profit (Millions USD)": "opinc_m",
    "Operation Margin (%)": "opm_pct",
    "Net Income(Millions USD)": "netinc_m",
    "Net Income Growth Rate (%)": "netinc_growth",
    "Net income margin (%)": "nim_pct",
    "Free Cash Flow (Millions USD)": "fcf_m",
    "Free Cash Flow Margin (%)": "fcfm_pct",
    "Capital Expenditure (Millions USD)": "capex_m",
    "Maintenance Capital Expenditure (Millions USD)": "maint_capex_m",
    "Owner’s Earnings (Millions USD)": "owners_earn_m",
    "Owner’s Earnings Margin (%)": "oem_pct",
    "Retained Earnings (Millions USD)": "retained_m",
    "Retained Earnings Growth Rate (%)": "retained_growth",
    "Return on Retained Earnings (%)": "rore_pct",
    "Return on Invested Capital (%)": "roic_pct",
    "Return on Equity (%)": "roe_pct",
    "Return on Assets (%)": "roa_pct",
    "Owner’s Earnings (Millions USD)": "owners_earn_m",
    "Owner’s Earnings / Total Market Cap (%)": "oe_over_mcap_pct",
    "Shareholder’s Equity (Millions USD)": "equity_m",
    "Shareholder’s Equity / Total Market Cap (%)": "equity_over_mcap_pct",
    "Interest Expense / Operating Income (%)": "int_over_op_pct",
    "Current Ratio": "current_ratio",
    "Debt to equity ratio": "d_to_e",
    "Debt to asset ratio": "d_to_a",
    "Depreciation / Gross Profit (%)": "dep_over_gross_pct",
}
EXTRA_MAP = {
    "Earnings Per Share – Diluted (USD)": "eps_dil",
    "Operating Cash Flow (Millions USD)": "cfo_m",
    "Operating Cash Flow Margin (%)": "cfo_margin",
    "Depreciation & Amortization (Millions USD)": "ddna_m",
    "EBITDA (Millions USD)": "ebitda_m",
    "EBITDA Margin (%)": "ebitda_margin",
    "Effective Tax Rate (%)": "eff_tax_rate_pct",
    "Total Debt (Millions USD)": "total_debt_m",
    "Net Debt (Millions USD)": "net_debt_m",
    "Enterprise Value (Millions USD)": "ev_m",
    "EV / EBITDA": "ev_ebitda",
    "EV / Sales": "ev_sales",
    "Price to Sales ratio (P/S)": "ps_ratio",
    "Price to Free Cash Flow (P/FCF)": "p_fcf",
    "Dividend per Share (USD)": "dps",
    "Dividend Yield (%)": "div_yield_pct",
    "Dividend Payout Ratio (%)": "payout_ratio_pct",
    "Share Repurchases (Millions USD)": "buybacks_m",
    "Share-Based Compensation (Millions USD)": "sbc_m",
    "Research & Development (Millions USD)": "rnd_m",
    "Selling, General & Administrative (Millions USD)": "sga_m",
    "Goodwill & Intangibles (Millions USD)": "goodwill_intang_m",
    "Tangible Book Value (Millions USD)": "tbv_m",
    "Tangible Book Value per Share (USD)": "tbv_ps",
    "Working Capital (Millions USD)": "working_cap_m",
    "Quick Ratio": "quick_ratio",
    "Inventory (Millions USD)": "inventory_m",
    "Interest Coverage (x)": "int_coverage",
    "Net Debt / EBITDA": "net_debt_ebitda",
    "Debt / EBITDA": "debt_ebitda",
    "Capex / Revenue (%)": "capex_over_sales_pct",
    "Cash Conversion (%)": "cash_conv_pct",
    "Share Count Change (% YoY)": "share_change_pct",
    "Weighted Average Diluted Shares": "wad_shares",
}

def write_company_csv(ticker, years, rows, outdir, full_metrics=False, company_name=""):
    cols = CORE_COLS + (EXTRA_COLS if full_metrics else [])
    header = ["Year"] + cols
    safe_company = safe_name_component(company_name)
    path = os.path.join(outdir, f"{ticker.upper()}-{safe_company}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header)
        for (year, met) in sorted(rows, key=lambda x:x[0]):
            def V(x): return "" if x is None else x
            row = [year]
            for c in CORE_COLS: row.append(V(met.get(CORE_MAP[c])))
            if full_metrics:
                for c in EXTRA_COLS: row.append(V(met.get(EXTRA_MAP[c])))
            w.writerow(row)
    return path

# ---------------- Deep validation ----------------
CRITICAL_FOR_CAP = {
    "Book Value per Share (USD)",
    "Total Number of Shares",
    "Total Market Cap (Millions USD)",
    "Owner’s Earnings / Total Market Cap (%)",
    "Shareholder’s Equity / Total Market Cap (%)"
}
def deep_repair_row(row_dict, facts, cik, prices_df, is_latest_year, maint_policy, maint_fraction, sleep=0.6):
    def F(name):
        v = row_dict.get(name, "")
        try: return float(v) if v != "" else None
        except: return None
    price = F("Market Price (USD)")
    equity_m = F("Shareholder’s Equity (Millions USD)")
    owners_earn_m = F("Owner’s Earnings (Millions USD)")
    shares = F("Total Number of Shares")
    bvps = F("Book Value per Share (USD)")
    mcap_m = F("Total Market Cap (Millions USD)")
    repaired=False
    if shares is None and facts is not None:
        s,_ = pick_latest_shares_asof(facts, asof=None)
        if s is None:
            for yr in range(datetime.now().year, datetime.now().year-10, -1):
                s = pick_weighted_avg_shares(facts, yr)
                if s: break
        if s:
            row_dict["Total Number of Shares"] = str(int(round(s))); repaired=True; shares=s
    if mcap_m is None and price is not None and shares not in (None,0):
        row_dict["Total Market Cap (Millions USD)"] = str(M(price*shares)); repaired=True; mcap_m = M(price*shares)
    if bvps is None and equity_m is not None and shares not in (None,0):
        row_dict["Book Value per Share (USD)"] = str((equity_m*1_000_000.0)/shares); repaired=True
    if row_dict.get("Owner’s Earnings / Total Market Cap (%)","") in ("", None) and \
       owners_earn_m is not None and (mcap_m not in (None,0)):
        oe = owners_earn_m*1_000_000.0; mc = mcap_m*1_000_000.0
        row_dict["Owner’s Earnings / Total Market Cap (%)"] = str((oe/mc)*100.0); repaired=True
    if row_dict.get("Shareholder’s Equity / Total Market Cap (%)","") in ("", None) and \
       equity_m is not None and (mcap_m not in (None,0)):
        mc = mcap_m*1_000_000.0; eq = equity_m*1_000_000.0
        row_dict["Shareholder’s Equity / Total Market Cap (%)"] = str((eq/mc)*100.0); repaired=True
    return repaired

def run_deep_validation(csv_paths, full_metrics, cik_map, maint_policy, maint_fraction):
    report = [["Ticker","Year","Missing Metrics (critical)","Was Repaired"]]
    for p in csv_paths:
        base = os.path.basename(p)
        tkr = base.split("-", 1)[0]  # extract ticker from "TICKER-CompanyName.csv"
        cik = cik_map.get(tkr.upper())
        facts = fetch_sec_company_facts(cik, sleep=0.0) if cik else None
        prices_df = fetch_stooq_prices(tkr)
        with open(p, "r", newline="") as f:
            rdr = csv.DictReader(f); header = rdr.fieldnames; rows = list(rdr)
        changed_any=False
        years = [int(r["Year"]) for r in rows if r.get("Year")]
        if not years: continue
        max_year = max(years)
        for row in rows:
            year = int(row.get("Year","0") or 0)
            missing = [col for col in CRITICAL_FOR_CAP if row.get(col,"") in ("", None)]
            repaired = deep_repair_row(row, facts, cik, prices_df, year==max_year, maint_policy, maint_fraction, sleep=0.4)
            changed_any = changed_any or repaired
            report.append([tkr, year, "; ".join(missing), "yes" if repaired else "no"])
        if changed_any:
            with open(p, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=header); w.writeheader()
                for row in rows: w.writerow(row)
    return report

# ---------------- Processing & cap ordering ----------------
def write_blank_csv_for_symbol(ticker, years, outdir, full_metrics, prices_df, company_name=""):
    cols = CORE_COLS + (EXTRA_COLS if full_metrics else [])
    header = ["Year"] + cols
    safe_company = safe_name_component(company_name)
    path = os.path.join(outdir, f"{ticker.upper()}-{safe_company}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header)
        for y in years:
            if prices_df is not None and not prices_df.empty:
                tdate = f"{y}-12-31"; price = nearest_trading_close(prices_df, tdate)
            else:
                price = None
            row = [y] + ["" for _ in cols]
            if price is not None:
                mp_idx = header.index("Market Price (USD)")
                row[mp_idx] = price
            w.writerow(row)
    return path

def process_company(ticker, cik_map, years, sleep=0.8, maint_policy="dda", maint_fraction=0.5,
                    ttm_for_latest=True, ttm_date=None):
    cik = cik_map.get(ticker.upper())
    if not cik: raise RuntimeError(f"CIK not found for {ticker}")
    facts = fetch_sec_company_facts(cik, sleep=sleep)
    if facts is None: raise RuntimeError(f"SEC companyfacts not available for {ticker}")
    company_name = (facts or {}).get("entityName") or ""
    prices_df = fetch_stooq_prices(ticker)
    rows=[]
    end_year = years[-1]
    collect=[]
    for fy in years:
        if fy == end_year and ttm_for_latest:
            met = compute_company_ttm(cik, facts, prices_df, maint_policy, maint_fraction, ttm_date, sleep=sleep)
        else:
            met = compute_company_year(facts, prices_df, fy, maint_policy, maint_fraction)
        collect.append((fy, met))
    fill_growths(collect)
    for fy, met in collect: rows.append((fy, met))
    return rows, company_name

def estimate_quick_mcap(tkr, cik_map, sleep=0.1):
    cik = cik_map.get(tkr.upper())
    if not cik: return None
    try:
        facts = fetch_sec_company_facts(cik, sleep=sleep)
        if not facts: return None
        shares,_ = pick_latest_shares_asof(facts, asof=None)
        if shares is None:
            for yr in range(datetime.now().year, datetime.now().year-6, -1):
                shares = pick_weighted_avg_shares(facts, yr)
                if shares: break
        if shares is None: return None
        pr = fetch_stooq_prices(tkr)
        price = None
        if pr is not None and not pr.empty:
            price = float(pr["close"].iloc[-1])
        if price is None: return None
        return price * float(shares)
    except Exception:
        return None

def order_by_estimated_mcap(tickers, cik_map, sleep=0.1):
    log(f"Estimating market caps for ordering ({len(tickers)} symbols)…")
    ranked=[]
    for i, t in enumerate(tickers, 1):
        mc = estimate_quick_mcap(t, cik_map, sleep=sleep)
        ranked.append((t, mc if mc is not None else -1.0))
        if i % 200 == 0: log(f"  progress: {i}/{len(tickers)}")
    ranked.sort(key=lambda x: (x[1] if x[1] is not None else -1.0), reverse=True)
    return [t for t,_ in ranked]

# ---------------- ETF scraping helpers ----------------
def soup_from(url):
    try:
        r = http_get(url, headers=WEB_HEADERS, timeout=30)
        if r.status_code == 404 or not r.text.strip(): return None
        return BeautifulSoup(r.text, "lxml")
    except Exception:
        return None

def text2num(s):
    if s is None: return None
    s = re.sub(r"[,\s]+","", str(s))
    m = re.match(r"^\$?(-?\d+(\.\d+)?)([MBT]?)$", s, re.I)
    if m:
        val = float(m.group(1)); suf = (m.group(3) or "").upper()
        mult = {"M":1e6,"B":1e9,"T":1e12}.get(suf,1.0)
        return val*mult
    try:
        if s.endswith("%"): return float(s[:-1])
        if s.startswith("$"): return float(s[1:])
        return float(s)
    except: return None

def find_label_value_pairs(soup, labels):
    out={}
    if soup is None: return out
    text = soup.get_text(" ", strip=True)
    for lab in labels:
        pattern = re.compile(rf"{re.escape(lab)}\s*[:\-–]\s*([^\|·•\n\r]+?)\s{1,3}([A-Z][a-z]+:|$)", re.I)
        m = pattern.search(text + " Next:")
        if m:
            val = m.group(1).strip()
            out[lab] = val
            continue
        m2 = re.search(rf"{re.escape(lab)}\s*[:\-–]?\s*([^\n\r]{{1,25}})", text, re.I)
        if m2:
            out[lab]=m2.group(1).strip()
    return out

def _fund_name_from_soup(soup, ticker):
    if not soup: return None
    try_tags = []
    title = soup.find("title")
    if title and title.text: try_tags.append(title.text.strip())
    for tag in soup.find_all(["h1","h2","h3"]):
        txt = (tag.get_text(" ", strip=True) or "").strip()
        if txt: try_tags.append(txt)
    meta = soup.find("meta", attrs={"property":"og:title"})
    if meta and meta.get("content"): try_tags.append(meta.get("content").strip())
    t = ticker.upper()
    for s in try_tags:
        if t in s or re.search(r"\bETF\b|\bFund\b", s, re.I):
            s1 = re.sub(r"\s*\|.*$","", s).strip()
            s1 = re.sub(r"\s*\("+re.escape(t)+r"\)\s*$","", s1).strip()
            if s1: return s1
    for s in try_tags:
        if re.search(r"\bETF\b|\bFund\b", s, re.I):
            s1 = re.sub(r"\s*\|.*$","", s).strip()
            return s1
    return None

def scraper_vanguard(ticker):
    urls = [
        f"https://investor.vanguard.com/investment-products/etfs/profile/{ticker.lower()}",
        f"https://investor.vanguard.com/investment-products/etfs/profile/{ticker.upper()}",
    ]
    labels = [
        "Expense ratio","SEC yield","Distribution frequency","Inception date",
        "Net assets","Median bid/ask spread","Number of stocks","Number of bonds",
        "Benchmark"
    ]
    for u in urls:
        soup = soup_from(u)
        if soup:
            vals = find_label_value_pairs(soup, labels)
            fn = _fund_name_from_soup(soup, ticker)
            if vals or fn:
                return {"source":u, "vals":vals, "name":fn}
    return None

def scraper_invesco(ticker):
    urls = [
        f"https://www.invesco.com/us/financial-products/etfs/product-detail?productId={ticker.upper()}",
    ]
    labels = [
        "Expense ratio","30 day SEC yield","Distribution frequency","Inception date",
        "Total net assets","Median bid/ask spread","Number of holdings","Benchmark"
    ]
    for u in urls:
        soup = soup_from(u)
        if soup:
            vals = find_label_value_pairs(soup, labels)
            fn = _fund_name_from_soup(soup, ticker)
            if vals or fn:
                return {"source":u, "vals":vals, "name":fn}
    return None

def scraper_ark(ticker):
    urls = [
        f"https://ark-funds.com/funds/{ticker.lower()}",
        f"https://ark-funds.com/funds/{ticker.upper()}",
    ]
    labels = [
        "Expense Ratio","30 Day SEC Yield","Distribution Frequency","Inception Date",
        "Net Assets","Median Bid/Ask Spread","Holdings","Benchmark"
    ]
    for u in urls:
        soup = soup_from(u)
        if soup:
            vals = find_label_value_pairs(soup, labels)
            fn = _fund_name_from_soup(soup, ticker)
            if vals or fn:
                return {"source":u, "vals":vals, "name":fn}
    return None

def scraper_spdr(ticker): return None
def scraper_ishares(ticker): return None
def scraper_schwab(ticker): return None
def scraper_firsttrust(ticker): return None

SCRAPERS = [scraper_vanguard, scraper_invesco, scraper_ark, scraper_spdr, scraper_ishares, scraper_schwab, scraper_firsttrust]

def parse_issuer_vals(tkr, vals):
    m = {}
    for k in ("Expense ratio","Expense Ratio"):
        if k in vals:
            v = vals[k].replace("%","").strip()
            m["Expense Ratio (Net, %)"] = text2num(v); break
    for k in ("SEC yield","30 day SEC yield","30 Day SEC Yield"):
        if k in vals:
            m["30-Day SEC Yield (%)"] = text2num(vals[k].replace("%","")); break
    for k in ("Distribution frequency","Distribution Frequency"):
        if k in vals: m["Distribution Frequency"] = vals[k]
    for k in ("Inception date","Inception Date"):
        if k in vals: m["Inception Date"] = vals[k]
    for k in ("Net assets","Total net assets","Net Assets"):
        if k in vals:
            m["Total Net Assets (USD)"] = text2num(vals[k]); break
    for k in ("Median bid/ask spread","Median Bid/Ask Spread"):
        if k in vals:
            v = vals[k].replace("%","").strip()
            m["30-Day Median Bid/Ask (%)"] = text2num(v); break
    for k in ("Number of holdings","Holdings","Number of stocks","Number of bonds"):
        if k in vals:
            m["Number of Holdings"] = text2num(vals[k]); break
    if "Benchmark" in vals: m["Benchmark"] = vals["Benchmark"]
    return m

def fetch_etf_metrics(ticker, cik, entity_name, prices_df, exch_hint=None):
    data = {k:"" for k in ETF_COLS}
    data["Ticker"] = ticker.upper()
    data["CIK"] = cik or ""
    data["Issuer (Entity)"] = (entity_name or "").strip()
    last_close = None; as_of = None
    if prices_df is not None and not prices_df.empty:
        last_close = float(prices_df["close"].iloc[-1])
        as_of = str(prices_df["date"].iloc[-1]) if "date" in prices_df.columns else None
    data["Last Close (USD)"] = last_close if last_close is not None else ""
    data["As Of Date"] = as_of or ""
    if exch_hint: data["Exchange"] = exch_hint

    pr = compute_price_returns(prices_df)
    if pr["r_1y"] is not None: data["1Y Return (Price, %)"] = pr["r_1y"]
    if pr["cagr_3y"] is not None: data["3Y CAGR (Price, %)"] = pr["cagr_3y"]
    if pr["cagr_5y"] is not None: data["5Y CAGR (Price, %)"] = pr["cagr_5y"]
    if pr["cagr_10y"] is not None: data["10Y CAGR (Price, %)"] = pr["cagr_10y"]
    if pr["cagr_si"] is not None: data["Since Inception CAGR (Price, %)"] = pr["cagr_si"]

    scraped=None
    for fn in SCRAPERS:
        try:
            out = fn(ticker)
            if out and (out.get("vals") or out.get("name")):
                scraped = out; break
        except Exception:
            continue
    if scraped:
        vals = parse_issuer_vals(ticker, scraped.get("vals", {}))
        for k,v in vals.items():
            if k in data and v not in (None,""): data[k] = v
        if not data.get("Fund Name"):
            fnm = scraped.get("name")
            if fnm: data["Fund Name"] = fnm

    if not data.get("Fund Name"):
        if entity_name: data["Fund Name"] = entity_name

    return data

def write_etf_csv_row(path, header, rowdict):
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header: w.writeheader()
        out = {h: rowdict.get(h, "") for h in header}
        w.writerow(out)

# ---------------- Universe build helpers for US market ----------------
def build_universe_usmarket(include_etfs, include_otc, order_by_mcap_flag, prefetch_sleep, cik_map):
    comb = fetch_usmarket_tables(include_etfs=include_etfs)
    tickers = set(comb["__SYM__"].tolist())
    if include_otc:
        sec_syms = set(cik_map.keys())
        otc = {t for t in sec_syms if t not in tickers and re.match(r"^[A-Z0-9\.\-]+$", t)}
        tickers |= otc
    tickers = sorted(tickers)
    etf_set = set(comb.loc[comb["ETF"].astype(str).str.upper()=="Y","__SYM__"].tolist())
    exch_map = dict(zip(comb["__SYM__"], comb["Listing Exchange"])) if "Listing Exchange" in comb.columns else {}
    cap_sorted=False
    if order_by_mcap_flag:
        tickers = order_by_estimated_mcap(tickers, cik_map, sleep=prefetch_sleep)
        cap_sorted=True
    return tickers, etf_set, exch_map, cap_sorted

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="Companies + ETF reports (file names include names).")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--sp500", action="store_true")
    grp.add_argument("--r2000", action="store_true")
    grp.add_argument("--nasdaq100", action="store_true")
    grp.add_argument("--usmarket", action="store_true")

    ap.add_argument("--include-etfs", action="store_true", help="Include ETFs/ETPs when using --usmarket")
    ap.add_argument("--include-otc", action="store_true")

    ap.add_argument("--etf-policy", choices=["metrics","blank","skip"], default="skip",
                    help="How to handle ETFs: metrics (scrape + price returns), blank (price-only), or skip")
    ap.add_argument("--etf-outdir", type=str, default="", help="Separate ETF reports folder (default: <outdir>/_etf_reports)")

    ap.add_argument("--tickers", type=str, default="")
    ap.add_argument("--tickers-file", type=str)
    ap.add_argument("--outdir", type=str, default="out_csv")
    ap.add_argument("--years", nargs=2, type=int, default=[2015, 2025])
    ap.add_argument("--sleep", type=float, default=0.8)

    ap.add_argument("--top", type=int, default=0)
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--validate-deep", action="store_true")

    ap.add_argument("--maint-policy", choices=["dda","blank","fraction"], default="dda")
    ap.add_argument("--maint-fraction", type=float, default=0.5)

    ap.add_argument("--ishares-asof", type=str, default="")
    ap.add_argument("--ttm-for-latest", dest="ttm_for_latest", action="store_true", default=True)
    ap.add_argument("--no-ttm-for-latest", dest="ttm_for_latest", action="store_false")
    ap.add_argument("--ttm-date", type=str, default="")
    ap.add_argument("--full-metrics", action="store_true")
    ap.add_argument("--order-by-mcap", dest="order_by_mcap", action="store_true", default=True)
    ap.add_argument("--no-order-by-mcap", dest="order_by_mcap", action="store_false")
    ap.add_argument("--prefetch-sleep", type=float, default=0.1)

    args = ap.parse_args()
    if not SEC_EMAIL:
        log("WARNING: SEC_EMAIL not set. Set it for polite SEC API usage.")
    os.makedirs(args.outdir, exist_ok=True)

    # ETF outdir
    etf_outdir = args.etf_outdir or os.path.join(args.outdir, "_etf_reports")
    if args.etf_policy != "skip":
        os.makedirs(etf_outdir, exist_ok=True)

    cik_map = fetch_sec_ticker_map(sleep=args.sleep)

    etf_set = set(); exch_map={}
    cap_sorted=False
    if args.tickers or args.tickers_file:
        if args.tickers_file:
            with open(args.tickers_file) as f:
                tickers = [line.strip().upper() for line in f if line.strip()]
        else:
            tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        log(f"Using explicit tickers ({len(tickers)}).")
        comb = fetch_usmarket_tables(include_etfs=True)
        etf_set = set(comb.loc[comb["ETF"].astype(str).str.upper()=="Y","__SYM__"].tolist())
        if "Listing Exchange" in comb.columns:
            exch_map = dict(zip(comb["__SYM__"], comb["Listing Exchange"]))
    else:
        if args.sp500:
            tickers = fetch_sp500_tickers_slickcharts() or []
            if not tickers:
                log("Slickcharts failed; fallback to iShares IVV…")
                tickers = fetch_sp500_tickers_ivv(as_of=args.ishares_asof or None)
            if not tickers:
                log("IVV fallback failed; fallback to Wikipedia…")
                tickers = fetch_sp500_tickers_wiki()
            cap_sorted=True
        elif args.r2000:
            tickers = fetch_r2000_tickers_from_ishares(as_of=args.ishares_asof or None)
            cap_sorted=True
        elif args.nasdaq100:
            tickers = fetch_nasdaq100_tickers_slickcharts()
            if not tickers:
                log("Slickcharts NDX failed; fallback to Wikipedia…")
                tickers = fetch_nasdaq100_tickers_wiki() or []
            cap_sorted=True
        else:
            tickers, etf_set, exch_map, cap_sorted = build_universe_usmarket(
                include_etfs=args.include_etfs,
                include_otc=args.include_otc,
                order_by_mcap_flag=args.order_by_mcap,
                prefetch_sleep=args.prefetch_sleep,
                cik_map=cik_map
            )
            log(f"US market selected: {len(tickers)} tickers (ETFs included={args.include_etfs}, ETF count flagged={len(etf_set)})")

    if cap_sorted and args.top and args.top > 0:
        tickers = tickers[:args.top]; log(f"Using top {len(tickers)} by cap/weight.")
    if args.max and args.max > 0:
        tickers = tickers[:args.max]; log(f"--max applied: first {len(tickers)} symbols.")

    y0, y1 = args.years
    years = list(range(y0, y1+1))
    ttm_date = args.ttm_date if args.ttm_date else None

    csv_paths=[]
    etf_csv_paths=[]

    for i, t in enumerate(tickers, 1):
        is_etf = (t in etf_set)

        # ETFs
        if is_etf:
            try:
                if args.etf_policy == "skip":
                    log(f"{i}/{len(tickers)} {t}: ETF → skipped (use --etf-policy metrics/blank)")
                    continue
                prices_df = fetch_stooq_prices(t)
                if args.etf_policy == "blank":
                    row = fetch_etf_metrics(t, cik_map.get(t), None, prices_df, exch_hint=exch_map.get(t))
                    safe_fund = safe_name_component(row.get("Fund Name") or row.get("Issuer (Entity)") or "")
                    efp = os.path.join(etf_outdir, f"{t}-{safe_fund}.csv")
                    write_etf_csv_row(efp, ETF_COLS, row)
                    etf_csv_paths.append(efp)
                    log(f"{i}/{len(tickers)} {t}: ETF (blank) → {os.path.basename(efp)}")
                else:
                    facts = fetch_sec_company_facts(cik_map.get(t), sleep=args.sleep) if cik_map.get(t) else None
                    entity_name = (facts or {}).get("entityName")
                    row = fetch_etf_metrics(t, cik_map.get(t), entity_name, prices_df, exch_hint=exch_map.get(t))
                    safe_fund = safe_name_component(row.get("Fund Name") or entity_name or "")
                    efp = os.path.join(etf_outdir, f"{t}-{safe_fund}.csv")
                    write_etf_csv_row(efp, ETF_COLS, row)
                    etf_csv_paths.append(efp)
                    log(f"{i}/{len(tickers)} {t}: ETF (metrics) → {os.path.basename(efp)}")
                continue
            except Exception as e:
                log(f"{i}/{len(tickers)} {t}: ETF ERROR: {e}")
                continue

        # Companies
        try:
            rows, company_name = process_company(
                t, cik_map, years,
                sleep=args.sleep,
                maint_policy=args.maint_policy,
                maint_fraction=args.maint_fraction,
                ttm_for_latest=args.ttm_for_latest,
                ttm_date=ttm_date
            )
            path = write_company_csv(t, years, rows, args.outdir, full_metrics=args.full_metrics, company_name=company_name)
            csv_paths.append(path)
            log(f"{i}/{len(tickers)} {t}: done → {os.path.basename(path)}")
        except Exception as e:
            log(f"{i}/{len(tickers)} {t}: ERROR: {e}")

    # ----- Header validation
    if args.validate:
        if csv_paths:
            expected = ["Year"] + CORE_COLS + (EXTRA_COLS if args.full_metrics else [])
            mismatches=[]
            for p in csv_paths:
                try:
                    with open(p, "r") as f:
                        header = next(csv.reader(f))
                    if header == expected:
                        log(f"Header validated: {os.path.basename(p)}")
                    else:
                        mismatches.append((p, header))
                except Exception as ex:
                    mismatches.append((p, f"READ_ERROR: {ex}"))
            if mismatches:
                msg = "Company CSV header mismatch for: " + ", ".join(os.path.basename(x[0]) for x in mismatches)
                raise AssertionError(msg)
        if etf_csv_paths:
            for p in etf_csv_paths:
                with open(p, "r") as f:
                    header = next(csv.reader(f))
                if header != ETF_COLS:
                    raise AssertionError(f"ETF header mismatch in {os.path.basename(p)}")

    # Deep validation (companies only)
    if args.validate_deep and csv_paths:
        report = run_deep_validation(csv_paths, args.full_metrics, cik_map, args.maint_policy, args.maint_fraction)
        rep_path = os.path.join(args.outdir, "_validation_report.csv")
        with open(rep_path, "w", newline="") as f:
            w = csv.writer(f); w.writerows(report)
        log(f"Validation report: {rep_path}")

    # Zip outputs
    zip_path = os.path.join(args.outdir, "financials.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in csv_paths:
            if p: zf.write(p, arcname=os.path.basename(p))
        rep_path = os.path.join(args.outdir, "_validation_report.csv")
        if os.path.exists(rep_path): zf.write(rep_path, arcname=os.path.basename(rep_path))
    log(f"Company ZIP created: {zip_path}")

    if etf_csv_paths:
        etf_zip = os.path.join(os.path.dirname(etf_csv_paths[0]), "etf_reports.zip")
        with zipfile.ZipFile(etf_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in etf_csv_paths:
                if p: zf.write(p, arcname=os.path.basename(p))
        log(f"ETF ZIP created: {etf_zip}")

    print(zip_path)
    if etf_csv_paths:
        print(etf_zip)

if __name__ == "__main__":
    main()
