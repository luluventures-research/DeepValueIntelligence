#!/usr/bin/env python3
"""
Run daily analysis for S&P 500 companies reporting earnings on a given date.

Designed for cron usage:
- Finds S&P 500 tickers (Wikipedia, with local cache fallback)
- Fetches earnings calendar for that date from Finnhub
- Intersects both lists
- Runs `main.py` for each ticker with default settings and `--date`
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Iterable, List, Set
from zoneinfo import ZoneInfo

import pandas as pd
import requests


WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
GITHUB_SP500_CSV_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
FINNHUB_EARNINGS_URL = "https://finnhub.io/api/v1/calendar/earnings"
FINNHUB_CONSTITUENTS_URL = "https://finnhub.io/api/v1/index/constituents"
PACIFIC_TZ = ZoneInfo("America/Los_Angeles")
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}


def _today_pacific() -> str:
    return datetime.now(PACIFIC_TZ).strftime("%Y-%m-%d")


def _load_sp500_from_wikipedia(timeout_s: int = 20) -> List[str]:
    response = requests.get(WIKI_SP500_URL, timeout=timeout_s, headers=DEFAULT_HEADERS)
    response.raise_for_status()
    tables = pd.read_html(response.text)
    if not tables:
        raise RuntimeError("No tables found on S&P 500 Wikipedia page.")
    table = tables[0]
    if "Symbol" not in table.columns:
        raise RuntimeError("S&P 500 table missing Symbol column.")
    symbols = [str(x).strip().upper() for x in table["Symbol"].dropna().tolist()]
    return [s for s in symbols if s]


def _load_sp500_from_finnhub(finnhub_api_key: str, timeout_s: int = 20) -> List[str]:
    params = {"symbol": "^GSPC", "token": finnhub_api_key}
    response = requests.get(FINNHUB_CONSTITUENTS_URL, params=params, timeout=timeout_s, headers=DEFAULT_HEADERS)
    response.raise_for_status()
    data = response.json() or {}
    constituents = data.get("constituents", [])
    symbols = [str(x).strip().upper() for x in constituents if str(x).strip()]
    if not symbols:
        raise RuntimeError("Finnhub constituents response did not include symbols.")
    return symbols


def _load_sp500_from_github_csv(timeout_s: int = 20) -> List[str]:
    response = requests.get(GITHUB_SP500_CSV_URL, timeout=timeout_s, headers=DEFAULT_HEADERS)
    response.raise_for_status()
    table = pd.read_csv(StringIO(response.text))
    if "Symbol" not in table.columns:
        raise RuntimeError("GitHub S&P 500 CSV missing Symbol column.")
    symbols = [str(x).strip().upper() for x in table["Symbol"].dropna().tolist()]
    return [s for s in symbols if s]


def _load_sp500_from_local_file(sp500_file: Path) -> List[str]:
    if not sp500_file.exists():
        raise RuntimeError(f"S&P 500 file not found: {sp500_file}")
    text = sp500_file.read_text(encoding="utf-8")
    symbols: List[str] = []
    for line in text.splitlines():
        t = line.strip().upper()
        if not t or t.startswith("#"):
            continue
        symbols.append(t.split(",")[0].strip())
    symbols = [s for s in symbols if s and s != "SYMBOL"]
    if not symbols:
        raise RuntimeError(f"S&P 500 file has no symbols: {sp500_file}")
    return symbols


def _write_cache(cache_file: Path, tickers: Iterable[str]) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text("\n".join(sorted(set(tickers))) + "\n", encoding="utf-8")


def _read_cache(cache_file: Path) -> List[str]:
    if not cache_file.exists():
        return []
    lines = cache_file.read_text(encoding="utf-8").splitlines()
    return [line.strip().upper() for line in lines if line.strip()]


def get_sp500_tickers(cache_file: Path, finnhub_api_key: str, sp500_file: str = "") -> List[str]:
    if sp500_file:
        try:
            tickers = _load_sp500_from_local_file(Path(sp500_file).expanduser().resolve())
            _write_cache(cache_file, tickers)
            print(f"[info] Loaded {len(tickers)} S&P 500 tickers from local file.")
            return tickers
        except Exception as exc:
            print(f"[warn] Local S&P 500 file source failed: {exc}")

    try:
        tickers = _load_sp500_from_wikipedia()
        _write_cache(cache_file, tickers)
        print(f"[info] Loaded {len(tickers)} S&P 500 tickers from Wikipedia.")
        return tickers
    except Exception as exc:
        print(f"[warn] Wikipedia source failed: {exc}")
        try:
            tickers = _load_sp500_from_github_csv()
            _write_cache(cache_file, tickers)
            print(f"[info] Loaded {len(tickers)} S&P 500 tickers from GitHub CSV mirror.")
            return tickers
        except Exception as github_exc:
            print(f"[warn] GitHub CSV source failed: {github_exc}")
        try:
            tickers = _load_sp500_from_finnhub(finnhub_api_key)
            _write_cache(cache_file, tickers)
            print(f"[info] Loaded {len(tickers)} S&P 500 tickers from Finnhub constituents.")
            return tickers
        except Exception as finnhub_exc:
            print(f"[warn] Finnhub constituents source failed: {finnhub_exc}")

        cached = _read_cache(cache_file)
        if cached:
            print(f"[warn] Using cached S&P 500 universe ({len(cached)} tickers).")
            return cached
        raise RuntimeError(
            "Failed to load S&P 500 tickers from all sources (local/Wikipedia/GitHub/Finnhub), and cache is empty."
        ) from exc


def get_earnings_symbols_for_date(date_str: str, finnhub_api_key: str, timeout_s: int = 20) -> List[str]:
    params = {"from": date_str, "to": date_str, "token": finnhub_api_key}
    response = requests.get(FINNHUB_EARNINGS_URL, params=params, timeout=timeout_s)
    response.raise_for_status()
    data = response.json()
    events = data.get("earningsCalendar", [])
    symbols: List[str] = []
    for event in events:
        symbol = str(event.get("symbol", "")).strip().upper()
        if symbol:
            symbols.append(symbol)
    return symbols


def _report_exists(project_root: Path, ticker: str, analysis_date: str) -> bool:
    report = project_root / "results" / ticker / analysis_date / "reports" / f"{ticker}_deep_value_intelligence_{analysis_date}.md"
    return report.exists()


def run_main_for_ticker(project_root: Path, python_bin: str, ticker: str, analysis_date: str) -> int:
    cmd = [python_bin, "main.py", "--ticker", ticker, "--date", analysis_date]
    print(f"[run] {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(project_root))
    return proc.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run daily earnings analyses for S&P 500 companies.")
    parser.add_argument("--date", default=_today_pacific(), help="Analysis date YYYY-MM-DD (default: today in America/Los_Angeles).")
    parser.add_argument("--python-bin", default=sys.executable, help="Python interpreter to run main.py.")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]), help="Project root containing main.py.")
    parser.add_argument("--cache-file", default=str(Path(__file__).resolve().parent / "cache" / "sp500_tickers.txt"), help="Local S&P 500 ticker cache file.")
    parser.add_argument("--sp500-file", default="", help="Optional local text/csv file containing S&P 500 symbols (one per line).")
    parser.add_argument("--max-tickers", type=int, default=0, help="Limit analyses to first N tickers (0 = no limit).")
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip ticker/date if report already exists (default: enabled).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print tickers to run, do not execute main.py.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    cache_file = Path(args.cache_file).resolve()

    if not (project_root / "main.py").exists():
        print(f"[error] main.py not found under project root: {project_root}")
        return 2

    finnhub_api_key = os.getenv("FINNHUB_API_KEY", "").strip()
    if not finnhub_api_key:
        print("[error] FINNHUB_API_KEY is required.")
        return 2

    try:
        sp500: Set[str] = set(get_sp500_tickers(cache_file, finnhub_api_key, args.sp500_file))
        earnings_symbols = get_earnings_symbols_for_date(args.date, finnhub_api_key)
    except Exception as exc:
        print(f"[error] Failed to load daily universe: {exc}")
        return 2

    targets = sorted(set(earnings_symbols) & sp500)
    if args.max_tickers > 0:
        targets = targets[: args.max_tickers]

    print(f"[info] Date={args.date} | Earnings symbols={len(earnings_symbols)} | S&P 500 earnings targets={len(targets)}")
    if not targets:
        print("[info] No S&P 500 earnings targets for date.")
        return 0

    failures = 0
    for ticker in targets:
        if args.skip_existing and _report_exists(project_root, ticker, args.date):
            print(f"[skip] Existing report found for {ticker} on {args.date}")
            continue
        if args.dry_run:
            print(f"[dry-run] Would run analysis for {ticker}")
            continue
        code = run_main_for_ticker(project_root, args.python_bin, ticker, args.date)
        if code != 0:
            failures += 1
            print(f"[fail] {ticker} exited with code {code}")

    if failures:
        print(f"[error] Completed with failures: {failures}")
        return 1
    print("[info] Completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
