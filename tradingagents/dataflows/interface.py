from typing import Annotated, Dict
from .reddit_utils import fetch_top_from_category
from .yfin_utils import *
from .stockstats_utils import *
from .googlenews_utils import *
from .finnhub_utils import get_data_in_range
from dateutil.relativedelta import relativedelta
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
import pandas as pd
from tqdm import tqdm
import yfinance as yf
from openai import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from .config import get_config, set_config, DATA_DIR

# Ticker mapping for problematic symbols
TICKER_MAPPINGS = {
    # Berkshire Hathaway variations
    "BRK.B": ["BRK-B", "BRKB", "BRK.B"],
    "BRK-B": ["BRK.B", "BRKB", "BRK-B"],
    "BRKB": ["BRK.B", "BRK-B", "BRKB"],
    "BRK.A": ["BRK-A", "BRKA", "BRK.A"],
    "BRK-A": ["BRK.A", "BRKA", "BRK-A"],
    "BRKA": ["BRK.A", "BRK-A", "BRKA"],
    # Add more mappings as needed
    "GOOGL": ["GOOGL", "GOOG"],
    "GOOG": ["GOOG", "GOOGL"],
}


def get_ticker_variations(ticker: str) -> list:
    """Get all possible ticker variations for a given ticker."""
    ticker_upper = ticker.upper()
    if ticker_upper in TICKER_MAPPINGS:
        return TICKER_MAPPINGS[ticker_upper]
    return [ticker_upper]


def try_with_ticker_variations(func, ticker: str, *args, **kwargs):
    """Try a function with different ticker variations until one succeeds."""
    variations = get_ticker_variations(ticker)
    errors = []
    
    for variation in variations:
        try:
            result = func(variation, *args, **kwargs)
            # Check if result is meaningful (not empty or error message)
            if result and not result.startswith("Error") and not "not found" in result.lower():
                return result
        except Exception as e:
            errors.append(f"{variation}: {str(e)}")
            continue
    
    # If all variations fail, return error summary
    return f"Unable to retrieve data for {ticker}. Tried variations: {variations}. Errors: {'; '.join(errors)}"


def get_enhanced_fundamentals(ticker: str, curr_date: str) -> str:
    """
    Enhanced fundamental data retrieval with multiple sources and ticker variations.
    Tries SimFin data first, then falls back to Google/OpenAI APIs.
    """
    results = []
    
    # Try SimFin data sources with ticker variations
    simfin_sources = [
        ("Balance Sheet", lambda t: get_simfin_balance_sheet(t, "annual", curr_date)),
        ("Income Statement", lambda t: get_simfin_income_statements(t, "annual", curr_date)),
        ("Cash Flow", lambda t: get_simfin_cashflow(t, "annual", curr_date)),
    ]
    
    for source_name, source_func in simfin_sources:
        result = try_with_ticker_variations(source_func, ticker)
        if result and not result.startswith("Unable to retrieve"):
            results.append(f"## {source_name} Data:\n{result}")
        else:
            results.append(f"## {source_name} Data: Not available ({result})")
    
    # Try quarterly data as fallback
    if not any("available" not in r for r in results):
        quarterly_sources = [
            ("Quarterly Balance Sheet", lambda t: get_simfin_balance_sheet(t, "quarterly", curr_date)),
            ("Quarterly Income Statement", lambda t: get_simfin_income_statements(t, "quarterly", curr_date)),
        ]
        
        for source_name, source_func in quarterly_sources:
            result = try_with_ticker_variations(source_func, ticker)
            if result and not result.startswith("Unable to retrieve"):
                results.append(f"## {source_name} Data:\n{result}")
    
    # If SimFin data is insufficient, try API-based fundamentals
    if len([r for r in results if "Not available" not in r]) < 2:
        config = get_config()
        api_result = ""
        
        # Try Google API if using Gemini models
        if (config.get("quick_think_llm", "").startswith(("gemini", "google")) or 
            config.get("deep_think_llm", "").startswith(("gemini", "google"))):
            try:
                api_result = get_fundamentals_google(ticker, curr_date)
            except:
                pass
        
        # Fallback to OpenAI API
        if not api_result:
            try:
                api_result = get_fundamentals_openai(ticker, curr_date)
            except:
                pass
        
        if api_result:
            results.append(f"## API-Based Fundamental Data:\n{api_result}")
    
    # Combine all results
    if results:
        header = f"# Comprehensive Fundamental Analysis for {ticker} as of {curr_date}\n\n"
        return header + "\n\n".join(results)
    else:
        return f"Unable to retrieve fundamental data for {ticker}. Consider using alternative tickers or checking data availability."


def get_finnhub_news(
    ticker: Annotated[
        str,
        "Search query of a company's, e.g. 'AAPL, TSM, etc.",
    ],
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "how many days to look back"],
):
    """
    Retrieve news about a company within a time frame

    Args
        ticker (str): ticker for the company you are interested in
        start_date (str): Start date in yyyy-mm-dd format
        end_date (str): End date in yyyy-mm-dd format
    Returns
        str: dataframe containing the news of the company in the time frame

    """

    start_date = datetime.strptime(curr_date, "%Y-%m-%d")
    before = start_date - relativedelta(days=look_back_days)
    before = before.strftime("%Y-%m-%d")

    result = get_data_in_range(ticker, before, curr_date, "news_data", DATA_DIR)

    if len(result) == 0:
        return ""

    combined_result = ""
    for day, data in result.items():
        if len(data) == 0:
            continue
        for entry in data:
            current_news = (
                "### " + entry["headline"] + f" ({day})" + "\n" + entry["summary"]
            )
            combined_result += current_news + "\n\n"

    return f"## {ticker} News, from {before} to {curr_date}:\n" + str(combined_result)


def get_finnhub_company_insider_sentiment(
    ticker: Annotated[str, "ticker symbol for the company"],
    curr_date: Annotated[
        str,
        "current date of you are trading at, yyyy-mm-dd",
    ],
    look_back_days: Annotated[int, "number of days to look back"],
):
    """
    Retrieve insider sentiment about a company (retrieved from public SEC information) for the past 15 days
    Args:
        ticker (str): ticker symbol of the company
        curr_date (str): current date you are trading on, yyyy-mm-dd
    Returns:
        str: a report of the sentiment in the past 15 days starting at curr_date
    """

    date_obj = datetime.strptime(curr_date, "%Y-%m-%d")
    before = date_obj - relativedelta(days=look_back_days)
    before = before.strftime("%Y-%m-%d")

    data = get_data_in_range(ticker, before, curr_date, "insider_senti", DATA_DIR)

    if len(data) == 0:
        return ""

    result_str = ""
    seen_dicts = []
    for date, senti_list in data.items():
        for entry in senti_list:
            if entry not in seen_dicts:
                result_str += f"### {entry['year']}-{entry['month']}:\nChange: {entry['change']}\nMonthly Share Purchase Ratio: {entry['mspr']}\n\n"
                seen_dicts.append(entry)

    return (
        f"## {ticker} Insider Sentiment Data for {before} to {curr_date}:\n"
        + result_str
        + "The change field refers to the net buying/selling from all insiders' transactions. The mspr field refers to monthly share purchase ratio."
    )


def get_finnhub_company_insider_transactions(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[
        str,
        "current date you are trading at, yyyy-mm-dd",
    ],
    look_back_days: Annotated[int, "how many days to look back"],
):
    """
    Retrieve insider transcaction information about a company (retrieved from public SEC information) for the past 15 days
    Args:
        ticker (str): ticker symbol of the company
        curr_date (str): current date you are trading at, yyyy-mm-dd
    Returns:
        str: a report of the company's insider transaction/trading informtaion in the past 15 days
    """

    date_obj = datetime.strptime(curr_date, "%Y-%m-%d")
    before = date_obj - relativedelta(days=look_back_days)
    before = before.strftime("%Y-%m-%d")

    data = get_data_in_range(ticker, before, curr_date, "insider_trans", DATA_DIR)

    if len(data) == 0:
        return ""

    result_str = ""

    seen_dicts = []
    for date, senti_list in data.items():
        for entry in senti_list:
            if entry not in seen_dicts:
                result_str += f"### Filing Date: {entry['filingDate']}, {entry['name']}:\nChange:{entry['change']}\nShares: {entry['share']}\nTransaction Price: {entry['transactionPrice']}\nTransaction Code: {entry['transactionCode']}\n\n"
                seen_dicts.append(entry)

    return (
        f"## {ticker} insider transactions from {before} to {curr_date}:\n"
        + result_str
        + "The change field reflects the variation in share count—here a negative number indicates a reduction in holdings—while share specifies the total number of shares involved. The transactionPrice denotes the per-share price at which the trade was executed, and transactionDate marks when the transaction occurred. The name field identifies the insider making the trade, and transactionCode (e.g., S for sale) clarifies the nature of the transaction. FilingDate records when the transaction was officially reported, and the unique id links to the specific SEC filing, as indicated by the source. Additionally, the symbol ties the transaction to a particular company, isDerivative flags whether the trade involves derivative securities, and currency notes the currency context of the transaction."
    )


def get_simfin_balance_sheet(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[
        str,
        "reporting frequency of the company's financial history: annual / quarterly",
    ],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
):
    data_path = os.path.join(
        DATA_DIR,
        "fundamental_data",
        "simfin_data_all",
        "balance_sheet",
        "companies",
        "us",
        f"us-balance-{freq}.csv",
    )
    df = pd.read_csv(data_path, sep=";")

    # Convert date strings to datetime objects and remove any time components
    df["Report Date"] = pd.to_datetime(df["Report Date"], utc=True).dt.normalize()
    df["Publish Date"] = pd.to_datetime(df["Publish Date"], utc=True).dt.normalize()

    # Convert the current date to datetime and normalize
    curr_date_dt = pd.to_datetime(curr_date, utc=True).normalize()

    # Filter the DataFrame for the given ticker and for reports that were published on or before the current date
    filtered_df = df[(df["Ticker"] == ticker) & (df["Publish Date"] <= curr_date_dt)]

    # Check if there are any available reports; if not, return a notification
    if filtered_df.empty:
        print("No balance sheet available before the given current date.")
        return ""

    # Get the most recent balance sheet by selecting the row with the latest Publish Date
    latest_balance_sheet = filtered_df.loc[filtered_df["Publish Date"].idxmax()]

    # drop the SimFinID column
    latest_balance_sheet = latest_balance_sheet.drop("SimFinId")

    return (
        f"## {freq} balance sheet for {ticker} released on {str(latest_balance_sheet['Publish Date'])[0:10]}: \n"
        + str(latest_balance_sheet)
        + "\n\nThis includes metadata like reporting dates and currency, share details, and a breakdown of assets, liabilities, and equity. Assets are grouped as current (liquid items like cash and receivables) and noncurrent (long-term investments and property). Liabilities are split between short-term obligations and long-term debts, while equity reflects shareholder funds such as paid-in capital and retained earnings. Together, these components ensure that total assets equal the sum of liabilities and equity."
    )


def get_simfin_cashflow(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[
        str,
        "reporting frequency of the company's financial history: annual / quarterly",
    ],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
):
    data_path = os.path.join(
        DATA_DIR,
        "fundamental_data",
        "simfin_data_all",
        "cash_flow",
        "companies",
        "us",
        f"us-cashflow-{freq}.csv",
    )
    df = pd.read_csv(data_path, sep=";")

    # Convert date strings to datetime objects and remove any time components
    df["Report Date"] = pd.to_datetime(df["Report Date"], utc=True).dt.normalize()
    df["Publish Date"] = pd.to_datetime(df["Publish Date"], utc=True).dt.normalize()

    # Convert the current date to datetime and normalize
    curr_date_dt = pd.to_datetime(curr_date, utc=True).normalize()

    # Filter the DataFrame for the given ticker and for reports that were published on or before the current date
    filtered_df = df[(df["Ticker"] == ticker) & (df["Publish Date"] <= curr_date_dt)]

    # Check if there are any available reports; if not, return a notification
    if filtered_df.empty:
        print("No cash flow statement available before the given current date.")
        return ""

    # Get the most recent cash flow statement by selecting the row with the latest Publish Date
    latest_cash_flow = filtered_df.loc[filtered_df["Publish Date"].idxmax()]

    # drop the SimFinID column
    latest_cash_flow = latest_cash_flow.drop("SimFinId")

    return (
        f"## {freq} cash flow statement for {ticker} released on {str(latest_cash_flow['Publish Date'])[0:10]}: \n"
        + str(latest_cash_flow)
        + "\n\nThis includes metadata like reporting dates and currency, share details, and a breakdown of cash movements. Operating activities show cash generated from core business operations, including net income adjustments for non-cash items and working capital changes. Investing activities cover asset acquisitions/disposals and investments. Financing activities include debt transactions, equity issuances/repurchases, and dividend payments. The net change in cash represents the overall increase or decrease in the company's cash position during the reporting period."
    )


def get_simfin_income_statements(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[
        str,
        "reporting frequency of the company's financial history: annual / quarterly",
    ],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
):
    data_path = os.path.join(
        DATA_DIR,
        "fundamental_data",
        "simfin_data_all",
        "income_statements",
        "companies",
        "us",
        f"us-income-{freq}.csv",
    )
    df = pd.read_csv(data_path, sep=";")

    # Convert date strings to datetime objects and remove any time components
    df["Report Date"] = pd.to_datetime(df["Report Date"], utc=True).dt.normalize()
    df["Publish Date"] = pd.to_datetime(df["Publish Date"], utc=True).dt.normalize()

    # Convert the current date to datetime and normalize
    curr_date_dt = pd.to_datetime(curr_date, utc=True).normalize()

    # Filter the DataFrame for the given ticker and for reports that were published on or before the current date
    filtered_df = df[(df["Ticker"] == ticker) & (df["Publish Date"] <= curr_date_dt)]

    # Check if there are any available reports; if not, return a notification
    if filtered_df.empty:
        print("No income statement available before the given current date.")
        return ""

    # Get the most recent income statement by selecting the row with the latest Publish Date
    latest_income = filtered_df.loc[filtered_df["Publish Date"].idxmax()]

    # drop the SimFinID column
    latest_income = latest_income.drop("SimFinId")

    return (
        f"## {freq} income statement for {ticker} released on {str(latest_income['Publish Date'])[0:10]}: \n"
        + str(latest_income)
        + "\n\nThis includes metadata like reporting dates and currency, share details, and a comprehensive breakdown of the company's financial performance. Starting with Revenue, it shows Cost of Revenue and resulting Gross Profit. Operating Expenses are detailed, including SG&A, R&D, and Depreciation. The statement then shows Operating Income, followed by non-operating items and Interest Expense, leading to Pretax Income. After accounting for Income Tax and any Extraordinary items, it concludes with Net Income, representing the company's bottom-line profit or loss for the period."
    )


def get_google_news(
    query: Annotated[str, "Query to search with"],
    curr_date: Annotated[str, "Curr date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:
    from .googlenews_utils import getNewsDataWithFallback
    
    query = query.replace(" ", "+")

    # Use intelligent fallback to get the most recent available data
    result = getNewsDataWithFallback(query, curr_date, max_lookback_days=look_back_days)
    
    news_results = result["data"]
    actual_date_range = result["actual_date_range"]
    fallback_used = result["fallback_used"]

    news_str = ""
    
    # Add data quality note if fallback was used
    if fallback_used:
        news_str += f"**Data Quality Note**: Target date {curr_date} data not available. Using most recent data from {actual_date_range}.\n\n"

    for news in news_results:
        news_str += (
            f"### {news['title']} (source: {news['source']}) \n\n{news['snippet']}\n\n"
        )

    if len(news_results) == 0:
        if "error" in result:
            return f"## {query} Google News Search Result:\n\nNo news data found for {query} within {look_back_days} days of {curr_date}. {result.get('description', '')}"
        return ""

    return f"## {query} Google News, from {actual_date_range}:\n\n{news_str}"


def get_reddit_global_news(
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "how many days to look back"],
    max_limit_per_day: Annotated[int, "Maximum number of news per day"],
) -> str:
    """
    Retrieve the latest top reddit news
    Args:
        start_date: Start date in yyyy-mm-dd format
        end_date: End date in yyyy-mm-dd format
    Returns:
        str: A formatted dataframe containing the latest news articles posts on reddit and meta information in these columns: "created_utc", "id", "title", "selftext", "score", "num_comments", "url"
    """

    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    before = start_date - relativedelta(days=look_back_days)
    before = before.strftime("%Y-%m-%d")

    posts = []
    # iterate from start_date to end_date
    curr_date = datetime.strptime(before, "%Y-%m-%d")

    total_iterations = (start_date - curr_date).days + 1
    pbar = tqdm(desc=f"Getting Global News on {start_date}", total=total_iterations)

    while curr_date <= start_date:
        curr_date_str = curr_date.strftime("%Y-%m-%d")
        fetch_result = fetch_top_from_category(
            "global_news",
            curr_date_str,
            max_limit_per_day,
            data_path=os.path.join(DATA_DIR, "reddit_data"),
        )
        posts.extend(fetch_result)
        curr_date += relativedelta(days=1)
        pbar.update(1)

    pbar.close()

    if len(posts) == 0:
        return ""

    news_str = ""
    for post in posts:
        if post["content"] == "":
            news_str += f"### {post['title']}\n\n"
        else:
            news_str += f"### {post['title']}\n\n{post['content']}\n\n"

    return f"## Global News Reddit, from {before} to {curr_date}:\n{news_str}"


def get_reddit_company_news(
    ticker: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "how many days to look back"],
    max_limit_per_day: Annotated[int, "Maximum number of news per day"],
) -> str:
    """
    Retrieve the latest top reddit news
    Args:
        ticker: ticker symbol of the company
        start_date: Start date in yyyy-mm-dd format
        end_date: End date in yyyy-mm-dd format
    Returns:
        str: A formatted dataframe containing the latest news articles posts on reddit and meta information in these columns: "created_utc", "id", "title", "selftext", "score", "num_comments", "url"
    """

    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    before = start_date - relativedelta(days=look_back_days)
    before = before.strftime("%Y-%m-%d")

    posts = []
    # iterate from start_date to end_date
    curr_date = datetime.strptime(before, "%Y-%m-%d")

    total_iterations = (start_date - curr_date).days + 1
    pbar = tqdm(
        desc=f"Getting Company News for {ticker} on {start_date}",
        total=total_iterations,
    )

    while curr_date <= start_date:
        curr_date_str = curr_date.strftime("%Y-%m-%d")
        fetch_result = fetch_top_from_category(
            "company_news",
            curr_date_str,
            max_limit_per_day,
            ticker,
            data_path=os.path.join(DATA_DIR, "reddit_data"),
        )
        posts.extend(fetch_result)
        curr_date += relativedelta(days=1)

        pbar.update(1)

    pbar.close()

    if len(posts) == 0:
        return ""

    news_str = ""
    for post in posts:
        if post["content"] == "":
            news_str += f"### {post['title']}\n\n"
        else:
            news_str += f"### {post['title']}\n\n{post['content']}\n\n"

    return f"##{ticker} News Reddit, from {before} to {curr_date}:\n\n{news_str}"


def get_stock_stats_indicators_window(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[
        str, "The current trading date you are trading on, YYYY-mm-dd"
    ],
    look_back_days: Annotated[int, "how many days to look back"],
    online: Annotated[bool, "to fetch data online or offline"],
) -> str:

    best_ind_params = {
        # Moving Averages
        "close_50_sma": (
            "50 SMA: A medium-term trend indicator. "
            "Usage: Identify trend direction and serve as dynamic support/resistance. "
            "Tips: It lags price; combine with faster indicators for timely signals."
        ),
        "close_200_sma": (
            "200 SMA: A long-term trend benchmark. "
            "Usage: Confirm overall market trend and identify golden/death cross setups. "
            "Tips: It reacts slowly; best for strategic trend confirmation rather than frequent trading entries."
        ),
        "close_10_ema": (
            "10 EMA: A responsive short-term average. "
            "Usage: Capture quick shifts in momentum and potential entry points. "
            "Tips: Prone to noise in choppy markets; use alongside longer averages for filtering false signals."
        ),
        # MACD Related
        "macd": (
            "MACD: Computes momentum via differences of EMAs. "
            "Usage: Look for crossovers and divergence as signals of trend changes. "
            "Tips: Confirm with other indicators in low-volatility or sideways markets."
        ),
        "macds": (
            "MACD Signal: An EMA smoothing of the MACD line. "
            "Usage: Use crossovers with the MACD line to trigger trades. "
            "Tips: Should be part of a broader strategy to avoid false positives."
        ),
        "macdh": (
            "MACD Histogram: Shows the gap between the MACD line and its signal. "
            "Usage: Visualize momentum strength and spot divergence early. "
            "Tips: Can be volatile; complement with additional filters in fast-moving markets."
        ),
        # Momentum Indicators
        "rsi": (
            "RSI: Measures momentum to flag overbought/oversold conditions. "
            "Usage: Apply 70/30 thresholds and watch for divergence to signal reversals. "
            "Tips: In strong trends, RSI may remain extreme; always cross-check with trend analysis."
        ),
        # Volatility Indicators
        "boll": (
            "Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands. "
            "Usage: Acts as a dynamic benchmark for price movement. "
            "Tips: Combine with the upper and lower bands to effectively spot breakouts or reversals."
        ),
        "boll_ub": (
            "Bollinger Upper Band: Typically 2 standard deviations above the middle line. "
            "Usage: Signals potential overbought conditions and breakout zones. "
            "Tips: Confirm signals with other tools; prices may ride the band in strong trends."
        ),
        "boll_lb": (
            "Bollinger Lower Band: Typically 2 standard deviations below the middle line. "
            "Usage: Indicates potential oversold conditions. "
            "Tips: Use additional analysis to avoid false reversal signals."
        ),
        "atr": (
            "ATR: Averages true range to measure volatility. "
            "Usage: Set stop-loss levels and adjust position sizes based on current market volatility. "
            "Tips: It's a reactive measure, so use it as part of a broader risk management strategy."
        ),
        # Volume-Based Indicators
        "vwma": (
            "VWMA: A moving average weighted by volume. "
            "Usage: Confirm trends by integrating price action with volume data. "
            "Tips: Watch for skewed results from volume spikes; use in combination with other volume analyses."
        ),
        "mfi": (
            "MFI: The Money Flow Index is a momentum indicator that uses both price and volume to measure buying and selling pressure. "
            "Usage: Identify overbought (>80) or oversold (<20) conditions and confirm the strength of trends or reversals. "
            "Tips: Use alongside RSI or MACD to confirm signals; divergence between price and MFI can indicate potential reversals."
        ),
    }

    if indicator not in best_ind_params:
        raise ValueError(
            f"Indicator {indicator} is not supported. Please choose from: {list(best_ind_params.keys())}"
        )

    end_date = curr_date
    curr_date = datetime.strptime(curr_date, "%Y-%m-%d")
    before = curr_date - relativedelta(days=look_back_days)

    if not online:
        # read from YFin data
        data = pd.read_csv(
            os.path.join(
                DATA_DIR,
                f"market_data/price_data/{symbol}-YFin-data-2015-01-01-2025-03-25.csv",
            )
        )
        data["Date"] = pd.to_datetime(data["Date"], utc=True)
        dates_in_df = data["Date"].astype(str).str[:10]

        ind_string = ""
        while curr_date >= before:
            # only do the trading dates
            if curr_date.strftime("%Y-%m-%d") in dates_in_df.values:
                indicator_value = get_stockstats_indicator(
                    symbol, indicator, curr_date.strftime("%Y-%m-%d"), online
                )

                ind_string += f"{curr_date.strftime('%Y-%m-%d')}: {indicator_value}\n"

            curr_date = curr_date - relativedelta(days=1)
    else:
        # online gathering
        ind_string = ""
        while curr_date >= before:
            indicator_value = get_stockstats_indicator(
                symbol, indicator, curr_date.strftime("%Y-%m-%d"), online
            )

            ind_string += f"{curr_date.strftime('%Y-%m-%d')}: {indicator_value}\n"

            curr_date = curr_date - relativedelta(days=1)

    result_str = (
        f"## {indicator} values from {before.strftime('%Y-%m-%d')} to {end_date}:\n\n"
        + ind_string
        + "\n\n"
        + best_ind_params.get(indicator, "No description available.")
    )

    return result_str


def get_stockstats_indicator(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[
        str, "The current trading date you are trading on, YYYY-mm-dd"
    ],
    online: Annotated[bool, "to fetch data online or offline"],
) -> str:

    curr_date = datetime.strptime(curr_date, "%Y-%m-%d")
    curr_date = curr_date.strftime("%Y-%m-%d")

    try:
        indicator_value = StockstatsUtils.get_stock_stats(
            symbol,
            indicator,
            curr_date,
            os.path.join(DATA_DIR, "market_data", "price_data"),
            online=online,
        )
    except Exception as e:
        print(
            f"Error getting stockstats indicator data for indicator {indicator} on {curr_date}: {e}"
        )
        return ""

    return str(indicator_value)


def get_YFin_data_window(
    symbol: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:
    # calculate past days
    date_obj = datetime.strptime(curr_date, "%Y-%m-%d")
    before = date_obj - relativedelta(days=look_back_days)
    start_date = before.strftime("%Y-%m-%d")

    # read in data
    data = pd.read_csv(
        os.path.join(
            DATA_DIR,
            f"market_data/price_data/{symbol}-YFin-data-2015-01-01-2025-03-25.csv",
        )
    )

    # Extract just the date part for comparison
    data["DateOnly"] = data["Date"].str[:10]

    # Filter data between the start and end dates (inclusive)
    filtered_data = data[
        (data["DateOnly"] >= start_date) & (data["DateOnly"] <= curr_date)
    ]

    # Drop the temporary column we created
    filtered_data = filtered_data.drop("DateOnly", axis=1)

    # Set pandas display options to show the full DataFrame
    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None, "display.width", None
    ):
        df_string = filtered_data.to_string()

    return (
        f"## Raw Market Data for {symbol} from {start_date} to {curr_date}:\n\n"
        + df_string
    )


def get_YFin_data_online(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
):

    datetime.strptime(start_date, "%Y-%m-%d")
    datetime.strptime(end_date, "%Y-%m-%d")

    # Create ticker object
    ticker = yf.Ticker(symbol.upper())

    # Fetch historical data for the specified date range
    data = ticker.history(start=start_date, end=end_date)

    # Check if data is empty
    if data.empty:
        return (
            f"No data found for symbol '{symbol}' between {start_date} and {end_date}"
        )

    # Remove timezone info from index for cleaner output
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    # Round numerical values to 2 decimal places for cleaner display
    numeric_columns = ["Open", "High", "Low", "Close", "Adj Close"]
    for col in numeric_columns:
        if col in data.columns:
            data[col] = data[col].round(2)

    # Convert DataFrame to CSV string
    csv_string = data.to_csv()

    # Add header information
    header = f"# Stock data for {symbol.upper()} from {start_date} to {end_date}\n"
    header += f"# Total records: {len(data)}\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    return header + csv_string


def get_YFin_data(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    # read in data
    data = pd.read_csv(
        os.path.join(
            DATA_DIR,
            f"market_data/price_data/{symbol}-YFin-data-2015-01-01-2025-03-25.csv",
        )
    )

    if end_date > "2025-03-25":
        raise Exception(
            f"Get_YFin_Data: {end_date} is outside of the data range of 2015-01-01 to 2025-03-25"
        )

    # Extract just the date part for comparison
    data["DateOnly"] = data["Date"].str[:10]

    # Filter data between the start and end dates (inclusive)
    filtered_data = data[
        (data["DateOnly"] >= start_date) & (data["DateOnly"] <= end_date)
    ]

    # Drop the temporary column we created
    filtered_data = filtered_data.drop("DateOnly", axis=1)

    # remove the index from the dataframe
    filtered_data = filtered_data.reset_index(drop=True)

    return filtered_data


def get_stock_news_openai(ticker, curr_date):
    config = get_config()
    
    if not config.get("openai_api_key"):
        return "Error: OPENAI_API_KEY is required for OpenAI news functionality. Please set the environment variable or use get_stock_news_google as an alternative."
    
    # Always use OpenAI API for OpenAI-specific functions
    client = OpenAI(
        api_key=config.get("openai_api_key"),
        base_url=config.get("openai_api_base", "https://api.openai.com/v1")
    )

    response = client.responses.create(
        model=config["quick_think_llm"],
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"Can you search Social Media for {ticker} from 7 days before {curr_date} to {curr_date}? Make sure you only get the data posted during that period.",
                    }
                ],
            }
        ],
        text={"format": {"type": "text"}},
        reasoning={},
        tools=[
            {
                "type": "web_search_preview",
                "user_location": {"type": "approximate"},
                "search_context_size": "low",
            }
        ],
        temperature=1,
        max_output_tokens=4096,
        top_p=1,
        store=True,
    )

    return response.output[1].content[0].text


def get_fundamentals_openai(ticker, curr_date):
    config = get_config()
    
    if not config.get("openai_api_key"):
        return "Error: OPENAI_API_KEY is required for OpenAI fundamentals functionality. Please set the environment variable or use get_fundamentals_google/get_enhanced_fundamentals as alternatives."
    
    # Always use OpenAI API for OpenAI-specific functions
    client = OpenAI(
        api_key=config.get("openai_api_key"),
        base_url=config.get("openai_api_base", "https://api.openai.com/v1")
    )

    response = client.responses.create(
        model=config["quick_think_llm"],
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"Can you search Fundamental for discussions on {ticker} during of the month before {curr_date} to the month of {curr_date}. Make sure you only get the data posted during that period. List as a table, with PE/PS/Cash flow/ etc",
                    }
                ],
            }
        ],
        text={"format": {"type": "text"}},
        reasoning={},
        tools=[
            {
                "type": "web_search_preview",
                "user_location": {"type": "approximate"},
                "search_context_size": "low",
            }
        ],
        temperature=1,
        max_output_tokens=4096,
        top_p=1,
        store=True,
    )

    return response.output[1].content[0].text


def get_stock_news_google(ticker, curr_date):
    """
    Retrieve the latest news about a given stock using Google Gemini API.
    This is an alternative to get_stock_news_openai when using Google models.
    """
    config = get_config()
    
    if not config.get("google_api_key"):
        return "Error: Google API key not configured. Please set GOOGLE_API_KEY environment variable."
    
    try:
        from datetime import datetime, timedelta
        
        # Parse the current date and calculate fallback dates
        try:
            target_date = datetime.strptime(curr_date, "%Y-%m-%d")
            one_week_ago = (target_date - timedelta(days=7)).strftime("%Y-%m-%d")
            one_month_ago = (target_date - timedelta(days=30)).strftime("%Y-%m-%d")
        except:
            target_date = datetime.now()
            one_week_ago = (target_date - timedelta(days=7)).strftime("%Y-%m-%d")
            one_month_ago = (target_date - timedelta(days=30)).strftime("%Y-%m-%d")
            curr_date = target_date.strftime("%Y-%m-%d")
        
        client = ChatGoogleGenerativeAI(
            model=config["quick_think_llm"],
            google_api_key=config["google_api_key"]
        )
        
        prompt = f"""Provide a comprehensive analysis of {ticker} for investment decisions. Target analysis date: {curr_date}

        **Priority 1: Try to find information as close to {curr_date} as possible**
        **Priority 2: If {curr_date} data unavailable, use the most recent data before this date**
        **Priority 3: Focus on data from the past week ({one_week_ago} to {curr_date}) if available**
        **Priority 4: Fall back to data from the past month ({one_month_ago} to {curr_date}) if needed**

        Please provide:
        1. **Recent News & Developments**: Latest company announcements, earnings, strategic changes
        2. **Social Media Sentiment**: Market discussions, investor sentiment, trending topics
        3. **Financial Performance**: Recent financial metrics, analyst ratings, price movements
        4. **Market Context**: Industry trends, competitive landscape, regulatory changes
        5. **Investment Signals**: Buy/sell recommendations, target prices, risk factors

        **IMPORTANT INSTRUCTIONS:**
        - Always specify the actual date range of the information you're providing
        - If you cannot find data for {curr_date}, clearly state what date range you're using instead
        - Prioritize the most recent available information before {curr_date}
        - If using older data, explain why more recent data isn't available
        - Structure your response with clear headings and cite timeframes for each section

        **Data Quality Note**: Please indicate the confidence level and recency of your information sources."""
        
        response = client.invoke(prompt)
        return response.content
        
    except Exception as e:
        return f"Error retrieving news with Google API: {str(e)}. Attempting to fall back to alternative news sources or cached data if available."


def get_global_news_google(curr_date):
    """
    Retrieve global/macroeconomic news using Google Gemini API.
    This is an alternative to get_global_news_openai when using Google models.
    """
    config = get_config()
    
    if not config.get("google_api_key"):
        return "Error: Google API key not configured. Please set GOOGLE_API_KEY environment variable."
    
    try:
        client = ChatGoogleGenerativeAI(
            model=config["quick_think_llm"],
            google_api_key=config["google_api_key"]
        )
        
        # Calculate fallback date ranges
        from datetime import datetime, timedelta
        
        try:
            target_date = datetime.strptime(curr_date, "%Y-%m-%d")
            one_week_ago = (target_date - timedelta(days=7)).strftime("%Y-%m-%d")
            two_weeks_ago = (target_date - timedelta(days=14)).strftime("%Y-%m-%d")
            one_month_ago = (target_date - timedelta(days=30)).strftime("%Y-%m-%d")
        except:
            target_date = datetime.now()
            one_week_ago = (target_date - timedelta(days=7)).strftime("%Y-%m-%d")
            two_weeks_ago = (target_date - timedelta(days=14)).strftime("%Y-%m-%d")
            one_month_ago = (target_date - timedelta(days=30)).strftime("%Y-%m-%d")
            curr_date = target_date.strftime("%Y-%m-%d")
        
        prompt = f"""Provide comprehensive global macroeconomic analysis for investment decisions. Target date: {curr_date}

        **Data Priority Strategy:**
        1. **Primary**: Information as close to {curr_date} as possible
        2. **Fallback 1**: Most recent data from the past week ({one_week_ago} to {curr_date})
        3. **Fallback 2**: Data from past two weeks ({two_weeks_ago} to {curr_date})
        4. **Fallback 3**: Data from past month ({one_month_ago} to {curr_date})

        **Required Analysis Areas:**
        1. **Monetary Policy**: Federal Reserve decisions, interest rate changes, quantitative easing
        2. **Economic Indicators**: GDP growth, inflation (CPI/PCE), employment data, consumer confidence
        3. **Geopolitical Events**: Trade wars, sanctions, military conflicts affecting markets
        4. **Central Bank Actions**: ECB, Bank of Japan, Bank of England policy changes
        5. **Market Dynamics**: VIX levels, currency fluctuations, commodity prices
        6. **Risk Assessment**: Systemic risks, market volatility drivers, crisis indicators

        **Critical Instructions:**
        - **ALWAYS specify the actual date range** of information you're providing
        - If {curr_date} data is unavailable, clearly state what earlier period you're analyzing
        - Prioritize the most recent available information before {curr_date}
        - Explain any data gaps or limitations
        - Focus on events and trends that impact investment decisions
        - Structure with clear headings showing timeframes

        **Output Requirements:**
        - Begin with a data quality disclaimer showing your information timeframe
        - Use recent data even if it's from before {curr_date}
        - Indicate confidence levels for different types of information"""
        
        response = client.invoke(prompt)
        return response.content
        
    except Exception as e:
        return f"Error retrieving global news with Google API: {str(e)}. You may want to try using alternative news sources like get_google_news or get_reddit_news."


def get_fundamentals_google(ticker, curr_date):
    """
    Retrieve fundamental analysis information using Google Gemini API.
    This is an alternative to get_fundamentals_openai when using Google models.
    """
    config = get_config()
    
    if not config.get("google_api_key"):
        return "Error: Google API key not configured. Please set GOOGLE_API_KEY environment variable."
    
    try:
        client = ChatGoogleGenerativeAI(
            model=config["quick_think_llm"],
            google_api_key=config["google_api_key"],
            temperature=0.1,
            max_output_tokens=4000
        )
        
        # Calculate fallback date ranges for fundamental analysis
        from datetime import datetime, timedelta
        
        try:
            target_date = datetime.strptime(curr_date, "%Y-%m-%d")
            one_quarter_ago = (target_date - timedelta(days=90)).strftime("%Y-%m-%d")
            six_months_ago = (target_date - timedelta(days=180)).strftime("%Y-%m-%d")
            one_year_ago = (target_date - timedelta(days=365)).strftime("%Y-%m-%d")
        except:
            target_date = datetime.now()
            one_quarter_ago = (target_date - timedelta(days=90)).strftime("%Y-%m-%d")
            six_months_ago = (target_date - timedelta(days=180)).strftime("%Y-%m-%d")
            one_year_ago = (target_date - timedelta(days=365)).strftime("%Y-%m-%d")
            curr_date = target_date.strftime("%Y-%m-%d")
        
        prompt = f"""Provide comprehensive fundamental analysis for {ticker}. Target analysis date: {curr_date}

        **Data Priority Strategy:**
        1. **Primary**: Most recent financial data available as of {curr_date}
        2. **Fallback 1**: Latest quarterly data (within past 3 months from {one_quarter_ago})
        3. **Fallback 2**: Recent semi-annual data (within 6 months from {six_months_ago})
        4. **Fallback 3**: Annual data (within past year from {one_year_ago})

        **REQUIRED FUNDAMENTAL ANALYSIS:**
        1. **Current Valuation Metrics**: P/E, P/B, EV/EBITDA, P/S ratios with effective dates
        2. **Profitability Analysis**: ROE, ROA, ROIC, gross/operating/net margins with trend analysis
        3. **Growth Assessment**: Revenue, earnings, FCF growth (YoY, 3Y, 5Y trends where available)
        4. **Balance Sheet Health**: Debt ratios, liquidity, cash position, working capital trends
        5. **Recent Financial Performance**: Latest quarterly results, YoY comparisons, guidance updates
        6. **Competitive Position**: Industry ranking, market share, competitive advantages
        7. **Investment Quality Scores**: Dividend yield, payout ratio, earnings quality, management effectiveness

        **CRITICAL REQUIREMENTS:**
        - **Always specify the exact date/quarter** of financial data you're referencing
        - If {curr_date} data unavailable, clearly state what period you're analyzing instead
        - Use the most recent available data before {curr_date}
        - Compare current metrics to historical averages (3-5 year trends)
        - Include data quality disclaimers for any estimates or outdated information
        - Prioritize accuracy over completeness - better to provide recent verified data than guess

        **Output Format:**
        - Start with data quality note specifying your information timeframe
        - Use specific numbers with dates/quarters where available
        - Flag any significant data gaps or limitations
        - Structure with clear headings showing data recency"""
        
        response = client.invoke(prompt)
        
        if response and response.content and len(response.content.strip()) > 50:
            return f"# Google Fundamentals Analysis for {ticker}\n\n{response.content}"
        else:
            return f"Google API returned insufficient data for {ticker}. Try using get_enhanced_fundamentals for comprehensive fundamental analysis from multiple sources."
        
    except Exception as e:
        error_msg = f"Error retrieving fundamentals with Google API: {str(e)}"
        fallback_msg = "Available alternative tools: get_enhanced_fundamentals, get_simfin_balance_sheet, get_simfin_income_stmt, get_simfin_cashflow"
        return f"{error_msg}\n\n{fallback_msg}"


def get_global_news_openai(curr_date):
    config = get_config()
    
    if not config.get("openai_api_key"):
        return "Error: OPENAI_API_KEY is required for OpenAI news functionality. Please set the environment variable or use get_global_news_google as an alternative."
    
    # Always use OpenAI API for OpenAI-specific functions
    client = OpenAI(
        api_key=config.get("openai_api_key"),
        base_url=config.get("openai_api_base", "https://api.openai.com/v1")
    )

    response = client.responses.create(
        model=config["quick_think_llm"],
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"Can you search global or macroeconomics news from 7 days before {curr_date} to {curr_date} that would be informative for trading purposes? Make sure you only get the data posted during that period.",
                    }
                ],
            }
        ],
        text={"format": {"type": "text"}},
        reasoning={},
        tools=[
            {
                "type": "web_search_preview",
                "user_location": {"type": "approximate"},
                "search_context_size": "low",
            }
        ],
        temperature=1,
        max_output_tokens=4096,
        top_p=1,
        store=True,
    )

    return response.output[1].content[0].text


def get_fundamentals_openai(ticker, curr_date):
    config = get_config()
    
    if not config.get("openai_api_key"):
        return "Error: OPENAI_API_KEY is required for OpenAI fundamentals functionality. Please set the environment variable or use get_fundamentals_google/get_enhanced_fundamentals as alternatives."
    
    # Always use OpenAI API for OpenAI-specific functions
    client = OpenAI(
        api_key=config.get("openai_api_key"),
        base_url=config.get("openai_api_base", "https://api.openai.com/v1")
    )

    response = client.responses.create(
        model=config["quick_think_llm"],
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"Can you search Fundamental for discussions on {ticker} during of the month before {curr_date} to the month of {curr_date}. Make sure you only get the data posted during that period. List as a table, with PE/PS/Cash flow/ etc",
                    }
                ],
            }
        ],
        text={"format": {"type": "text"}},
        reasoning={},
        tools=[
            {
                "type": "web_search_preview",
                "user_location": {"type": "approximate"},
                "search_context_size": "low",
            }
        ],
        temperature=1,
        max_output_tokens=4096,
        top_p=1,
        store=True,
    )

    return response.output[1].content[0].text

