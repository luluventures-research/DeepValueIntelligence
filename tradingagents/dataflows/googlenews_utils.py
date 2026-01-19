import json
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import time
import random
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    retry_if_result,
)


def is_rate_limited(response):
    """Check if the response indicates rate limiting (status code 429)"""
    return response.status_code == 429


@retry(
    retry=(retry_if_result(is_rate_limited)),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5),
)
def make_request(url, headers):
    """Make a request with retry logic for rate limiting"""
    # Random delay before each request to avoid detection
    time.sleep(random.uniform(2, 6))
    response = requests.get(url, headers=headers)
    return response


def getNewsData(query, start_date, end_date):
    """
    Scrape Google News search results for a given query and date range.
    Implements intelligent fallback to find the most recent available data.
    
    query: str - search query
    start_date: str - start date in the format yyyy-mm-dd or mm/dd/yyyy
    end_date: str - end date in the format yyyy-mm-dd or mm/dd/yyyy
    """
    if "-" in start_date:
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        start_date = start_date.strftime("%m/%d/%Y")
    if "-" in end_date:
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
        end_date = end_date.strftime("%m/%d/%Y")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/101.0.4951.54 Safari/537.36"
        )
    }

    news_results = []
    page = 0
    while True:
        offset = page * 10
        url = (
            f"https://www.google.com/search?q={query}"
            f"&tbs=cdr:1,cd_min:{start_date},cd_max:{end_date}"
            f"&tbm=nws&start={offset}"
        )

        try:
            response = make_request(url, headers)
            if response.status_code != 200:
                print(f"HTTP error {response.status_code} for URL: {url}")
                break
                
            soup = BeautifulSoup(response.content, "html.parser")
            results_on_page = soup.select("div.SoaBEf")

            if not results_on_page:
                print(f"No results found on page {page + 1}, ending search.")
                break  # No more results found

            for el in results_on_page:
                try:
                    # Safely extract link
                    link_element = el.find("a")
                    if not link_element or "href" not in link_element.attrs:
                        continue
                    link = link_element["href"]
                    
                    # Safely extract title
                    title_element = el.select_one("div.MBeuO")
                    title = title_element.get_text() if title_element else "No title available"
                    
                    # Safely extract snippet
                    snippet_element = el.select_one(".GI74Re")
                    snippet = snippet_element.get_text() if snippet_element else "No snippet available"
                    
                    # Safely extract date
                    date_element = el.select_one(".LfVVr")
                    date = date_element.get_text() if date_element else "No date available"
                    
                    # Safely extract source
                    source_element = el.select_one(".NUnG9d span")
                    source = source_element.get_text() if source_element else "Unknown source"
                    
                    news_results.append(
                        {
                            "link": link,
                            "title": title,
                            "snippet": snippet,
                            "date": date,
                            "source": source,
                        }
                    )
                except Exception as e:
                    print(f"Error processing result: {e}")
                    # If one of the fields is not found, skip this result
                    continue

            # Update the progress bar with the current count of results scraped

            # Check for the "Next" link (pagination)
            next_link = soup.find("a", id="pnnext")
            if not next_link:
                break

            page += 1

        except Exception as e:
            print(f"Failed after multiple retries: {e}")
            break

    return news_results


def getNewsDataWithFallback(query, target_date, max_lookback_days=30):
    """
    Get news data with intelligent fallback to recent dates if target date has no results.
    
    Args:
        query: str - search query
        target_date: str - target date in YYYY-MM-DD format
        max_lookback_days: int - maximum days to look back for data
    
    Returns:
        dict with 'data', 'actual_date_range', and 'fallback_used' keys
    """
    try:
        target_dt = datetime.strptime(target_date, "%Y-%m-%d")
    except ValueError:
        # If date parsing fails, use current date
        target_dt = datetime.now()
        target_date = target_dt.strftime("%Y-%m-%d")
    
    # Define fallback date ranges (in order of preference)
    fallback_ranges = [
        # Primary: target date ± 3 days
        {
            "start": (target_dt - timedelta(days=3)).strftime("%Y-%m-%d"),
            "end": target_date,
            "description": f"3 days before {target_date}"
        },
        # Fallback 1: past week
        {
            "start": (target_dt - timedelta(days=7)).strftime("%Y-%m-%d"),
            "end": target_date,
            "description": f"Week before {target_date}"
        },
        # Fallback 2: past 2 weeks
        {
            "start": (target_dt - timedelta(days=14)).strftime("%Y-%m-%d"),
            "end": target_date,
            "description": f"2 weeks before {target_date}"
        },
        # Fallback 3: past month
        {
            "start": (target_dt - timedelta(days=max_lookback_days)).strftime("%Y-%m-%d"),
            "end": target_date,
            "description": f"{max_lookback_days} days before {target_date}"
        }
    ]
    
    for i, date_range in enumerate(fallback_ranges):
        try:
            print(f"Attempting news search: {date_range['description']}")
            news_data = getNewsData(query, date_range["start"], date_range["end"])
            
            if news_data and len(news_data) > 0:
                return {
                    "data": news_data,
                    "actual_date_range": f"{date_range['start']} to {date_range['end']}",
                    "fallback_used": i > 0,
                    "fallback_level": i,
                    "description": date_range["description"]
                }
        except Exception as e:
            print(f"Error searching {date_range['description']}: {e}")
            continue
    
    # If all fallbacks fail, return empty with explanation
    return {
        "data": [],
        "actual_date_range": "No data available",
        "fallback_used": True,
        "fallback_level": len(fallback_ranges),
        "description": f"No news data found within {max_lookback_days} days of {target_date}",
        "error": "All date ranges exhausted without finding news data"
    }
