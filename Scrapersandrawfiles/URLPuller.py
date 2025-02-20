import os
import pandas as pd
import datetime
import calendar
from serpapi import GoogleSearch  

def generate_monthly_date_ranges(start_year, start_month, end_year, end_month):
    """
    Generate a list of (start_date, end_date) tuples for each month between the given start and end.
    Dates are returned in MM/DD/YYYY format.
    """
    ranges = []
    current_year = start_year
    current_month = start_month
    while (current_year < end_year) or (current_year == end_year and current_month <= end_month):
        start_date = datetime.date(current_year, current_month, 1)
        last_day = calendar.monthrange(current_year, current_month)[1]
        end_date = datetime.date(current_year, current_month, last_day)
        ranges.append((start_date.strftime("%m/%d/%Y"), end_date.strftime("%m/%d/%Y")))
        # Move to the next month.
        if current_month == 12:
            current_month = 1
            current_year += 1
        else:
            current_month += 1
    return ranges

def fetch_urls_with_date(query, source, start_date, end_date, num_results=20, serpapi_key="YOUR_SERPAPI_KEY"):
    """
    Uses SerpAPI to search Google for URLs from a given source (domain) matching the query,
    restricted to a specific date range (MM/DD/YYYY).
    """
    tbs_value = f"cdr:1,cd_min:{start_date},cd_max:{end_date}"
    params = {
        "engine": "google",
        "q": f"site:{source} {query}",
        "api_key": serpapi_key,
        "num": num_results,
        "tbs": tbs_value
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    urls = []
    for result in results.get("organic_results", []):
        link = result.get("link")
        if link:
            urls.append(link)
    return urls

def main():
    # Define your search query.
    query = "NVIDIA Stock OR NVDA OR NVIDIA"
    
    # Define the target sources.
    sources = [
        #"forbes.com",
        #"fool.com",
        "wsj.com"
        #"bloomberg.com",
        #"cnbc.com"
    ]
    
    # For simplicity, we hard-code our time range: January 2022 through February 2025 (38 months).
    start_year, start_month = 2022, 1
    end_year, end_month = 2025, 2
    monthly_ranges = generate_monthly_date_ranges(start_year, start_month, end_year, end_month)
    
    # We want exactly 13 URLs per month.
    target_per_month = 13
    results = []
    
    '''
    REPLACE API KEY HERE WITH YOUR API KEY.
    '''
    serpapi_key = ""
    
    for start_date, end_date in monthly_ranges:
        time_range_str = f"{start_date} - {end_date}"
        print(f"Collecting for time range: {time_range_str}")
        month_urls = []
        # Loop over sources until we get 13 unique URLs for this month.
        for source in sources:
            if len(month_urls) >= target_per_month:
                break
            needed = target_per_month - len(month_urls)
            new_urls = fetch_urls_with_date(query, source, start_date, end_date, num_results=needed, serpapi_key=serpapi_key)
            for url in new_urls:
                if url not in month_urls:
                    month_urls.append(url)
                    results.append({
                        "time_range": time_range_str,
                        "url": url,
                        "source": source
                    })
                    if len(month_urls) >= target_per_month:
                        break  # We have enough URLs for this month.
        print(f"Collected {len(month_urls)} URLs for time range {time_range_str}")
    
    # Save all the results to a CSV file.
    df = pd.DataFrame(results)
    df.to_csv("Scrapersandrawfiles/YFinanceData/aggregated_urls.csv", index=False)
    print(f"Saved aggregated URLs to aggregated_urls.csv. Total URLs: {len(df)}")

if __name__ == "__main__":
    main()
