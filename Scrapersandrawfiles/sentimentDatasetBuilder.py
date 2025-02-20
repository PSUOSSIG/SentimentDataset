import requests
from bs4 import BeautifulSoup
import nltk
import pandas as pd
import random
import re
import time
from transformers import pipeline

# Download the NLTK tokenizer data
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# Set up the sentiment analysis pipeline using FinBERT.
sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")

# List of user agents to rotate.
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.96 Safari/537.36",
    # Add more user agents as needed.
]

# Optional: Define a list of proxies to rotate.
PROXIES = [
    # Example format: "http://username:password@proxyaddress:port"
    # "http://proxy1.example.com:8080",
    # "http://proxy2.example.com:8080",
]

def assign_label(sentiment_str):
    """
    Convert FinBERT sentiment label to a numeric label:
      1 for Positive,
      0 for Neutral,
      -1 for Negative.
    """
    sentiment_str = sentiment_str.lower()
    if sentiment_str == "positive":
        return 1
    elif sentiment_str == "negative":
        return -1
    else:
        return 0

def parse_date_from_url(url):
    """
    Attempts to parse a publication date from the URL.
    Expects a URL with a structure like: .../YYYY/MM/DD/...
    Returns a string in the format 'YYYY-MM-DD' if successful, otherwise None.
    """
    match = re.search(r'/(\d{4})/(\d{2})/(\d{2})/', url)
    if match:
        year, month, day = match.groups()
        return f"{year}-{month}-{day}"
    return None

def get_random_headers():
    """Return a headers dict with a random user agent."""
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/"
    }

def get_random_proxy():
    """Return a random proxy from the list if available, else None."""
    if PROXIES:
        proxy = random.choice(PROXIES)
        return {"http": proxy, "https": proxy}
    return None

def scrape_article(url, session, max_retries=3):
    """
    Scrapes an article's text and publication date from the given URL.
    Incorporates random delays, user agent rotation, and retry logic.
    Returns a tuple (text_content, pub_date). If not found, pub_date may be None.
    """
    delay = random.uniform(1, 5)
    time.sleep(delay)  # Random delay to mimic human behavior

    retries = 0
    backoff = 1  # initial backoff in seconds
    while retries < max_retries:
        try:
            headers = get_random_headers()
            proxies = get_random_proxy()
            response = session.get(url, headers=headers, proxies=proxies, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                # Extract text from all <p> tags.
                paragraphs = soup.find_all('p')
                text_content = " ".join([p.get_text() for p in paragraphs])

                # Try to extract the publication date from meta tags or <time> element.
                pub_date = None
                meta_date = soup.find("meta", property="article:published_time")
                if meta_date and meta_date.get("content"):
                    pub_date = meta_date.get("content")
                else:
                    meta_date = soup.find("meta", attrs={"name": "date"})
                    if meta_date and meta_date.get("content"):
                        pub_date = meta_date.get("content")
                if not pub_date:
                    time_tag = soup.find("time")
                    if time_tag:
                        pub_date = time_tag.get("datetime") or time_tag.get_text().strip()

                # Fallback: If still no date, try parsing from the URL.
                if not pub_date:
                    pub_date = parse_date_from_url(url)

                return text_content, pub_date
            else:
                print(f"Error: Received status code {response.status_code} for URL: {url}")
        except Exception as e:
            print(f"Error scraping {url}: {e}")
        # Exponential backoff before retrying
        retries += 1
        time.sleep(backoff)
        backoff *= 2

    return None, None

def build_sentiment_dataset(urls):
    """
    For each URL in 'urls':
      - Scrapes the article text and publication date.
      - Splits the text into sentences.
      - Randomly groups between 1 to 3 consecutive sentences to form one case.
      - Uses FinBERT to assign sentiment for each group, returning scores for all classes.
      - Determines overall sentiment (highest score) and assigns a numeric label.
      - Adds the probability scores for all sentiment types along with the article date.
    Returns a list of dictionaries representing the dataset.
    """
    dataset = []
    total_cases = 0
    session = requests.Session()  # Use a persistent session for connection pooling

    for url in urls:
        article_text, pub_date = scrape_article(url, session)
        case_count = 0  # Count cases per URL

        if article_text:
            sentences = sent_tokenize(article_text)
            i = 0
            while i < len(sentences):
                # Randomly choose group size (1 to 3 sentences)
                group_size = random.choice([1, 2, 3])
                # Ensure we don't go out of bounds.
                sentence_group = sentences[i:i+group_size]
                # Combine the sentences into a single text block.
                case_text = " ".join(sentence_group).strip()
                i += group_size  # Move index by the number of sentences used.

                if not case_text:
                    continue

                try:
                    # Run sentiment analysis with return_all_scores=True.
                    result = sentiment_pipeline(case_text[:512], return_all_scores=True)
                    # result[0] is a list of dictionaries for each sentiment class.
                    scores_dict = {r['label'].lower(): r['score'] for r in result[0]}
                    # Determine the overall sentiment by selecting the label with the highest score.
                    overall = max(result[0], key=lambda r: r['score'])
                    overall_label = overall['label']
                    overall_score = overall['score']
                    numeric_label = assign_label(overall_label)

                    dataset.append({
                        'url': url,
                        'date': pub_date,
                        'case_text': case_text,
                        'label': overall_label,
                        'score': overall_score,
                        'positive_prob': scores_dict.get('positive'),
                        'neutral_prob': scores_dict.get('neutral'),
                        'negative_prob': scores_dict.get('negative'),
                        'numeric_label': numeric_label
                    })
                    total_cases += 1
                    case_count += 1
                except Exception as e:
                    print(f"Error processing case: {case_text}\nError: {e}")
                    continue

            print(f"Scraped URL: {url} - {case_count} cases extracted.")
        else:
            print(f"Scraped URL: {url} - No article text found.")
            
    return dataset

def main():
    # Load URLs from aggregated_urls.csv.
    try:
        aggregated_urls_df = pd.read_csv("Scrapersandrawfiles/YFinanceData/aggregated_urls.csv")
        # Assuming the CSV file has a column named "url" that contains the URLs.
        urls = aggregated_urls_df["url"].dropna().tolist()
    except Exception as e:
        print("Error reading aggregated_urls.csv:", e)
        return

    # Build the sentiment dataset.
    dataset = build_sentiment_dataset(urls)
    
    # Convert to a DataFrame.
    df = pd.DataFrame(dataset)
    print(f"Total cases processed: {len(df)}")
    
    # Save the dataset to a CSV file.
    df.to_csv("Scrapersandrawfiles/YFinanceData/nvda_sentence_sentiment_dataset.csv", index=False)
    print("Dataset saved to nvda_sentence_sentiment_dataset_fool_uncleaned.csv")

if __name__ == "__main__":
    main()
