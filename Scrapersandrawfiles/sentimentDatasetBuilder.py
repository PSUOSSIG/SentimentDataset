import requests
from bs4 import BeautifulSoup
import nltk
import pandas as pd
import random
import re
from transformers import pipeline

# Download the NLTK tokenizer data
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# Set up the sentiment analysis pipeline using FinBERT.
# Ensure you have the correct model available.
sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")

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

def scrape_article(url):
    """
    Scrapes an article's text and publication date from the given URL.
    Attempts to extract the publication date using common meta tags or <time> elements.
    If not found, falls back to parsing the date from the URL.
    Returns a tuple (text_content, pub_date). If not found, pub_date may be None.
    """
    try:
        response = requests.get(url, timeout=10)
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
            return None, None
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None, None

def build_sentiment_dataset(urls):
    """
    For each URL in 'urls':
      - Scrapes the article text and publication date.
      - Splits the text into sentences.
      - Randomly groups between 1 to 3 consecutive sentences to form one case.
      - Uses FinBERT to assign sentiment for each group.
      - Adds the raw FinBERT sentiment, its score, a numeric label, and the article date.
    Returns a list of dictionaries representing the dataset.
    """
    dataset = []
    total_cases = 0

    for url in urls:
        article_text, pub_date = scrape_article(url)
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

                if case_text == "":
                    continue

                try:
                    # Run sentiment analysis on the text block (truncated if needed).
                    result = sentiment_pipeline(case_text[:512])
                    finbert_label = result[0]['label']
                    finbert_score = result[0]['score']
                    numeric_label = assign_label(finbert_label)

                    dataset.append({
                        'url': url,
                        'date': pub_date,
                        'case_text': case_text,
                        'label': finbert_label,
                        'score': finbert_score,
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
        aggregated_urls_df = pd.read_csv("aggregated_urls.csv")
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
    df.to_csv("nvda_sentence_sentiment_dataset.csv", index=False)
    print("Dataset saved to nvda_sentence_sentiment_dataset.csv")

if __name__ == "__main__":
    main()
