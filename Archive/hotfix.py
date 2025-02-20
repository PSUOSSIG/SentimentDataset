import pandas as pd
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize

# Download NLTK tokenizer data if not already downloaded.
nltk.download('punkt')

# Set up the FinBERT sentiment analysis pipeline.
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

def process_text(text):
    """
    Given a text, run sentiment analysis with FinBERT, returning a dictionary
    with overall classification (label, score, numeric_label) and the individual
    positive, neutral, and negative probabilities.
    """
    # Run sentiment analysis with return_all_scores=True.
    result = sentiment_pipeline(text[:512], return_all_scores=True)
    # result[0] is a list of dicts for each sentiment class.
    scores = result[0]
    # Create a dictionary mapping each sentiment to its score.
    scores_dict = {entry['label'].lower(): entry['score'] for entry in scores}
    # Determine the overall sentiment based on the highest score.
    overall = max(scores, key=lambda entry: entry['score'])
    overall_label = overall['label']
    overall_score = overall['score']
    numeric_label = assign_label(overall_label)

    return {
        "label": overall_label,
        "score": overall_score,
        "numeric_label": numeric_label,
        "positive_prob": scores_dict.get("positive"),
        "neutral_prob": scores_dict.get("neutral"),
        "negative_prob": scores_dict.get("negative")
    }

def main():
    # Read the CSV file that contains the dataset.
    input_filename = "Scrutinize/nvda_sentence_sentiment_dataset_forbes_cleaned.csv"
    try:
        df = pd.read_csv(input_filename)
    except Exception as e:
        print(f"Error reading {input_filename}: {e}")
        return

    # Ensure there is a "case_text" column in the dataframe.
    if "case_text" not in df.columns:
        print("Error: The CSV file does not contain a 'case_text' column.")
        return

    # Process each row and update the classification columns.
    classification_results = []
    for idx, row in df.iterrows():
        text = row["case_text"]
        if isinstance(text, str) and text.strip():
            try:
                result = process_text(text)
            except Exception as e:
                print(f"Error processing text at row {idx}: {e}")
                result = {
                    "label": None,
                    "score": None,
                    "numeric_label": None,
                    "positive_prob": None,
                    "neutral_prob": None,
                    "negative_prob": None
                }
        else:
            result = {
                "label": None,
                "score": None,
                "numeric_label": None,
                "positive_prob": None,
                "neutral_prob": None,
                "negative_prob": None
            }
        classification_results.append(result)

    # Create a DataFrame from the classification results.
    classification_df = pd.DataFrame(classification_results)
    
    # Replace or add the classification columns in the original DataFrame.
    for col in ["label", "score", "numeric_label", "positive_prob", "neutral_prob", "negative_prob"]:
        df[col] = classification_df[col]

    # Save the updated dataset to a new CSV file.
    output_filename = "nvda_sentence_sentiment_dataset_updated.csv"
    df.to_csv(output_filename, index=False)
    print(f"Updated dataset saved to {output_filename}")

if __name__ == "__main__":
    main()
