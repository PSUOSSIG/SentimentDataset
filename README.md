# **Semantic Analysis Dataset**
---
## **TL:DR**
#### Your dataset to train on is **sentiment_data.csv** feel free to download just this raw file.
#### Format for the CSV file
 - **url**: URL the scraped text belongs to
 - **date**: YYYY-MM-DD format. Date of the published article the text is from.
 - **case_text**: Text, ranging from 1-3 sentences for granularity. **NOTE: Group by URL to get the full document text.**
 - **label**: The categorical label for the sentiment. {positive,negative,neutral}
 - **score**: The probability (assurance) of the correctly assigned label. (Winning label probability) 
 - **numeric_label**: Numeric version of label. 1 = positive, 0 = neutral, -1 = negative.
 - **positive_prob**: FinBERT's positive probability assignment. sum of all probabilities = 1.
 - **neutral_prob**: FinBERT's neutral probability assignment. sum of all probabilities = 1.
 - **negative_prob**: FinBERT's negative probability assignment. sum of all probabilities = 1.

---
## Dataset Summary
This dataset was constructed by aggregating data from multiple finance websites with a focus on NVDA (NVIDIA Corporation) stock. The data collection process involved the following steps:

#### Web Scraping: 
 - URL Collection: We used SerpAPI to scrape URLs from various finance websites.
 - Content Extraction: For each URL, we employed BeautifulSoup4 to extract individual sentences, ensuring we capture detailed and context-rich information.
#### Sentiment Analysis:
 - After collecting the textual data, we applied sentiment classification using FinBERT, a domain-specific adaptation of BERT optimized for financial text. This allowed us to tag each sentence with sentiment scores, providing insights into the market mood surrounding NVDA stock. **NOTE THAT THIS IS A TEMPORARY IMPLEMENTATION, AND WE ARE CURRENTLY WORKING ON OUR OWN SENTIMENT CLASSIFIER TO ACHIEVE BETTER SENTIMENT MARGINS**.
---

## Requirements

- Make sure you load the requirements from requirements.txt.
- **NLTK** Will open up a window, go ahead and select "download" at the bottom and download all the necessary libraries, as we will be working with this a lot.
- **Note For SerpAPI:** you *must* have your own key in order for this to work. Visit [SerpAPI](https://serpapi.com/) for more info.
---
## **Directory Explanation: **

#### **Key**:
 - ***<u>Directories will be bolded and italicized</u>***
 - **Scripts will be bolded (.py files)**
 - **Data Cleaners and combiners will be bolded (.ipynb files).**
 - *csv's will appear italicized*
 - **^^**: Put at the end of descriptions to denote the csv will have the same name as others within the parent directory for modularizeability.

#### The following is the format for the directory:
 - ***<u>Archive</u>***: Directory that contains legacy files that have been updated since. Reserved as a backup.
   - **hotfix.py**: Script developed to add positive, neutral, AND negative probabilities to our first dataset. Now implemented in our main script.
 - ***<u>FullSet</u>***: Directory that contains the final datasets for every individual news source, as well as a combination dataset of all news sources.
   - ***<u>FullSetUncombined</u>***: Directory that contains only the final datasets for every individual news source
     - *nvda_sentence_sentiment_dataset_fool_cleaned.csv*: Motley fool cleaned full dataset
     - *nvda_sentence_sentiment_dataset_forbes_cleaned.csv*: Forbes cleaned full dataset
     - *nvda_sentence_sentiment_dataset_yfinance_cleaned.csv*: YFinance fool cleaned full 
   - *Full_Dataset.csv*: The final **full dataset**. Is not undersampled for training. This is what we will be performing more manipulation on.
   - **<u>FullSet_combiner.ipynb</u>**: combines all files inside of **<u>FullSetUncombined</u>** and constructs *Full_Dataset.csv*. Contains some sanity checks.
 - ***<u>Scrapersandrawfiles</u>***: Contains all of our webscraper modules, as well as the initial pulled data from them
   - **<u>ForbesData</u>**: Contains raw scraped csv's from Forbes.
     - *aggregated_urls.csv*: List of csv's provided by SerpAPI (**URLPuller.py**). This is fed into our BeatifulSoup4 dataset builder (**sentimentDatasetBuilder.py**).**^^**
     - **<u>forbs_eda.ipynb</u>**: Cleans the file and saves it under ../../FullSet/FullSetUncombined/nvda_sentence_sentiment_dataset_forbes_cleaned.csv for the full csv and ../../UnderSampledUncombined/sentiment_data_yfinance.csv for the undersampled partition, respectively.
     - *nvda_sentence_sentiment_dataset.csv*: Output from the dataset builder (**sentimentDatasetBuilder.py**). Is fed into **<u>forbs_eda.ipynb</u>** for cleanup.**^^**
   - **<u>MotleyFoolData</u>**: Contains raw scraped csvs from Motley Fool.
     - *aggregated_urls.csv*: List of csv's provided by SerpAPI (**URLPuller.py**). This is fed into our BeatifulSoup4 dataset builder (**setimentDatasetBuilder.py**).**^^**
     - **<u>Motley_eda.ipynb</u>**: Cleans the file and saves it under ../../FullSet/FullSetUncombined/nvda_sentence_sentiment_dataset_fool_cleaned.csv for the full csv and ../../UnderSampledUncombined/sentiment_data_fool.csv for the undersampled partition, respectively.
     - *nvda_sentence_sentiment_dataset.csv*: Output from the dataset builder (**sentimentDatasetBuilder.py**). Is fed into **<u>Motley_eda.ipynb</u>** for cleanup.**^^**
   - **<u>YFinanceData</u>**: Contains raw scraped csvs from YFinance.
     - *aggregated_urls.csv*: List of csv's provided by SerpAPI (**URLPuller.py**). This is fed into our BeatifulSoup4 dataset builder (**setimentDatasetBuilder.py**).**^^**
     - *nvda_sentence_sentiment_dataset.csv*: Output from the dataset builder (**setimentDatasetBuilder.py**). Is fed into **<u>yfinanceEDA.ipynb</u>** for cleanup.**^^**
     - **<u>yFinanceEDA.ipynb</u>**: Cleans the file and saves it under ../../FullSet/FullSetUncombined/nvda_sentence_sentiment_dataset_fool_cleaned.csv for the full csv and ../../UnderSampledUncombined/sentiment_data_fool.csv for the undersampled partition, respectively.
   - **sentimentDatasetBuilder.py**: Utilizes BeautifulSoup and FinBERT to create a text dataset with sentiments. Identfiers for URL and Date are attached for sentiment and potential time series analysis. Constructs a 1-3 sentence granularity for each text case. Will save file to specified location as *nvda_sentence_sentiment_dataset.csv*. See comments inside for more information on parameters.
   - **URLPuller.py**: Utilizes SerpAPI to pull URLs from specified websites. **Only put in 1 news site at a time into the feed. Input your API Key into this file as instructed by the comments.** Will save file to specified location as *aggregated_urls.csv*. See comments inside for more information on parameters.
 - ***<u>UnderSampledUncombined</u>***: Contains final cleaned data from every individual specified news outlet. Also contains a combiner file.
   - *sentiment_data_fool.csv*: truncated version of *nvda_sentence_sentiment_dataset_fool_cleaned.csv*. Undersampled to balance dataset.
   - *sentiment_data_forbes.csv*: truncated version of *nvda_sentence_sentiment_dataset_forbes_cleaned.csv*. Undersampled to balance dataset.
   - *sentiment_data_yfinance.csv*: truncated version of *nvda_sentence_sentiment_dataset_yfinance_cleaned.csv*. Undersampled to balance dataset.
   - **<u>undersampled_combined.ipynb</u>**: Aggregates all 3 of these files for construction of the final dataset for training sentiment, *sentiment_data.csv*.
 - README.md: This file.
 - requirements.txt: Contains all of the necessary imports to ensure the functionality of the module.
 - *sentiment_data.csv*: the final training file for sentiment analysis.

# Closing Note:
This is a **WORK IN PROGRESS**, and will constantly be updated.
