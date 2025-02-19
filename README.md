# **Initial Dataset for the testing and training for Sentiment Analysis**
---

Make sure you load the requirements.

**NLTK** Will open up a window, go ahead and select "download" at the bottom and download all the necessary libraries, as we will be working with this a lot.

## **Main files:**
 - *_sentiment_data.csv_*: we will use this for training and testing our model.
 - *_forbes_eda.ipynb_*: contains basic information about the dataset, and the cleaning operations performed
 - *_Scrutinize/nvda_sentence_sentiment_dataset_forbes_cleaned.csv_*: This dataset will have to be manually pruned/feature engineered/augmented for better functionality
 - *_Scraperandrawfiles/nvda_sentence_sentiment_dataset_forbes_uncleaned.csv_*: Constructed dataset based on scraped data before cleaning.
 - *_Scraperandrawfiles/URLPuller.py_*: utilizes SerpAPI to pull URLs for NVDA stock data. Currently only had enough requests for Forbes. Utilizes empty aggregated_urls.csv to work.
 - *_Scraperandrawfiles/sentimentDatasetBuilder.py_*: Constructs the nvda_sentence_sentiment_dataset.csv base. Randomly chooses between 1-3 sentences to construct cases for better generalizeability. Utilizes Fin-BERT to perform sentiment analysis and pulls dates from the URLs.
 - *_Scraperandrawfiles/aggregated_urls.csv_*: empty csv for URLPuller.py Ensure this file is not missing. Once URL puller is done, rename the csv and construct a new one. (I will come back and fix this for better operability)

