# **Initial Dataset for the testing and training for Sentiment Analysis**
---

Make sure you load the requirements.

**NLTK** Will open up a window, go ahead and select "download" at the bottom and download all the necessary libraries, as we will be working with this a lot.

## **Main files:**
 - *sentiment_data.csv*: we will use this for training and testing our model.
 - *forbes_eda.ipynb*: contains basic information about the dataset, and the cleaning operations performed
 - *Scrutinize/nvda_sentence_sentiment_dataset_forbes_cleaned.csv*: This dataset will have to be manually pruned/feature engineered/augmented for better functionality
 - *Scraperandrawfiles/URLPuller.py*: utilizes SerpAPI to pull URLs for NVDA stock data. Currently only had enough requests for Forbes. Utilizes empty aggregated_urls.csv to work.
 - *Scraperandrawfiles/sentimentDatasetBuilder.py*: Constructs the nvda_sentence_sentiment_dataset.csv base. Randomly chooses between 1-3 sentences to construct cases for better generalizeability. Utilizes Fin-BERT to perform sentiment analysis and pulls dates from the URLs.
 - *Scraperandrawfiles/aggregated_urls.csv*: empty csv for URLPuller.py Ensure this file is not missing. Once URL puller is done, rename the csv and construct a new one. (I will come back and fix this for better operability)
