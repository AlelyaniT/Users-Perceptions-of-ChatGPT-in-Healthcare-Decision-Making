import pandas as pd
from textblob import TextBlob

# Set the path to the text file in your Google Drive
data_file_path = '/content/drive/My Drive/Responses_cleaned.txt'

# Read the data from a text file
data = pd.read_csv(data_file_path, delimiter="\t", header=None, names=["Response"])

# Perform sentiment analysis
data['Sentiment'] = data['Response'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

# Categorize sentiment
data['Sentiment Category'] = pd.cut(data['Sentiment'], bins=[-1, -0.1, 0.1, 1], labels=['Negative', 'Neutral', 'Positive'])

# Save sentiment data to a file
sentiment_file_path = '/content/drive/My Drive/Sentiment_Analysis_Results.csv'
data.to_csv(sentiment_file_path, index=False)
