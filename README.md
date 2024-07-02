# Users-Perceptions-of-ChatGPT-in-Healthcare-Decision-Making


This repository contains a comprehensive analysis of user responses related to ChatGPT's role in healthcare decision-making. The analysis includes sentiment distribution, detailed sentiment breakdown, word cloud generation, topic modeling, correlation analysis, and frequent bigrams identification. The results are visualized and saved to Google Drive.

## Features

- **Sentiment Analysis**: Classifies the sentiment of user responses into positive, neutral, and negative categories.
- **Detailed Sentiment Breakdown**: Provides a detailed histogram of sentiment scores.
- **Word Cloud**: Generates a word cloud to visualize the most frequent words in the responses.
- **Topic Modeling**: Uses Latent Dirichlet Allocation (LDA) to identify key topics in the responses.
- **Correlation Analysis**: Analyzes the correlation between sentiment scores and the presence of specific keywords.
- **Frequent Bigrams**: Identifies and visualizes the most frequent bigrams (pairs of words) in the responses.

## Requirements

- Google Colab
- Google Drive (for storing input and output files)
- Python packages:
  - pandas
  - matplotlib
  - seaborn
  - textblob
  - wordcloud
  - scikit-learn

## Usage

### Step 1: Upload Your Data to Google Drive

Ensure your cleaned responses file (`Responses_cleaned.txt`) is uploaded to your Google Drive.

### Step 2: Mount Google Drive in Colab

Mount your Google Drive in your Google Colab notebook:

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 3: Run the Analysis Script

Copy and paste the following code into your Colab notebook:

```python
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

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

# Plot the sentiment distribution
sentiment_counts = data['Sentiment Category'].value_counts().sort_index()
plt.figure(figsize=(10, 6))
plt.bar(sentiment_counts.index, sentiment_counts.values, color=['red', 'blue', 'green'])
plt.title('Sentiment Analysis of ChatGPT Responses')
plt.xlabel('Sentiment')
plt.ylabel('Number of Responses')
plt.savefig('/content/drive/My Drive/Sentiment_Distribution.png')
plt.show()

# Detailed Sentiment Breakdown
plt.figure(figsize=(10, 6))
sns.histplot(data['Sentiment'], bins=30, kde=True, color='blue')
plt.title('Detailed Sentiment Scores')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.savefig('/content/drive/My Drive/Detailed_Sentiment_Scores.png')
plt.show()

# Generate a word cloud
text = ' '.join(data['Response'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Responses')
plt.savefig('/content/drive/My Drive/Word_Cloud.png')
plt.show()

# Topic Modeling
count_vectorizer = CountVectorizer(stop_words='english')
count_data = count_vectorizer.fit_transform(data['Response'])

lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(count_data)

def display_topics(model, feature_names, no_top_words):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        topic_words = " ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])
        topics.append(f"Topic {topic_idx+1}: {topic_words}")
    return topics

no_top_words = 10
topics = display_topics(lda, count_vectorizer.get_feature_names_out(), no_top_words)

# Save topics to a file
topics_file_path = '/content/drive/My Drive/Topic_Modeling_Results.txt'
with open(topics_file_path, 'w') as f:
    for topic in topics:
        f.write(f"{topic}\n")

# Correlation Analysis
keyword = 'doctor'
data[keyword] = data['Response'].str.contains(keyword).astype(int)
correlation = data['Sentiment'].corr(data[keyword])

correlation_file_path = '/content/drive/My Drive/Correlation_Analysis.txt'
with open(correlation_file_path, 'w') as f:
    f.write(f"Correlation between sentiment and the word '{keyword}': {correlation:.2f}\n")

# Frequent Bigrams
bigram_vectorizer = CountVectorizer(ngram_range=(2, 2), stop_words='english')
bigram_data = bigram_vectorizer.fit_transform(data['Response'])
sum_bigrams = bigram_data.sum(axis=0)

bigrams_freq = [(word, sum_bigrams[0, idx]) for word, idx in bigram_vectorizer.vocabulary_.items()]
bigrams_freq = sorted(bigrams_freq, key=lambda x: x[1], reverse=True)[:10]

bigrams, freqs = zip(*bigrams_freq)
plt.figure(figsize=(10, 6))
plt.barh(bigrams, freqs, color='purple')
plt.xlabel('Frequency')
plt.title('Top 10 Bigrams in Responses')
plt.gca().invert_yaxis()
plt.savefig('/content/drive/My Drive/Top_Bigrams.png')
plt.show()
```

### Results

- **Sentiment Analysis Results**: Saved as `Sentiment_Analysis_Results.csv`
- **Sentiment Distribution Plot**: Saved as `Sentiment_Distribution.png`
- **Detailed Sentiment Scores Plot**: Saved as `Detailed_Sentiment_Scores.png`
- **Word Cloud**: Saved as `Word_Cloud.png`
- **Topic Modeling Results**: Saved as `Topic_Modeling_Results.txt`
- **Correlation Analysis**: Saved as `Correlation_Analysis.txt`
- **Top 10 Bigrams Plot**: Saved as `Top_Bigrams.png`

### Contributing

Feel free to open issues or submit pull requests if you find bugs or have suggestions for improvements.

### License

This project is licensed under the MIT License.

