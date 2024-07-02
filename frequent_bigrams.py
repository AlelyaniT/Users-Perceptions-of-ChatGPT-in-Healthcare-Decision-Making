import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

# Read sentiment data
sentiment_file_path = '/content/drive/My Drive/Sentiment_Analysis_Results.csv'
data = pd.read_csv(sentiment_file_path)

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
