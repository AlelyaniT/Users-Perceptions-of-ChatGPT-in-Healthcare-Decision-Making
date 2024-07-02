import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read sentiment data
sentiment_file_path = '/content/drive/My Drive/Sentiment_Analysis_Results.csv'
data = pd.read_csv(sentiment_file_path)

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
