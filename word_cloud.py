import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Read sentiment data
sentiment_file_path = '/content/drive/My Drive/Sentiment_Analysis_Results.csv'
data = pd.read_csv(sentiment_file_path)

# Generate a word cloud
text = ' '.join(data['Response'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Responses')
plt.savefig('/content/drive/My Drive/Word_Cloud.png')
plt.show()
