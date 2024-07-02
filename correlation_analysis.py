import pandas as pd

# Read sentiment data
sentiment_file_path = '/content/drive/My Drive/Sentiment_Analysis_Results.csv'
data = pd.read_csv(sentiment_file_path)

# Correlation Analysis
keyword = 'doctor'
data[keyword] = data['Response'].str.contains(keyword).astype(int)
correlation = data['Sentiment'].corr(data[keyword])

correlation_file_path = '/content/drive/My Drive/Correlation_Analysis.txt'
with open(correlation_file_path, 'w') as f:
    f.write(f"Correlation between sentiment and the word '{keyword}': {correlation:.2f}\n")
