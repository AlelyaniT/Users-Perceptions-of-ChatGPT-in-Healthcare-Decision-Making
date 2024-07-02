# Exploring Factors Influencing User Perceptions of ChatGPT in Healthcare Decision-Making

This repository contains a comprehensive analysis of user responses related to ChatGPT's role in healthcare decision-making. The analysis includes sentiment distribution, detailed sentiment breakdown, word cloud generation, topic modeling, correlation analysis, and frequent bigrams identification. The results are visualized and saved to Google Drive.

## Features

- **Sentiment Analysis**: Classifies the sentiment of user responses into positive, neutral, and negative categories.
- **Detailed Sentiment Breakdown**: Provides a detailed histogram of sentiment scores.
- **Word Cloud**: Generates a word cloud to visualize the most frequent words in the responses.
- **Topic Modeling**: Uses Latent Dirichlet Allocation (LDA) to identify key topics in the responses.
- **Correlation Analysis**: Analyzes the correlation between sentiment scores and the presence of specific keywords.
- **Frequent Bigrams**: Identifies and visualizes the most frequent bigrams (pairs of words) in the


## Instructions for Using the Scripts

1. Mount Google Drive:
Run mount_drive.py to mount your Google Drive.
```
   from google.colab import drive
   drive.mount('/content/drive')
```


3. Run Each Analysis Script:
 Run each script in sequence to perform the analysis and save the results to your Google Drive.

```
python scripts/sentiment_analysis.py
python scripts/detailed_sentiment_breakdown.py
python scripts/word_cloud.py
python scripts/topic_modeling.py
python scripts/correlation_analysis.py
python scripts/frequent_bigrams.py
```




