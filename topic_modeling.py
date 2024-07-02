import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Read sentiment data
sentiment_file_path = '/content/drive/My Drive/Sentiment_Analysis_Results.csv'
data = pd.read_csv(sentiment_file_path)

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
