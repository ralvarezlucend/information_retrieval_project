# This is importing regular expression
import re

# Word_tokenize is used to do tokenization
from nltk import word_tokenize

# Importing the Lematizer 
from nltk.stem import WordNetLemmatizer

# Importing the stopwords
from nltk.corpus import stopwords

# Tfidf vectorizer used to create the computational vectors
from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

# Importing nltk (natural language toolkit library)
import nltk
# nltk.download('omw-1.4')

# # Downloading punctuations
# nltk.download('punkt')

# # Downloading stopwords
# nltk.download('stopwords')

# # Downloading wordnet
# nltk.download('wordnet') 

# Create a function to tokenize the text
def tokenize(text):
    
    # Making each letter as lowercase and removing non-alphabetical text
    text = re.sub(r"[^a-zA-Z]"," ", text.lower())
    
    # Extracting each word in the text
    tokens = word_tokenize(text)
    
    # Removing stopwords
    words = [word for word in tokens if word not in stopwords.words("english")]
    
    # Lemmatize the words
    text_lems = [WordNetLemmatizer().lemmatize(lem).strip() for lem in words]

    return text_lems

df = pd.read_csv('data/kaggle_songs.csv')

df = df[:1000]

# Convert release_date column from int to string
df['release_date'] = df['release_date'].astype(str)

df_small = pd.DataFrame({
    'track_name': df['track_name'],
    'artist_name': df['artist_name'],
    'release_date': df['release_date'],
    'text': df['track_name'] + ' ' + df['artist_name'] + ' ' + df['genre'] + ' ' + df['release_date']
})

# Drop the duplicates from the title column
df_small = df_small.drop_duplicates(subset='track_name')

# Set the title column as the index
df_small = df_small.set_index('track_name')

# Create tfidf vectorizer 
tfidf = TfidfVectorizer(tokenizer = tokenize)

song_tfidf = tfidf.fit_transform(df_small['text'].values).toarray()

# Calculating the cosine similarity
similar_songs = cosine_similarity(song_tfidf, song_tfidf)

# Function that takes in song title as input and returns the top 10 recommended songs
def recommendations(title, similar_songs, n):
    recommended_songs = []
    indices = pd.Series(df_small.index)
    idx = indices[indices == title].index[0]

    score_series = pd.Series(similar_songs[idx]).sort_values(ascending = False)

    top_10_indexes = list(score_series.iloc[1 : n+1].index)
    top_10_rankings = list(score_series.iloc[1 : n+1].values)
    
    for i in top_10_indexes:
        recommended_songs.append(list(df_small.index)[i])

    return list(zip(recommended_songs, top_10_rankings))

print(recommendations('necessary evil', similar_songs, n=10))









# # only keep numeric columns 
# df = df.select_dtypes(include=np.number)

# X = df[:5000]
# Y = df[5000:10000]
# sim = cosine_similarity(X)

# # compute mean per row
# mean = np.mean(sim, axis=0)