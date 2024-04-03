import pandas as pd
import numpy as np
import re
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("new_movie_data/final_cleaned_2.csv")
df['overview'].fillna('', inplace=True)
df['text'] = df['title'] + ' ' + df['release_date'].apply(str)  # ' ' + df['overview'] +

attributes = {
    'genres': 1,
    'production_companies': 0,
    'spoken_languages': 0,
    'keywords': 0,
    'crew': 0,
    'cast': 0,
    'text': 0
}

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

def get_similarity(id_1, id_2):
    score = 0
    
    movie_1 = df[df['id'] == id_1]
    movie_2 = df[df['id'] == id_2]

    # similarity based on text (overview, title, release_date)
    text_1 = movie_1['text'].values[0]
    text_2 = movie_2['text'].values[0]

    # Comment out when not using the text!!!!!!!!!
    tfidf = TfidfVectorizer(tokenizer = tokenize)
    movie_tfidf = tfidf.fit_transform([text_1, text_2])
    cos_sim = cosine_similarity(movie_tfidf[0:1], movie_tfidf[1:2])[0][0]
    score += attributes['text'] * cos_sim

    # similarity based on other attributes
    for attr, weight in attributes.items():
        if attr == 'text':
            continue
        attr_1 = set(eval(movie_1[attr].values[0]))
        attr_2 = set(eval(movie_2[attr].values[0]))
        attr_score = len(attr_1 & attr_2) / len(attr_1 | attr_2)
        score += weight * attr_score
    
    return score

def compute_dissimilarity(selected_ids, movie_id):
    """Intra-list diversity using MMR"""
    dissimilarity = np.mean([1 - get_similarity(s_id, movie_id) for s_id in selected_ids])
    return dissimilarity

def diversity_using_mmr(recommendations, top_n, lambda_param=0.5):
    recommendations_list = recommendations.to_dict('records')
    selected = {}
    remaining = recommendations_list.copy()

    while len(selected) < top_n and remaining:
        mmr_scores = []

        for movie in remaining:
            movie_id = movie['movie_id']

            if not selected:
                mmr_score = lambda_param * movie['score']
                mmr_scores.append(mmr_score)
                break
            else:
                # Compute dissimilarity to already selected songs
                dissimilarity = compute_dissimilarity(selected.keys(), movie_id)
                mmr_score = lambda_param * movie['score'] + (1 - lambda_param) * dissimilarity
                mmr_scores.append(mmr_score)

        # Select song with highest MMR score
        max_mmr_idx = np.argmax(mmr_scores)
        selected_movie = remaining.pop(max_mmr_idx)
        selected_movie_id = selected_movie['movie_id']
        selected_movie_score = selected_movie['score']
        selected[selected_movie_id] = selected_movie_score

    return pd.DataFrame(selected.items())
