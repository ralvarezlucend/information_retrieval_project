from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Reader, Dataset
from surprise.model_selection import train_test_split
from surprise import accuracy
from surprise import dump
import pandas as pd
from tqdm import tqdm
import numpy as np


def split(df, test_size: float = 0.25):
    """
    Splits a Pandas DataFrame into a Surprise trainset and a testset.
    Normalizes the ratings to be between 0 and 1.
    """

    # Rename columns to match surprise's expected input
    df.rename(columns={'user_id': 'userID', 'song_id': 'itemID', 'play_count': 'rating'}, inplace=True)

    # Normalize ratings to be between 0 and 1
    max_rating = df['rating'].max()
    df['rating'] = df['rating'] / max_rating

    # Load the data into a surprise dataset
    reader = Reader(rating_scale=(0, 1))
    data = Dataset.load_from_df(df[["userID", "itemID", "rating"]], reader)

    # Split the data into a trainset and a testset
    trainset, testset = train_test_split(data, test_size=test_size)
    return trainset, testset


def train_and_save_model(trainset, algorithm, dump_file):
    """
    Trains a model and saves it to a file as a pickle object.
    """
    algorithm.fit(trainset)
    dump.dump(dump_file, algo=algorithm, verbose=1)


def compute_rmse(algorithm, testset):
    """
    Computes rmse on the testset.
    """
    predictions = algorithm.test(testset)
    rmse = accuracy.rmse(predictions, verbose=True)
    return rmse


def get_top_n(user_ids, song_ids, algo, n):
    """
    user_id - list of raw ids of the users (uid)
    song_ids - list of raw ids of the songs (iid)
    n - number of recommendations to return
    """
    # rec = {"uid": [], "song_id": [], "rank": []}
    recommendations = []
    for uid in tqdm(user_ids):
        user_rec = [algo.predict(uid, iid) for iid in song_ids]
        user_rec.sort(key=lambda x: x.est, reverse=True)
        top_n = user_rec[:n]

        for rank, rec in enumerate(top_n, start=1):
            recommendations.append({
                'user_id': rec.uid,
                'song_id': rec.iid,
                'rank': rec.est
            })

    # print(top_n)
    # break
    # rec["uid"].extend([r.uid for r in top_n])
    # rec["song_id"].extend([r.iid for r in top_n])
    # rec["rank"].extend([r.est for r in top_n])

    rec_df = pd.DataFrame(recommendations)
    return rec_df


def calculate_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    # tfidf_matrix = vectorizer.fit_transform(text_data)
    # cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    texts = [text1, text2]

    tfidf_matrix = vectorizer.fit_transform(texts)
    # print(tfidf_matrix)

    cos_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    # print(cos_sim)

    return cos_sim[0][0]

def compute_dissimilarity(selected, song_text):
    """Intra-list diversity using MMR"""
    dissimilarity = np.mean([1 - calculate_similarity(song_text, s['text']) for s in selected])
    return dissimilarity


def diversity_using_mmr(recommendations, top_n, lambda_param=0.7):
    recommendations['song_id'] = recommendations['song_id'].astype(str)
    recommendations_list = recommendations.to_dict('records')

    selected = []
    remaining = recommendations_list.copy()

    while len(selected) < top_n and remaining:
        mmr_scores = []

        for song in remaining:
            song_text = song['text']

            if not selected:
                mmr_score = lambda_param * song['rank']
            else:
                # Compute dissimilarity to already selected songs
                dissimilarity = compute_dissimilarity(selected, song_text)
                mmr_score = lambda_param * song['rank'] - (1 - lambda_param) * dissimilarity
                # print(str(dissimilarity) + " " + str(mmr_score))
           
            mmr_scores.append(mmr_score)

        # Select song with highest MMR score
        max_mmr_idx = np.argmax(mmr_scores)
        selected_song = remaining.pop(max_mmr_idx)
        selected.append(selected_song)

    return pd.DataFrame(selected)
