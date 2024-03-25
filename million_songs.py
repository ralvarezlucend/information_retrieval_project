from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
import pandas as pd
import numpy as np
from surprise import SVD, dump

from utils import diversity_using_mmr, split, train_and_save_model, get_top_n


def calculate_feature_similarity(feature_vec_1, feature_vec_2):
    cos_sim = cosine_similarity([feature_vec_1], [feature_vec_2])[0][0]
    return cos_sim


# song_hotness not present for all songs
def get_feature_vector(song_data):
    features = np.array([song_data['tempo'], song_data['loudness'], song_data['artist_familiarity']])
    return features


# use this in mmr in utils.py
def compute_similarity(df, song_id_1, song_id_2):
    song_data_1 = df.loc[df['song_id'] == song_id_1].iloc[0]
    song_data_2 = df.loc[df['song_id'] == song_id_2].iloc[0]

    feature_vec_1 = get_feature_vector(song_data_1)
    feature_vec_2 = get_feature_vector(song_data_2)

    similarity = calculate_feature_similarity(feature_vec_1, feature_vec_2)
    return similarity


# TODO: how do we train the model when no user_id and no play_count in this data


data_file = 'data/10000_songs.csv'
df = pd.read_csv(data_file, sep='\t', header=None, names=['user_id', 'song_id', 'play_count'])

# dump_file = 'models/svd_million_songs.pkl'
# algorithm = SVD()
# trainset, testset = split(df)
# train_and_save_model(trainset, algorithm, dump_file)
#
# _, algorithm = dump.load(dump_file)
#
# rec_df = get_top_n(rec_user_ids, song_raw_ids, algorithm, n=100)
#
# diversified_recommendation = diversity_using_mmr(df, top_n=5)
