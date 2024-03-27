import pandas as pd
from surprise import SVD
from surprise import dump
from utils import split, train_and_save_model, compute_rmse, get_top_n, diversity_using_mmr  # , intra_list_diversity

# Dataset taken from: "http://millionsongdataset.com/tasteprofile/"
# user_id and song_id fields are anonymized.
data_file = 'data/10000.txt'
# Use SVD as it scales well, KNN causes python process to crash (out of memory)
dump_file = 'models/svd.pkl'

# Load the data into a pandas dataframe
df = pd.read_csv(data_file, sep='\t', header=None, names=['user_id', 'song_id', 'play_count'])

num_users = df['user_id'].nunique()
num_songs = df['song_id'].nunique()
# print(num_users)
# print(num_songs)

# # Train the model and save it to a file
# algorithm = SVD()
# trainset, testset = split(df)
# train_and_save_model(trainset, algorithm, dump_file)
# rmse = compute_rmse(dump_file)


_, algorithm = dump.load(dump_file)

# Get the inner ids of the users and songs
user_inner_ids = algorithm.trainset.all_users()
song_inner_ids = algorithm.trainset.all_items()

# Convert inner ids to raw ids
user_raw_ids = [algorithm.trainset.to_raw_uid(uid) for uid in user_inner_ids]
song_raw_ids = [algorithm.trainset.to_raw_iid(iid) for iid in song_inner_ids]

# TODO: how to make sure recommended songs are not already in the user's playlist?
# One way to solve this is to filter out songs that the user has already listened to.

rec_user_ids = user_raw_ids[:1]
rec_df = get_top_n(rec_user_ids, song_raw_ids, algorithm, n=100)
song_info_df = pd.read_csv('data/song_data.csv')
# Merge the recommendations with the song information
merged_df = pd.merge(rec_df, song_info_df, on='song_id')

merged_df.to_csv('results/recommendations.csv', index=False)

# Add text info to compute similarity between songs (working with the current data)
text_info = merged_df['title'] + ' ' + merged_df['release'] + ' ' + merged_df['artist_name']
merged_df["text"] = text_info

diversified_recommendation = diversity_using_mmr(merged_df, top_n=5, lambda_param=0.5)

# Save the recommendations to a file
diversified_recommendation.to_csv('results/recommendations_diversed.csv', index=False)