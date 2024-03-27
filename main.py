import pandas as pd
from surprise import SVD, KNNBasic
from surprise import dump
from utils import transform_and_split, train_and_save_model, compute_rmse, get_top_n
import plotly.graph_objs as go
from plotly.offline import iplot

# Dataset taken from: "http://millionsongdataset.com/tasteprofile/"
# user_id and song_id fields are anonymized.
data_file = 'data/10000.txt'
# Use SVD as it scales well, KNN causes python process to crash (out of memory)
dump_file = 'models/svd.pkl'

# Load the data into a pandas dataframe
df = pd.read_csv(data_file, sep='\t', header=None, names=['user_id', 'song_id', 'play_count'])

# # filter out users with little ratings
min_user_ratings = 50
filter_users = df['user_id'].value_counts() > min_user_ratings
filter_users = filter_users[filter_users].index.tolist()

# # filter out songs with little ratings
# min_song_ratings = 50
# filter_songs = df['song_id'].value_counts() > min_song_ratings
# filter_songs = filter_songs[filter_songs].index.tolist()

# Filter the data frame
df_new = df[(df['user_id'].isin(filter_users))]
print('The original data frame shape:\t{}'.format(df.shape))
print('The new data frame shape:\t{}'.format(df_new.shape))

# Plot the number of ratings per user
# data = df.groupby('user_id')['play_count'].count()
# trace = go.Histogram(x = data.values)
# fig = go.Figure(data=[trace])
# fig.update_layout(xaxis_title="ratings per user", yaxis_title="count")
# iplot(fig)

# # Plot the number of ratings per song
# data = df.groupby('song_id')['play_count'].count()
# trace = go.Histogram(x = data.values)
# fig = go.Figure(data=[trace])
# fig.update_layout(xaxis_title="ratings per song", yaxis_title="count")
# iplot(fig)

# Print some information about the data
# print("Number of unique users:", df['user_id'].nunique())
# print("Number of unique songs:", df['song_id'].nunique())
# print("Number of records", df.shape[0])

# # Train the model and save it to a file
# algorithm = KNNBasic()
# trainset, testset = transform_and_split(df_new, test_size=0.3)
# train_and_save_model(algorithm, trainset, dump_file)
# rmse = compute_rmse(algorithm, testset)


# _, algorithm = dump.load(dump_file)


# # Get the inner ids of the users and songs
# user_inner_ids = algorithm.trainset.all_users()
# song_inner_ids = algorithm.trainset.all_items()

# # Convert inner ids to raw ids
# user_raw_ids = [algorithm.trainset.to_raw_uid(uid) for uid in user_inner_ids]
# song_raw_ids = [algorithm.trainset.to_raw_iid(iid) for iid in song_inner_ids]

# # # TODO: how to make sure recommended songs are not already in the user's playlist?
# # # One way to solve this is to filter out songs that the user has already listened to.

# rec_df = get_top_n(user_raw_ids[:3], song_raw_ids, algorithm, n=5)
# song_info_df = pd.read_csv('data/song_data.csv')
# # Merge the recommendations with the song information
# merged_df = pd.merge(rec_df, song_info_df, on='song_id')
# # Save the recommendations to a file
# merged_df.to_csv('results/recommendations.csv', index=False)