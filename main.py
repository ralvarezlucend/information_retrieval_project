import pandas as pd
from surprise import SVD
from surprise import dump
from utils import split, train_and_save_model, compute_rmse, get_top_n

# Dataset taken from: "http://millionsongdataset.com/tasteprofile/"
# user_id and song_id fields are anonymized.
data_file = 'data/10000.txt'
# Use SVD as it scales well, KNN causes python process to crash (out of memory)
dump_file = 'models/svd.pkl'

# Load the data into a pandas dataframe
df = pd.read_csv(data_file, sep='\t', header=None, names=['user_id', 'song_id', 'play_count'])

# Print some information about the data
print("Number of unique users:", df['user_id'].nunique())
print("Number of unique songs:", df['song_id'].nunique())
print("Number of records", df.shape[0])

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

# Get the top 10 recommendations for a user
user = user_raw_ids[1]
recommendations = get_top_n(user, song_raw_ids, algorithm, n=10)

print("Top 10 recommendations for user", user)
for i, r in enumerate(recommendations):
    print(i+1, r.iid, r.est)