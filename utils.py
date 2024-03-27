from surprise import Reader, Dataset
from surprise.model_selection import train_test_split
from surprise import accuracy
from surprise import dump
import pandas as pd
from tqdm import tqdm

def transform_and_split(df, test_size: float = 0.25):
	"""
	Splits a Pandas DataFrame into a Surprise trainset and a testset. 
	"""

	# Normalize the play_count column per user
	df['normalized_play_count'] = (df.groupby('user_id')['play_count']
								.transform(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)))
	
	# Rename columns to match surprise's expected input
	df.rename(columns={'user_id': 'userID', 'song_id': 'itemID', 'normalized_play_count': 'rating'},
		    inplace=True)
	
	# Keep only the columns we need
	df = df[["userID", "itemID", "rating"]]
	print(df.head())

	# Load the data into a surprise dataset and split it into a trainset and a testset
	reader = Reader(rating_scale=(0., 1.))
	data = Dataset.load_from_df(df, reader)
	trainset, testset = train_test_split(data, test_size=test_size)

	return trainset, testset


def train_and_save_model(algorithm, trainset, dump_file):
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
	rec = {"user_id": [], "song_id": [], "score": []}	
	for uid in tqdm(user_ids):
		user_rec = [algo.predict(uid, iid) for iid in song_ids]
		user_rec.sort(key=lambda x: x.est, reverse=True)		
		top_n = user_rec[:n]
		
		# save the recommendations
		for r in top_n:
			rec["user_id"].append(r.uid)
			rec["song_id"].append(r.iid)
			rec["score"].append(r.est)

	return pd.DataFrame(rec)


