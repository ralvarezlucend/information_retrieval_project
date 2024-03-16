from surprise import Reader, Dataset
from surprise.model_selection import train_test_split
from surprise import accuracy
from surprise import dump

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
    return  rmse


def get_top_n(user_id, song_ids, algorithm, n):
    """
    user_id - raw id of the user (uid)
    song_ids - list of raw ids of the songs (iid)
    n - number of recommendations to return
    """
    recommendations = [algorithm.predict(uid=user_id, iid=song_id) for song_id in song_ids]
    recommendations.sort(key=lambda x: x.est, reverse=True)
    top_n = recommendations[:n]
    return top_n