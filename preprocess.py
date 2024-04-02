import pandas as pd

pd.options.mode.chained_assignment = None

def normalize_user():
    """"Normalize the nDCG rating from 0 to 1"""
    df_recs = pd.read_csv("results/recommendations.tsv", sep='\t', names=['user_id', 'movie_id', 'score'])
    df_recs_norm = pd.DataFrame()

    user_ids = df_recs['user_id'].unique()
    # recommendations for one user
    for user_id in user_ids:
        df_1 = df_recs[df_recs['user_id'] == user_id]

        # normalize score between 0 and 1, should be done per user
        max_score, min_score = df_1['score'].max(), df_1['score'].min()
        df_1['score'] = (df_1['score'] - min_score) / (max_score - min_score + 1e-6)

        # fill in dataframe
        df_recs_norm = pd.concat([df_recs_norm, df_1])

    df_recs_norm.to_csv('results/normalized_recs.tsv', sep='\t', index=False)
