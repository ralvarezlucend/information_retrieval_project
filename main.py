import pandas as pd

from diversify import diversity_using_mmr
from preprocess import normalize_user

pd.options.mode.chained_assignment = None

# Normalize the ratings for all users
normalize_user()
# Get recommendations for one user
df_recs = pd.read_csv("results/normalized_recs.tsv", sep='\t')
user = 2
df_1 = df_recs[df_recs['user_id'] == user]
# print(df_1[df_1['movie_id'] == 34])

# save diversified recommendations
top_n = 10
diversed_recs = diversity_using_mmr(df_1, top_n)
diversed_recs.to_csv('results/diversed_recs.tsv', sep='\t', index=False, header=['movie_id', 'score'])

# load diversified recommendations
diversed_recs = pd.read_csv('results/diversed_recs.tsv', sep='\t', header=0)


recs = df_1[:top_n]
print("user ", user, "\n")
print("recs:\n", recs[['movie_id', 'score']])
print("\ndiversified recs:\n", diversed_recs)
print('\nnormal score:', recs['score'].mean())
print('diversed score', diversed_recs['score'].astype(float).mean())
not_common = set(diversed_recs.loc[:, 'movie_id']) - set(recs.loc[:, 'movie_id'])
print('NOT COMMON:', not_common)
