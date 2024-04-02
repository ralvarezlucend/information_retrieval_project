import pandas as pd

from diversify import diversity_using_mmr
from preprocess import normalize_user

pd.options.mode.chained_assignment = None

# Normalize the ratings for all users
# normalize_user()

df_recs = pd.read_csv("results/normalized_recs.tsv", sep='\t')

# Get recommendations for one user
df_1 = df_recs[df_recs['user_id'] == 2]


top_n = 10

# save diversified recommendations
diversed_recs = diversity_using_mmr(df_1, top_n)
diversed_recs.to_csv('results/diversed_recs_1.tsv', sep='\t', index=False, header=['movie_id', 'score'])

# load diversified recommendations
diversed_recs = pd.read_csv('results/diversed_recs_1.tsv', sep='\t', header=0)


recs = df_1[:top_n]

print("user 2\n")
print("recs:\n", recs[['movie_id', 'score']])
print("\ndiversified recs:\n", diversed_recs)

print('\nnormal score:', recs['score'].mean())
print('diversed score', diversed_recs['score'].astype(float).mean())

not_common = set(diversed_recs.loc[:, 'movie_id']) - set(recs.loc[:, 'movie_id'])
print('NOT COMMON:', not_common)
