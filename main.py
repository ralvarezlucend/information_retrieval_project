import pandas as pd

from diversify import diversity_using_mmr

pd.options.mode.chained_assignment = None

df_recs = pd.read_csv("results/recommendations.tsv", sep='\t', names=['user_id', 'movie_id', 'score'])

# recommendations for one user
df_1 = df_recs[df_recs['user_id'] == 2]

# normalize score between 0 and 1, should be done per user
max_score, min_score = df_1['score'].max(), df_1['score'].min()
df_1['score'] = (df_1['score'] - min_score) / (max_score - min_score + 1e-6)

# save diversified recommendations
top_n = 10
# diversed_recs = diversity_using_mmr(df_1, top_n)
# diversed_recs.to_csv('results/diversed_recs.tsv', sep='\t', index=False, header=['movie_id', 'score'])

# load diversified recommendations
diversed_recs = pd.read_csv('results/diversed_recs.tsv', sep='\t', header=0)


recs = df_1[:top_n]

print('diversed score', diversed_recs['score'].astype(float).mean())
print('normal score:', recs['score'].mean())

not_common = set(diversed_recs.loc[:, 'movie_id']) - set(recs.loc[:, 'movie_id'])
print('NOT COMMON:', not_common)