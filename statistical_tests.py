import pandas as pd
import numpy as np
from diversify import diversity_using_mmr
from tqdm import tqdm

# for reproducibility
np.random.seed(42)

recs = pd.read_csv("results/normalized_recs.tsv", sep='\t')
all_user_ids = list(recs['user_id'].unique())


def strategy_comparison(num_users, strategies):
    user_ids = np.random.choice(all_user_ids, num_users, replace=False)
    
    results = {
        "user": [],
        "strategy": [],
        "original_mean": [],
        "diversified_mean": []
    }

    for i, uid in tqdm(enumerate(user_ids)):
        df_or = recs[recs['user_id'] == uid]
        df_or = df_or.sort_values(by='score', ascending=False)
        # mean_or = df_or['score'][:10].mean()
        weights = np.arange(1, 0, -0.1) 
        weighted_mean_or = np.average(df_or['score'][:10], weights=weights)

        # diversify
        attributes = {
            'genres': 0, 
            'production_companies': 0, 
            'spoken_languages': 0, 
            'keywords': 0, 
            'crew': 0, 
            'cast': 0,
            'text': 0
        }

        for s in strategies:
            attributes[s] = 1
            df_div = diversity_using_mmr(df_or, top_n=10, lambda_param=0.5, attributes=attributes)
            # mean_div = df_div[1].mean()
            weighted_mean_div = np.average(df_div[1], weights=weights)
            attributes[s] = 0

            results["user"].append(i+1)
            results["strategy"].append(s)
            results["original_mean"].append(weighted_mean_or)
            results["diversified_mean"].append(weighted_mean_div)

    return pd.DataFrame(results)

num_users=20
strategies = ['genres', 'crew', 'text', 'production_companies']
# df_results = strategy_comparison(num_users, strategies)
# df_results.to_csv("results/strategy_comparison.tsv", sep='\t', index=False)

df_results = pd.read_csv("results/strategy_comparison.tsv", sep='\t')
df_results['difference'] = df_results['original_mean'] - df_results['diversified_mean']

# round to 3 decimal places
df_results['original_mean'] = df_results['original_mean'].round(3)
df_results['diversified_mean'] = df_results['diversified_mean'].round(3)
df_results['difference'] = df_results['difference'].round(3)

for s in strategies:
    df_s = df_results[df_results['strategy'] == s][['user', 'original_mean', 'diversified_mean', 'difference']]
    table = df_s.to_latex(index=False, caption=s, column_format='cccc')
    print(table)