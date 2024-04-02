import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from diversify import diversity_using_mmr
matplotlib.use('TkAgg')


def plot_lambda():
    """"Plot the mean normalized nDCG for different lambda parameters."""
    df_recs = pd.read_csv("results/normalized_recs.tsv", sep='\t')

    # Get recommendations for one user
    user = 2
    df_1 = df_recs[df_recs['user_id'] == user]

    top_n = 10
    recs = df_1[:top_n]

    # save diversified recommendations
    lambda_params = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    recs_means = []
    diversed_recs_means = []

    # 1 USER
    for lambda_param in lambda_params:
        diversed_recs = diversity_using_mmr(df_1, top_n, lambda_param=lambda_param)
        diversed_recs.to_csv(f'results/diversed_recs_lambda{lambda_param}=.tsv', sep='\t', index=False, header=['movie_id', 'score'])
        diversed_recs = pd.read_csv('results/diversed_recs.tsv', sep='\t', header=0)

        print("user ", user, "\n")
        print("recs:\n", recs[['movie_id', 'score']])
        print("\ndiversified recs:\n", diversed_recs)

        recs_mean = recs['score'].mean()
        recs_means.append(recs_mean)

        diversed_recs_mean = diversed_recs['score'].astype(float).mean()
        diversed_recs_means.append(diversed_recs_mean)

        # The new diversified recommendations that do not appear in the initial recommendations
        not_common = set(diversed_recs.loc[:, 'movie_id']) - set(recs.loc[:, 'movie_id'])
        print('NOT COMMON:', not_common)

    plt.plot(lambda_params, recs_means, label='mean nDCG recs')
    plt.plot(lambda_params, diversed_recs_means, label='mean nDCG diversified recs')

    plt.xlabel("Lambda")
    plt.ylabel("Mean normalized nDCG")
    plt.legend()
    plt.title('Mean nDCG of initial and diversified recommendations')
    plt.show()


plot_lambda()
