import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from diversify import diversity_using_mmr
matplotlib.use('TkAgg')


def plot_lambda(files=False):
    """"
    Plot the mean normalized nDCG for different lambda parameters.
    files - set to True if the files were already generated
    """
    df_recs = pd.read_csv("results/normalized_recs.tsv", sep='\t')

    # Get recommendations for one user
    user = 2
    df_1 = df_recs[df_recs['user_id'] == user]

    top_n = 10
    recs = df_1[:top_n]

    # save diversified recommendations
    lambda_params = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    recs_means = []
    diversed_recs_data = []
    diversed_recs_means = []

    # 1 USER
    for lambda_param in lambda_params:
        if not files:
            diversed_recs = diversity_using_mmr(df_1, top_n, lambda_param=lambda_param)
            diversed_recs.to_csv(f'results/diversed_recs_lambda{lambda_param}.tsv', sep='\t', index=False,
                                 header=['movie_id', 'score'])

        diversed_recs = pd.read_csv(f'results/diversed_recs_lambda{lambda_param}.tsv', sep='\t', header=0)

        weights = np.arange(1, 0, -0.1)

        weighted_mean_recs = np.average(recs['score'], weights=weights)

        # recs_mean = recs['score'].mean()
        recs_means.append(weighted_mean_recs)

        weighted_mean_div_recs = np.average(diversed_recs['score'], weights=weights)

        # diversed_recs_mean = diversed_recs['score'].astype(float).mean()
        # diversed_recs_means.append(diversed_recs_mean)

        diversed_recs_means.append(weighted_mean_div_recs)

        diversed_recs_data.append(diversed_recs['score'].astype(float))

        # The new diversified recommendations that do not appear in the initial recommendations
        not_common = set(diversed_recs.loc[:, 'movie_id']) - set(recs.loc[:, 'movie_id'])
        print('NOT COMMON:', not_common)

    plt.figure(figsize=(10, 7))
    plt.plot(lambda_params, recs_means, label='Weighted mean of the normalized relevance scores of elliot recommendations')  # Ensure the line is above the boxplots

    # Create box plots at the specified lambda positions with a smaller width
    plt.boxplot(diversed_recs_data, positions=lambda_params, widths=0.05, showmeans=True)

    # Plot a line connecting the means of the box plots
    plt.plot(lambda_params, diversed_recs_means, color='red', label='Distribution of the normalized relevance scores of diversification recommendations', linestyle='--')

    # Set x-axis limits to remove the empty space on the ends
    plt.xlim(min(lambda_params) - 0.1, max(lambda_params) + 0.1)

    plt.xlabel(r"$\lambda$")
    plt.ylabel("Mean normalized relevance score")
    plt.legend()
    plt.title('Weighted mean normalized relevance score of elliot recommendations and box plots of diversified recommendations')
    plt.savefig("box_plots.pdf")
    plt.show()


plot_lambda(files=False)
