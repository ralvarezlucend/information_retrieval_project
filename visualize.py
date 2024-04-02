import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read the data
recs = pd.read_csv("results/normalized_recs.tsv", sep='\t')
recs = recs[recs['user_id'] == 2][:10] # top 10 original recommendations for user 2
div_recs = pd.read_csv("results/diversed_recs.tsv", sep='\t')
movie = pd.read_csv("new_movie_data/final_cleaned_2.csv")
genre = pd.read_csv("mappings/ids_to_genres.csv")
genre.set_index('genre_id', inplace=True)

# merge to get more information about the movies
cols = ['title', 'genres']
div_merge = pd.merge(div_recs, movie, left_on='movie_id', right_on='id')[cols]
rec_merge = pd.merge(recs, movie, left_on='movie_id', right_on='id')[cols]

# show genres in readable format
div_merge['genres'] = div_merge['genres'].apply(lambda x: np.array([genre.loc[genre_id] for genre_id in eval(x)]).flatten())
rec_merge['genres'] = rec_merge['genres'].apply(lambda x: np.array([genre.loc[genre_id] for genre_id in eval(x)]).flatten())

def rerank(list1, list2):
    reranking, reranked_names = [], []
    old, new = [], []
    
    for i, el1 in enumerate(list1):
        for j, el2 in enumerate(list2):
            if el1 == el2:
                reranked_names.append(el1)
                reranking.append({
                    "name": el1, 
                    "method": ['ORIGINAL', 'DIVERSED'],
                    "rank": [i+1, j+1]
                })
                
    for i, el1 in enumerate(list1):
        if not el1 in reranked_names:  # noqa: E713
            old.append({"name": el1, "rank": i+1})
            
    for j, el2 in enumerate(list2):
        if not el2 in reranked_names:  # noqa: E713
            new.append({"name": el2, "rank": j+1})
    
    return reranking, old, new

def plot(list1, list2):
    reranking, old, new = rerank(rec_list, div_list)

    fig, ax = plt.subplots()

    # plot change of rank
    for el in reranking:
        ax.plot(el["method"], 
            el["rank"], 
            "o-",                       # format of marker / format of line
            markerfacecolor="white")

        ax.annotate(el["name"], 
                    xy=(1, el["rank"][-1]), 
                    xytext=(1.1, el["rank"][-1]), 
                    va="center")
    
    # plot new recommendations in diversed list
    for el in new:
        ax.annotate(el["name"], 
                    xy=(1, el["rank"]), 
                    xytext=(1.1, el["rank"]), 
                    va="center")
        

    plt.gca().invert_yaxis()
    top_n = 10
    plt.yticks(np.arange(1, top_n+1))

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.show()


rec_list = list(rec_merge['title'])
div_list = list(div_merge['title'])

# print("Original list:", rec_list)
# print("Diversed list:", div_list)
# plot(rec_list, div_list)