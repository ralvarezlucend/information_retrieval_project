import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read the data
recs = pd.read_csv("results/normalized_recs.tsv", sep='\t')
recs = recs[recs['user_id'] == 2] # top 10 original recommendations for user 2
div_recs = pd.read_csv("results/diversed_recs_languages.tsv", sep='\t')
movie = pd.read_csv("new_movie_data/final_cleaned_2.csv")
# read the genre mapping
genre = pd.read_csv("mappings/ids_to_genres.csv")
genre.set_index('genre_id', inplace=True)
# read the keywords mapping
keyword = pd.read_csv("mappings/ids_to_keywords.csv")
keyword.set_index('keyword_id', inplace=True)
# read the crew mapping
crew = pd.read_csv("mappings/ids_to_director_name.csv")
crew.set_index('crew_id', inplace=True)
# read the spoken languages mapping
spoken_languages = pd.read_csv("mappings/ids_to_iso.csv")
spoken_languages.set_index('lang_id', inplace=True)

# merge to get more information about the movies
cols = ['title', 'spoken_languages']
div_merge = pd.merge(div_recs, movie, left_on='movie_id', right_on='id')[cols]
rec_merge = pd.merge(recs, movie, left_on='movie_id', right_on='id')[cols]

# # show genres in readable format
# div_merge['genres'] = div_merge['genres'].apply(lambda x: np.array([genre.loc[genre_id] for genre_id in eval(x)]).flatten())
# rec_merge['genres'] = rec_merge['genres'].apply(lambda x: np.array([genre.loc[genre_id] for genre_id in eval(x)]).flatten())

# show keywords in readable format
# div_merge['keywords'] = div_merge['keywords'].apply(lambda x: np.array([keyword.loc[keyword_id] for keyword_id in eval(x)]).flatten())
# rec_merge['keywords'] = rec_merge['keywords'].apply(lambda x: np.array([keyword.loc[keyword_id] for keyword_id in eval(x)]).flatten())

# show spoken languages in readable format
div_merge['spoken_languages'] = div_merge['spoken_languages'].apply(lambda x: np.array([spoken_languages.loc[lang_id] for lang_id in eval(x)]).flatten())
rec_merge['spoken_languages'] = rec_merge['spoken_languages'].apply(lambda x: np.array([spoken_languages.loc[lang_id] for lang_id in eval(x)]).flatten())

# # show crew in readable format
# div_merge['crew'] = div_merge['crew'].apply(lambda x: np.array([crew.loc[crew_id] for crew_id in eval(x)]).flatten())
# rec_merge['crew'] = rec_merge['crew'].apply(lambda x: np.array([crew.loc[crew_id] for crew_id in eval(x)]).flatten())

def rerank(list1, list2):
    reranking, reranked_names = [], []

    for i, el1 in enumerate(list1):
        for j, el2 in enumerate(list2):
            if el1 == el2:
                reranked_names.append(el1)
                reranking.append({
                    "name": el1, 
                    "method": ['ORIGINAL\nRANK', 'DIVERSIFIED\nRANK'],
                    "old_rank": i+1,
                    "new_rank": j+1,
                    "rank_change": [i+1, j+1], 
                    "is_top_n": i+1 <= 10
                })

    return reranking

def plot(list1, list2):
    reranking = rerank(list1, list2)

    fig, ax = plt.subplots()
    ax.tick_params(top=False, labeltop=True, bottom=False, labelbottom=False)
    ax.tick_params(left=False)

    # plot change of rank
    for el in reranking:
        ax.plot(
            el["method"], 
            el["rank_change"], 
            "o-",                       # format of marker / format of line
            markerfacecolor="white"    
        )

        ax.annotate(
            el["new_rank"], 
            xy=(1, el["rank_change"][-1]), 
            xytext=(1.1, el["rank_change"][-1]), 
            va="center",
            # color="black" if el["is_top_n"] else "green",
            weight="bold" if not el["is_top_n"] else "normal"
        )
        
    plt.gca().invert_yaxis()
    
    y_ticks = [r['old_rank'] for r in reranking if not r["is_top_n"] or r['old_rank'] <= 10]
    y_tick_labels = [r['name'] + "       " + str(r['old_rank']) for r in reranking if not r["is_top_n"] or r['old_rank'] <= 10] 
    plt.yticks(y_ticks, y_tick_labels)


    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.title(r"Movie reranking based on genres for user 2 with $\lambda$ = 0.5", loc="center", pad=30, wrap=True)
    # make figure wider
    fig.set_figwidth(8)
    plt.tight_layout()
    plt.show()
    # plt.savefig("results/reranking_genres_u2.pdf")

rec_list = list(rec_merge['title'])
div_list = list(div_merge['title'])

print("Original: ", rec_merge[:10])
print("Diversified: ", div_merge)

plot(rec_list, div_list)

# rec_content = rec_merge[:3]
# rec_content['genres'] = rec_content['genres'].apply(lambda x: ', '.join(x))
# rec_content['rank'] = np.arange(1, 4)
# rec_table = rec_content[['rank', 'title', 'genres']].to_latex(index=False)
# print(rec_table)