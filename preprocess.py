import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None

def convert_dict(column_name,key,value,df):
    """store just the ids and None/NaN otherwise"""
    d = dict()
    for i in df[column_name]:
        j = eval(i)
        if j.get('id') != None:
            d.update({j.get(key) : j.get(value)})
    df[column_name] = [eval(i).get(key) for i in df[column_name]]
    return d,df

def convert_list_of_dict(column_name,key,value,df):
    """store just the list of ids in the df and dictionary stores all the mappings"""
    d_ = dict()
    l = []
    for i in df[column_name]:
        l2 = []
        for j in eval(i):
            l2.append(j.get(key))
            d_.update({j.get(key): j.get(value)})
        l.append(l2)
    df[column_name] = l
    return d_,df

def store_dict_to_df(d,index_name,value_name,path):
    """pass the dictionary and convert that to data frame"""
    print(f"stored {index_name} ")
    df = pd.DataFrame(d.items(),columns = [index_name,value_name])
    df.to_csv(f"{path}",index=False)
    
def preprocess(path,new_path):
    """Convert the dictionaries to just the ids and storing it in csv files"""
    df = pd.read_csv(path)
    
    for i in df.columns:
        col = df[f"{i}"]
        if type(col.iloc[0]) == str:
            try:
                if type(eval(col.iloc[0])) == dict:
                    if i in ['belongs_to_collection']:
                        d,df = convert_dict(i,'id','name',df)
                        store_dict_to_df(d,'collection_id','collection_name','./dataframes_/ids_to_collection.csv')
                        
                if type(eval(col.iloc[0])) == list:                        
                    if i in ['genres','production_companies','keywords','cast','crew']:
                        d,df = convert_list_of_dict(i,'id','name',df)
                        store_dict_to_df(d,f'{i}_id',f'{i}',f'./dataframes_/ids_to_{i}.csv')

                    if i == 'spoken_languages':
                        d,df = convert_list_of_dict(i,'iso_639_1','name',df)
                        store_dict_to_df(d,'lang_id','iso','./dataframes_/ids_to_iso.csv')       

            except:
                continue
    df.to_csv(new_path)
    return df                

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
