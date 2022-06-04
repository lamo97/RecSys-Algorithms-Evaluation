#!/usr/bin/env python
# coding: utf-8

# In[1]:
import pandas as pd
import csv

# correzione dell'ordine di stampa
import functools
from operator import rshift
print = functools.partial(print, flush=True)

# import dei moduli per Content Analyzer, Recommender System e Evaluation come librerie
from clayrs import content_analyzer as ca
from clayrs import recsys as rs
from clayrs import evaluation as eva

# path del dataset
path = 'C:/Users/glamo/Desktop/Repository/RecSys-Algorithms-Evaluation/Dataset/Movielens 100k/'

# apertura del file contenente i film
items = open(path + 'items_info.json')

# apertura del file con i ratings
ratings = open(path + 'ratings.csv')

# configurazione del content analyzer
ca_config = ca.ItemAnalyzerConfig(
    source = ca.JSONFile(path + 'items_info.json'),
    id = 'movielens_id',
    output_directory = path + 'movies_codified/'
)


# In[2]:

# Rappresentazione con TF-IDF (SkLearn) w/ Plot
ca_config.add_single_config('plot',
                            preprocessing=ca.NLTK(stopwords_removal=True, lemmatization=True),
                            id='tfidf')

# In[3]:





# In[4]:


# In[5]:


# serializzazione degli item
ca.ContentAnalyzer(config = ca_config).fit()


# In[6]:


# Recommender: Centroid Vector Algorithm
ratings = ca.Ratings(ca.CSVFile(path + 'ratings.csv'))

centroid_vector = rs.CentroidVector({'plot': 'tfidf'},
                                    similarity = rs.CosineSimilarity())                                    


# In[7]:


# Split Test Set e Training Set
train_list, test_list = rs.KFoldPartitioning(n_splits=2).split_all(ratings)
                        #s.KFoldPartitioning(n_splits=2).split_all(ratings)

first_training_set = train_list[0]

cbrs = rs.ContentBasedRS(centroid_vector, 
                         first_training_set, 
                         (path + '/movies_codified'))

first_test_set = test_list[0]

rank = cbrs.fit_rank(first_test_set, 
                     user_id_list = ['8', '2', '1'],
                     n_recs = 3)

print(rank)


# In[9]:


result_list = []

for train_set, test_set in zip(train_list, test_list):
    cbrs = rs.ContentBasedRS(centroid_vector, train_set, (path + '/movies_codified'))
    rank_to_append = cbrs.fit_rank(test_set)
    result_list.append(rank_to_append)


# In[10]:



# In[ ]:




