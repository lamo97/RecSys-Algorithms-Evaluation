# In[1]: Setup
import re
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

# lista che contiene i campi rappresentati
fields = []
# rappresentazione utilizzata
representation = 'tfidf'

# In[]: 
# ********************************
# Rappresentazione: TFIDF | Plot
# ********************************
ca_config.add_single_config('plot', 
                            ca.FieldConfig(ca.SkLearnTfIdf(),
                                            preprocessing=ca.NLTK(stopwords_removal=True, lemmatization=True),
                                            id='tfidf'))

# serializzazione degli item
ca.ContentAnalyzer(config = ca_config).fit()

# inserimento del campo rappresentato nella lista
fields.append('plot')

# In[]: 
# ********************************
# Rappresentazione: TFIDF | Genres
# ********************************
ca_config.add_single_config('genres', 
                            ca.FieldConfig(ca.SkLearnTfIdf(),
                                            preprocessing=ca.NLTK(stopwords_removal=True, lemmatization=True),
                                            id='tfidf'))

# serializzazione degli item
ca.ContentAnalyzer(config = ca_config).fit()

# inserimento del campo rappresentato nella lista
fields.append('genres')

# In[]:
# --------------------------------
# Centroid Vector
# --------------------------------
print('Fields:', fields)
print('Representation:', representation)
print('Algorithm: Centroid Vector')

n_fields = len(fields)
ratings = ca.Ratings(ca.CSVFile(path + 'ratings.csv'))

if(n_fields == 1):
    centroid_vector = rs.CentroidVector({   fields[0]: [representation]}, similarity = rs.CosineSimilarity())
elif(n_fields == 2):
    centroid_vector = rs.CentroidVector({   fields[0]: [representation], 
                                            fields[1]: [representation]}, similarity = rs.CosineSimilarity())
elif(n_fields == 3):
    centroid_vector = rs.CentroidVector({   fields[0]: [representation], 
                                            fields[1]: [representation],
                                            fields[2]: [representation]}, similarity = rs.CosineSimilarity())                                        

train_list, test_list = rs.KFoldPartitioning(n_splits = 2).split_all(ratings)                         
first_training_set = train_list[0]

cbrs = rs.ContentBasedRS(centroid_vector, first_training_set, (path + '/movies_codified'))
first_test_set = test_list[0]
rank = cbrs.fit_rank(first_test_set, user_id_list = ['8', '2', '1'], n_recs = 3)

print(rank)

result_list = []
for train_set, test_set in zip(train_list, test_list):
    cbrs = rs.ContentBasedRS(centroid_vector, train_set, (path + '/movies_codified'))
    rank_to_append = cbrs.fit_rank(test_set)
    result_list.append(rank_to_append)

for result in result_list:
    print(result)

# we save the result of each split numbered
for i, rank_generated in enumerate(result_list, start=1):
    rank_generated.to_csv(file_name=f'rank_split_{i}')

# we save  the result of each split numbered
for i, test_set in enumerate(test_list, start=1):
    test_set.to_csv(file_name=f'truth_split_{i}')

# In[]:
# --------------------------------
# Logistic Regression
# --------------------------------
print('Fields:', fields)
print('Representation:', representation)
print('Algorithm: Logistic Regression')

n_fields = len(fields)
ratings = ca.Ratings(ca.CSVFile(path + 'ratings.csv'))

if(n_fields == 1):
    logistic_regression = rs.ClassifierRecommender( {   fields[0]: [representation]}, rs.SkLogisticRegression())
elif(n_fields == 2):
    logistic_regression = rs.ClassifierRecommender( {   fields[0]: [representation],
                                                        fields[1]: [representation]}, rs.SkLogisticRegression())
elif(n_fields == 3):
    logistic_regression = rs.ClassifierRecommender( {   fields[0]: [representation],
                                                        fields[1]: [representation],
                                                        fields[2]: [representation]}, rs.SkLogisticRegression())

train_list, test_list = rs.KFoldPartitioning(n_splits = 2).split_all(ratings)                       
first_training_set = train_list[0]

cbrs = rs.ContentBasedRS(logistic_regression, first_training_set, (path + '/movies_codified'))
first_test_set = test_list[0]
rank = cbrs.fit_rank(first_test_set, user_id_list = ['8', '2', '1'], n_recs = 3)

result_list = []
for train_set, test_set in zip(train_list, test_list):
    cbrs = rs.ContentBasedRS(centroid_vector, train_set, (path + '/movies_codified'))
    rank_to_append = cbrs.fit_rank(test_set)
    result_list.append(rank_to_append)

# we save the result of each split numbered
for i, rank_generated in enumerate(result_list, start=1):
    rank_generated.to_csv(file_name=f'rank_split_2{i}')

# we save  the result of each split numbered
for i, test_set in enumerate(test_list, start=1):
    test_set.to_csv(file_name=f'truth_split_2{i}')

# In[]:
# --------------------------------
# Random Forest
# --------------------------------
print('Fields:', fields)
print('Representation:', representation)
print('Algorithm: Random Forest')

n_fields = len(fields)
ratings = ca.Ratings(ca.CSVFile(path + 'ratings.csv'))

if(n_fields == 1):
    random_forest = rs.ClassifierRecommender( { fields[0]: [representation]}, rs.SkRandomForest(n_estimators = 145))
elif(n_fields == 2):
    random_forest = rs.ClassifierRecommender( { fields[0]: [representation],
                                                fields[1]: [representation]}, rs.SkRandomForest(n_estimators = 145))
elif(n_fields == 3):
    random_forest = rs.ClassifierRecommender( { fields[0]: [representation],
                                                fields[1]: [representation],
                                                fields[2]: [representation]}, rs.SkRandomForest(n_estimators = 145))

train_list, test_list = rs.KFoldPartitioning(n_splits = 2).split_all(ratings)                      
first_training_set = train_list[0]

cbrs = rs.ContentBasedRS(random_forest,  first_training_set, (path + '/movies_codified'))
first_test_set = test_list[0]
rank = cbrs.fit_rank(first_test_set, user_id_list = ['8', '2', '1'], n_recs = 3)

print(rank)

# In[]:
# --------------------------------
# Linear SVM
# --------------------------------
# %%
