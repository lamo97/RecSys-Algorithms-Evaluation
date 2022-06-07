# In[1]: Setup
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
representation = 'RandomIndexing'

# In[]: 
# ***************************************
# Rappresentazione: RandomIndexing | Plot
# ***************************************
ca_config.add_single_config(
    'plot',
    ca.FieldConfig(
        ca.WordEmbeddingTechnique(
            ca.GensimRandomIndexing('glove-twitter-50'),
        ),
        ca.NLTK(stopwords_removal=True, lemmatization=True),
        id='Word2Doc'
    )
)

# serializzazione degli item
ca.ContentAnalyzer(config = ca_config).fit()

# inserimento del campo rappresentato nella lista
fields.append('plot')

# In[]: 
# ***********************************
# Rappresentazione: RandomIndexing | Genres
# ***********************************
ca_config.add_single_config(
    'genres',
    ca.FieldConfig(
        ca.WordEmbeddingTechnique(
            ca.GensimLatentSemanticAnalysis('glove-twitter-50'),
        ),
        ca.NLTK(stopwords_removal=True, lemmatization=True),
        id='Word2Doc'
    )
)
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

# we save the result of each split numbered
#for i, rank_generated in enumerate(result_list, start=1):
#    rank_generated.to_csv(file_name=f'rank_split_{i}')

# we save  the result of each split numbered
#for i, test_set in enumerate(test_list, start=1):
#    test_set.to_csv(file_name=f'truth_split_{i}')

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
    cbrs = rs.ContentBasedRS(logistic_regression, train_set, (path + '/movies_codified'))
    rank_to_append = cbrs.fit_rank(test_set)
    result_list.append(rank_to_append)

# we save the result of each split numbered
#for i, rank_generated in enumerate(result_list, start=1):
#    rank_generated.to_csv(file_name=f'rank_split_2{i}')

# we save  the result of each split numbered
#for i, test_set in enumerate(test_list, start=1):
#    test_set.to_csv(file_name=f'truth_split_2{i}')

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

result_list = []
for train_set, test_set in zip(train_list, test_list):
    cbrs = rs.ContentBasedRS(random_forest, train_set, (path + '/movies_codified'))
    rank_to_append = cbrs.fit_rank(test_set)
    result_list.append(rank_to_append)

# In[]:
# --------------------------------
# Linear SVM
# --------------------------------
print('Fields:', fields)
print('Representation:', representation)
print('Algorithm: Linear SVM')

n_fields = len(fields)
ratings = ca.Ratings(ca.CSVFile(path + 'ratings.csv'))

if(n_fields == 1):
    linear_svm = rs.ClassifierRecommender( {    fields[0]: [representation]}, rs.SkSVC())
elif(n_fields == 2):
    linear_svm = rs.ClassifierRecommender( {    fields[0]: [representation],
                                                fields[1]: [representation]}, rs.SkSVC())
elif(n_fields == 3):
    linear_svm = rs.ClassifierRecommender( {    fields[0]: [representation],
                                                fields[1]: [representation],
                                                fields[2]: [representation]}, rs.SkSVC())

train_list, test_list = rs.KFoldPartitioning(n_splits = 2).split_all(ratings)                      
first_training_set = train_list[0]

cbrs = rs.ContentBasedRS(linear_svm, first_training_set, (path + '/movies_codified'))
first_test_set = test_list[0]
rank = cbrs.fit_rank(first_test_set, user_id_list = ['8', '2', '1'], n_recs = 3)

result_list = []
for train_set, test_set in zip(train_list, test_list):
    cbrs = rs.ContentBasedRS(linear_svm, train_set, (path + '/movies_codified'))
    rank_to_append = cbrs.fit_rank(test_set)
    result_list.append(rank_to_append)

# we save the result of each split numbered
#for i, rank_generated in enumerate(result_list, start=1):
#    rank_generated.to_csv(file_name=f'rank_split_2{i}')

# we save  the result of each split numbered
#for i, test_set in enumerate(test_list, start=1):
#    test_set.to_csv(file_name=f'truth_split_2{i}')

# In[]:
# --------------------------------
# K-NN
# --------------------------------
print('Fields:', fields)
print('Representation:', representation)
print('Algorithm: K-NN')

n_fields = len(fields)
ratings = ca.Ratings(ca.CSVFile(path + 'ratings.csv'))

if(n_fields == 1):
    knn = rs.ClassifierRecommender( {   fields[0]: [representation]}, rs.SkKNN())
elif(n_fields == 2):
    knn = rs.ClassifierRecommender( {   fields[0]: [representation],
                                        fields[1]: [representation]}, rs.SkKNN())
elif(n_fields == 3):
    knn = rs.ClassifierRecommender( {   fields[0]: [representation],
                                        fields[1]: [representation],
                                        fields[2]: [representation]}, rs.SkKNN())

train_list, test_list = rs.KFoldPartitioning(n_splits = 2).split_all(ratings)                      
first_training_set = train_list[0]

cbrs = rs.ContentBasedRS(knn, first_training_set, (path + '/movies_codified'))
first_test_set = test_list[0]
rank = cbrs.fit_rank(first_test_set, user_id_list = ['8', '2', '1'], n_recs = 3)

result_list = []
for train_set, test_set in zip(train_list, test_list):
    cbrs = rs.ContentBasedRS(knn, train_set, (path + '/movies_codified'))
    rank_to_append = cbrs.fit_rank(test_set)
    result_list.append(rank_to_append)

# In[]:
# --------------------------------
# Decision Tree
# --------------------------------
print('Fields:', fields)
print('Representation:', representation)
print('Algorithm: Decision Tree')

n_fields = len(fields)
ratings = ca.Ratings(ca.CSVFile(path + 'ratings.csv'))

if(n_fields == 1):
    decision_tree = rs.ClassifierRecommender( { fields[0]: [representation]}, rs.SkDecisionTree())
elif(n_fields == 2):
    decision_tree = rs.ClassifierRecommender( { fields[0]: [representation],
                                                fields[1]: [representation]}, rs.SkDecisionTree())
elif(n_fields == 3):
    decision_tree = rs.ClassifierRecommender( { fields[0]: [representation],
                                                fields[1]: [representation],
                                                fields[2]: [representation]}, rs.SkDecisionTree())

train_list, test_list = rs.KFoldPartitioning(n_splits = 2).split_all(ratings)                      
first_training_set = train_list[0]

cbrs = rs.ContentBasedRS(decision_tree, first_training_set, (path + '/movies_codified'))
first_test_set = test_list[0]
rank = cbrs.fit_rank(first_test_set, user_id_list = ['8', '2', '1'], n_recs = 3)

result_list = []
for train_set, test_set in zip(train_list, test_list):
    cbrs = rs.ContentBasedRS(decision_tree, train_set, (path + '/movies_codified'))
    rank_to_append = cbrs.fit_rank(test_set)
    result_list.append(rank_to_append)

# In[]:
# --------------------------------
# Gaussian Preocess
# --------------------------------
print('Fields:', fields)
print('Representation:', representation)
print('Algorithm: Gaussian Preocess')

n_fields = len(fields)
ratings = ca.Ratings(ca.CSVFile(path + 'ratings.csv'))

if(n_fields == 1):
    gaussian_process = rs.ClassifierRecommender( {  fields[0]: [representation]}, rs.SkGaussianProcess())
elif(n_fields == 2):
    gaussian_process = rs.ClassifierRecommender( {  fields[0]: [representation],
                                                    fields[1]: [representation]}, rs.SkGaussianProcess())
elif(n_fields == 3):
    gaussian_process = rs.ClassifierRecommender( {  fields[0]: [representation],
                                                    fields[1]: [representation],
                                                    fields[2]: [representation]}, rs.SkGaussianProcess())

train_list, test_list = rs.KFoldPartitioning(n_splits = 2).split_all(ratings)                      
first_training_set = train_list[0]

cbrs = rs.ContentBasedRS(gaussian_process, first_training_set, (path + '/movies_codified'))
first_test_set = test_list[0]
rank = cbrs.fit_rank(first_test_set, user_id_list = ['8', '2', '1'], n_recs = 3)

result_list = []
for train_set, test_set in zip(train_list, test_list):
    cbrs = rs.ContentBasedRS(gaussian_process, train_set, (path + '/movies_codified'))
    rank_to_append = cbrs.fit_rank(test_set)
    result_list.append(rank_to_append)

# In[]:
# --------------------------------
# Stochastic Gradient Descent Regression
# --------------------------------
print('Fields:', fields)
print('Representation:', representation)
print('Algorithm: Stochastic Gradient Descent Regression')

n_fields = len(fields)
ratings = ca.Ratings(ca.CSVFile(path + 'ratings.csv'))

if(n_fields == 1):
    gradient_descent = rs.ClassifierRecommender( {  fields[0]: [representation]}, rs.SkLinearRegression.())
elif(n_fields == 2):
    gradient_descent = rs.ClassifierRecommender( {  fields[0]: [representation],
                                                    fields[1]: [representation]}, rs.SkGaussianProcess())
elif(n_fields == 3):
    gradient_descents = rs.ClassifierRecommender( {  fields[0]: [representation],
                                                    fields[1]: [representation],
                                                    fields[2]: [representation]}, rs.SkGaussianProcess())

train_list, test_list = rs.KFoldPartitioning(n_splits = 2).split_all(ratings)                      
first_training_set = train_list[0]

cbrs = rs.ContentBasedRS(gradient_descent, first_training_set, (path + '/movies_codified'))
first_test_set = test_list[0]
rank = cbrs.fit_rank(first_test_set, user_id_list = ['8', '2', '1'], n_recs = 3)

result_list = []
for train_set, test_set in zip(train_list, test_list):
    cbrs = rs.ContentBasedRS(gradient_descent, train_set, (path + '/movies_codified'))
    rank_to_append = cbrs.fit_rank(test_set)
    result_list.append(rank_to_append)
    
# In[]:
# --------------------------------
# Ridge Regression
# --------------------------------
print('Fields:', fields)
print('Representation:', representation)
print('Algorithm: Ridge Regression')

n_fields = len(fields)
ratings = ca.Ratings(ca.CSVFile(path + 'ratings.csv'))

if(n_fields == 1):
    ridge_reg = rs.ClassifierRecommender( {  fields[0]: [representation]}, rs.SkRidge())
elif(n_fields == 2):
    ridge_reg = rs.ClassifierRecommender( {  fields[0]: [representation],
                                                    fields[1]: [representation]}, rs.SkRidge())
elif(n_fields == 3):
    ridge_reg = rs.ClassifierRecommender( {  fields[0]: [representation],
                                                    fields[1]: [representation],
                                                    fields[2]: [representation]}, rs.SkRidge())

train_list, test_list = rs.KFoldPartitioning(n_splits = 2).split_all(ratings)                      
first_training_set = train_list[0]

cbrs = rs.ContentBasedRS(ridge_reg, first_training_set, (path + '/movies_codified'))
first_test_set = test_list[0]
rank = cbrs.fit_rank(first_test_set, user_id_list = ['8', '2', '1'], n_recs = 3)

result_list = []
for train_set, test_set in zip(train_list, test_list):
    cbrs = rs.ContentBasedRS(ridge_reg, train_set, (path + '/movies_codified'))
    rank_to_append = cbrs.fit_rank(test_set)
    result_list.append(rank_to_append)

# %%
