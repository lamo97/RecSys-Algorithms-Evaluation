# In[]: Setup e imports
from clayrs import content_analyzer as ca
from clayrs import recsys as rs
import sys
import runRSUtils as rsutils
import os.path

path = 'D:/Repository/RecSys-Algorithms-Evaluation/'

# ------------- CAMPI ------------- 
# description
# genres
# tags
# reviews
# description,genres,tags
# description,genres,reviews
# description,tags,reviews
# genres,tags,reviews
# description,genres,tags,reviews

fields = sys.argv[1].split(",")

# contiene informazioni sull'esecuzione corrente
run = {
    "fields_num" : len(fields),
    "fields" : sys.argv[1],
    "dataset" : sys.argv[3],
    "representation" : "",
    "algorithm" : "",
    "methodology" : ""
}

# scelta del dataset
if (run["dataset"] == '100k'):
    dataset_path = path + 'Dataset/Movielens 100k/'
else:
    dataset_path = path + 'Dataset/Movielens 1M/'

rs_path = path + f'RS Results {run["dataset"]}/'


#apretura file dei ratings
ratings = ca.Ratings(ca.CSVFile(dataset_path + 'ratings.csv'))

if(os.path.exists(f'{dataset_path}test_set.csv') and os.path.exists(f'{dataset_path}training_set.csv')):
    # ricerca dei file di test e training set
    print('Verranno utilizzati il Test Set e Training Set già presenti!')
    
    train_list, test_list = []
    train_list.append(ca.Rank(ca.CSVFile(dataset_path + 'test_set.csv')))
    train_list.append(ca.Rank(ca.CSVFile(dataset_path + 'training_set.csv')))

else:
    # split del dataset
    print('Test Set e Training Set non trovati, si procede allo splitting del dataset...')

    train_list, test_list = rs.HoldOutPartitioning(train_set_size=0.8).split_all(ratings)

    # salvataggio su file
    test_list[0].to_csv(dataset_path, 'test_set', overwrite=True)
    train_list[0].to_csv(dataset_path, 'training_set', overwrite=True)

# scelta delle rappresentazioni
if (sys.argv[2] == 'all'):
    representations_list =  [
        'SK-TFIDF',
        'Word2Vec', 'Doc2Vec',
        'GensimLDA','GensimRandomIndexing', 'GensimFastText', 'GensimLSA',
        'Word2Doc-GloVe','Sentence2Doc-Sbert']
else:
    representations_list = sys.argv[2].split(",")

# esecuzione algoritmi in base al numero di campi
if(run["fields_num"] == 1):

    # Centroid Vector
    for rep in representations_list:
        run['representation'] = rep

        centroid_vector = rs.CentroidVector({fields[0]:[rep]}, similarity = rs.CosineSimilarity())
        run['algorithm'] = "Centroid Vector"
        rsutils.predict(centroid_vector, run, train_list, test_list, ratings)

    # Logistic Regression
    for rep in representations_list:
        run['representation'] = rep
        logistic_regression = rs.ClassifierRecommender({fields[0]:[rep]}, rs.SkLogisticRegression())
        run['algorithm'] = "Logistic Regression"
        rsutils.predict(logistic_regression, run, train_list, test_list, ratings)

    # Random Forests
    for rep in representations_list:
        run['representation'] = rep

        random_forests = rs.ClassifierRecommender({fields[0]:[rep]}, rs.SkRandomForest(n_estimators = 145))
        run['algorithm'] = "Random Forest"
        rsutils.predict(random_forests, run, train_list, test_list, ratings)

    # SVC
    for rep in representations_list:
        run['representation'] = rep
        
        svc = rs.ClassifierRecommender({fields[0]:[rep]}, rs.SkSVC())
        run['algorithm'] = "SVC"
        rsutils.predict(svc, run, train_list, test_list, ratings)

if(run["fields_num"] == 3):
    # Centroid Vector
    for rep in representations_list:
        run['representation'] = rep

        centroid_vector = rs.CentroidVector(
            {
                fields[0]:[rep], 
                fields[1]:[rep], 
                fields[2]:[rep]
            }, 
            similarity = rs.CosineSimilarity()
        )
        run['algorithm'] = "Centroid Vector"
        rsutils.predict(centroid_vector, run, train_list, test_list, ratings)

    # Logistic Regression
    for rep in representations_list:
        run['representation'] = rep

        logistic_regression = rs.ClassifierRecommender(
            {
                fields[0]:[rep], 
                fields[1]:[rep], 
                fields[2]:[rep]
            }, 
            rs.SkLogisticRegression()
        )
        run['algorithm'] = "Logistic Regression"
        rsutils.predict(logistic_regression, run, train_list, test_list, ratings)

    # Random Forests
    for rep in representations_list:
        run['representation'] = rep

        random_forests = rs.ClassifierRecommender(
            {
                fields[0]:[rep], 
                fields[1]:[rep], 
                fields[2]:[rep]
            }, 
            rs.SkRandomForest(n_estimators = 145)
        )
        run['algorithm'] = "Random Forest"
        rsutils.predict(random_forests, run, train_list, test_list, ratings)

    # SVC
    for rep in representations_list:
        run['representation'] = rep

        svc = rs.ClassifierRecommender(
            {
                fields[0]:[rep], 
                fields[1]:[rep], 
                fields[2]:[rep]
            }, 
            rs.SkSVC()
        )
        run['algorithm'] = "SVC"
        rsutils.predict(svc, run, train_list, test_list, ratings)

# Algoritmi: ------------------------------------ 4 CAMPI ------------------------------------
if(run["fields_num"] == 4):

    # Centroid Vector
    for rep in representations_list:
        run['representation'] = rep
        
        centroid_vector = rs.CentroidVector(
            {
                fields[0]:[rep], 
                fields[1]:[rep], 
                fields[2]:[rep],
                fields[3]:[rep]
            }, 
            similarity = rs.CosineSimilarity()
        )
        run['algorithm'] = "Centroid Vector"
        rsutils.predict(centroid_vector, run, train_list, test_list, ratings)

    # Logistic Regression
    for rep in representations_list:
        run['representation'] = rep    
        
        logistic_regression = rs.ClassifierRecommender(
            {
                fields[0]:[rep], 
                fields[1]:[rep], 
                fields[2]:[rep],
                fields[3]:[rep]
            }, 
            rs.SkLogisticRegression()
        )
        run['algorithm'] = "Logistic Regression"
        rsutils.predict(logistic_regression, run, train_list, test_list, ratings)

    # Random Forests
    for rep in representations_list:
        run['representation'] = rep
        
        random_forests = rs.ClassifierRecommender(
            {
                fields[0]:[rep], 
                fields[1]:[rep], 
                fields[2]:[rep],
                fields[3]:[rep]
            }, 
            rs.SkRandomForest(n_estimators = 145)
        )
        run['algorithm'] = "Random Forest"
        rsutils.predict(random_forests, run, train_list, test_list, ratings)

    # SVC
    for rep in representations_list:
        run['representation'] = rep
        
        svc = rs.ClassifierRecommender(
            {
                fields[0]:[rep], 
                fields[1]:[rep], 
                fields[2]:[rep],
                fields[3]:[rep]
            }, 
            rs.SkSVC()
        )
        run['algorithm'] = "SVC"
        rsutils.predict(svc, run, train_list, test_list, ratings)
        
# %%
