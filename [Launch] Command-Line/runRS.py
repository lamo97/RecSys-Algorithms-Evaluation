# In[]: Setup e imports
from clayrs import content_analyzer as ca
from clayrs import recsys as rs
import sys
import runRSUtils as rsutils

path = 'D:/Repository/RecSys-Algorithms-Evaluation/'

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

# split del dataset
train_list, test_list = rs.HoldOutPartitioning(train_set_size=0.8).split_all(ratings)

# salvataggio della ground truth su file
test_list[0].to_csv(rs_path, 'test_set', overwrite=True)

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
    for rep in representations_list:
        run['representation'] = rep

        # Centroid Vector
        centroid_vector = rs.CentroidVector({fields[0]:[rep]}, similarity = rs.CosineSimilarity())
        run['algorithm'] = "Centroid Vector"
        rsutils.predict(centroid_vector, run, train_list, test_list, ratings)

        # Logistic Regression
        logistic_regression = rs.ClassifierRecommender({fields[0]:[rep]}, rs.SkLogisticRegression())
        run['algorithm'] = "Logistic Regression"
        rsutils.predict(logistic_regression, run, train_list, test_list, ratings)

        # Random Forests
        random_forests = rs.ClassifierRecommender({fields[0]:[rep]}, rs.SkRandomForest(n_estimators = 145))
        run['algorithm'] = "Random Forest"
        rsutils.predict(random_forests, run, train_list, test_list, ratings)

        # SVC
        svc = rs.ClassifierRecommender({fields[0]:[rep]}, rs.SkSVC())
        run['algorithm'] = "SVC"
        rsutils.predict(svc, run, train_list, test_list, ratings)

if(run["fields_num"] == 3):
    for rep in representations_list:
        run['representation'] = rep

        # Centroid Vector
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

if(run["fields_num"] == 4):
    for rep in representations_list:
        run['representation'] = rep

        # Centroid Vector
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
