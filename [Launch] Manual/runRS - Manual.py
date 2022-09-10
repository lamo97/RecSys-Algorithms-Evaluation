# In[1]: Setup and Imports
from clayrs import content_analyzer as ca
from clayrs import recsys as rs
import runRSUtils as rsutils

path = 'D:/Repository/RecSys-Algorithms-Evaluation/'

# ------------- FEATURES ------------- 
# description
# genres
# tags
# reviews
# description,genres,tags
# description,genres,reviews
# description,tags,reviews
# genres,tags,reviews
# description,genres,tags,reviews

fields = ['description', 'genres', 'reviews']    # LIST of features used for the predictions
fields_string = 'description,genres,reviews'     # STRING of features separated by commas, used to determine the output path

# ------------------------ REPRESENTATIONS -------------------------
# 'SK-TFIDF', 'Word2Vec', 'Doc2Vec',
# 'GensimLDA','GensimRandomIndexing', 'GensimFastText', 'GensimLSA', 
# 'Word2Doc-GloVe','Sentence2Doc-Sbert'

# LIST of representations used for the predictions
representations_list = ['Doc2Vec']

# Info on the current run
run = {
    "fields_num" : len(fields),
    "fields" : fields_string,
    "dataset" : '1M',
    "representation" : "",
    "algorithm" : "",
    "methodology" : ""
}

# Chosed dataset for the experiments
dataset_path = path + 'Dataset/Movielens 1M/'

# Result Ranks' path
rs_path = path + f'[{run["dataset"]}] Result Ranks/'

# Opens the ratings file
ratings = ca.Ratings(ca.CSVFile(dataset_path + 'ratings.csv'))

# Dataset Split
train_list, test_list = rs.HoldOutPartitioning(train_set_size=0.8).split_all(ratings)

# Ground truth saved on file (for the current run, overwritten next time the script is launched)
test_list[0].to_csv(rs_path, 'test_set', overwrite=True)

# Algorithms: ------------------------------------ 1 FEATURE ------------------------------------
# In[]: Centroid Vector
if(run['fields_num'] == 1):
    for rep in representations_list:
        run['representation'] = rep

        centroid_vector = rs.CentroidVector({fields[0]:[rep]}, similarity = rs.CosineSimilarity())
        run['algorithm'] = "Centroid Vector"
        rsutils.predict(centroid_vector, run, train_list, test_list, ratings)
else:
    print("WRONG NUMBER OF FEATURES!")

# In[]: Logistic Regression
if(run['fields_num'] == 1):
    for rep in representations_list:
        run['representation'] = rep
        
        logistic_regression = rs.ClassifierRecommender({fields[0]:[rep]}, rs.SkLogisticRegression())
        run['algorithm'] = "Logistic Regression"
        rsutils.predict(logistic_regression, run, train_list, test_list, ratings)
else:
    print("WRONG NUMBER OF FEATURES!")

# In[] Random Forests
if(run['fields_num'] == 1):
    for rep in representations_list:
        run['representation'] = rep
        
        random_forests = rs.ClassifierRecommender({fields[0]:[rep]}, rs.SkRandomForest(n_estimators = 145))
        run['algorithm'] = "Random Forest"
        rsutils.predict(random_forests, run, train_list, test_list, ratings)
else:
    print("WRONG NUMBER OF FEATURES!")

# In[] SVC
if(run['fields_num'] == 1):
    for rep in representations_list:
        run['representation'] = rep
        
        svc = rs.ClassifierRecommender({fields[0]:[rep]}, rs.SkSVC())
        run['algorithm'] = "SVC"
        rsutils.predict(svc, run, train_list, test_list, ratings)
else:
    print("WRONG NUMBER OF FEATURES!")


# Algorithms: ------------------------------------ 3 FEATURES ------------------------------------
# In[]: Centroid Vector
if(run['fields_num'] == 3):
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
else:
    print("WRONG NUMBER OF FEATURES!")
    
# In[]: Logistic Regression
if(run['fields_num'] == 3):
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
else:
    print("WRONG NUMBER OF FEATURES!")
    
# In[]: Random Forests
if(run['fields_num'] == 3):
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
else:
    print("WRONG NUMBER OF FEATURES!")
    
# In[]: SVC
if(run['fields_num'] == 3):
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
else:
    print("WRONG NUMBER OF FEATURES!")


# Algorithm: ------------------------------------ 4 FEATURES ------------------------------------
# In[]: Centroid Vector
if(run['fields_num'] == 4):
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
else:
    print("WRONG NUMBER OF FEATURES!")
    
# In[]: Logistic Regression
if(run['fields_num'] == 4):
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
else:
    print("WRONG NUMBER OF FEATURES!")
    
# In[]: Random Forests
if(run['fields_num'] == 4):
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
else:
    print("WRONG NUMBER OF FEATURES!")
    
# In[]: SVC
if(run['fields_num'] == 4):
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
else:
    print("WRONG NUMBER OF FEATURES!")
            
# %%
