# In[]: Setup e imports
from clayrs import content_analyzer as ca
from clayrs import recsys as rs
import runRSUtils as rsutils

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

fields = ['description', 'genres', 'tags']    # INSERIRE QUI I CAMPI PER I QUALI EFFETTUARE LE PREDIZIONI
fields_string = 'description,genres,tags' # INSERIRE QUI LA STRINGA FORMATA DAI CAMPI, SENZA SPAZI

# ------------------------ RAPPRESENTAZIONI -------------------------
# 'SK-TFIDF', 'Word2Vec', 'Doc2Vec',
# 'GensimLDA','GensimRandomIndexing', 'GensimFastText', 'GensimLSA', 
# 'Word2Doc-GloVe','Sentence2Doc-Sbert'

# inserire le rappresentazioni per le quali effettuare le predizioni
representations_list = ['Sentence2Doc-Sbert']

# contiene informazioni sull'esecuzione corrente
run = {
    "fields_num" : len(fields),
    "fields" : fields_string,
    "dataset" : '1M',
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

# Algoritmi: ------------------------------------ 1 CAMPO ------------------------------------
# In[]: Centroid Vector
if(run['fields_num'] == 1):
    for rep in representations_list:
        run['representation'] = rep

        centroid_vector = rs.CentroidVector({fields[0]:[rep]}, similarity = rs.CosineSimilarity())
        run['algorithm'] = "Centroid Vector"
        rsutils.predict(centroid_vector, run, train_list, test_list, ratings)
else:
    print("SI STA CERCANO DI ESEGUIRE L'ALGORITMO CON UN NUMERO DI CAMPI ERRATO!")

# In[]: Logistic Regression
if(run['fields_num'] == 1):
    for rep in representations_list:
        run['representation'] = rep
        
        logistic_regression = rs.ClassifierRecommender({fields[0]:[rep]}, rs.SkLogisticRegression())
        run['algorithm'] = "Logistic Regression"
        rsutils.predict(logistic_regression, run, train_list, test_list, ratings)
else:
    print("SI STA CERCANO DI ESEGUIRE L'ALGORITMO CON UN NUMERO DI CAMPI ERRATO!")

# In[] Random Forests
if(run['fields_num'] == 1):
    for rep in representations_list:
        run['representation'] = rep
        
        random_forests = rs.ClassifierRecommender({fields[0]:[rep]}, rs.SkRandomForest(n_estimators = 145))
        run['algorithm'] = "Random Forest"
        rsutils.predict(random_forests, run, train_list, test_list, ratings)
else:
    print("SI STA CERCANO DI ESEGUIRE L'ALGORITMO CON UN NUMERO DI CAMPI ERRATO!")

# In SVC
if(run['fields_num'] == 1):
    for rep in representations_list:
        run['representation'] = rep
        
        svc = rs.ClassifierRecommender({fields[0]:[rep]}, rs.SkSVC())
        run['algorithm'] = "SVC"
        rsutils.predict(svc, run, train_list, test_list, ratings)
else:
    print("SI STA CERCANO DI ESEGUIRE L'ALGORITMO CON UN NUMERO DI CAMPI ERRATO!")


# Algoritmi: ------------------------------------ 3 CAMPI ------------------------------------
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
    print("SI STA CERCANO DI ESEGUIRE L'ALGORITMO CON UN NUMERO DI CAMPI ERRATO!")
    
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
    print("SI STA CERCANO DI ESEGUIRE L'ALGORITMO CON UN NUMERO DI CAMPI ERRATO!")
    
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
    print("SI STA CERCANO DI ESEGUIRE L'ALGORITMO CON UN NUMERO DI CAMPI ERRATO!")
    
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
    print("SI STA CERCANO DI ESEGUIRE L'ALGORITMO CON UN NUMERO DI CAMPI ERRATO!")
    


# Algoritmi: ------------------------------------ 4 CAMPI ------------------------------------
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
    print("SI STA CERCANO DI ESEGUIRE L'ALGORITMO CON UN NUMERO DI CAMPI ERRATO!")
    
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
    print("SI STA CERCANO DI ESEGUIRE L'ALGORITMO CON UN NUMERO DI CAMPI ERRATO!")
    
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
    print("SI STA CERCANO DI ESEGUIRE L'ALGORITMO CON UN NUMERO DI CAMPI ERRATO!")
    
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
    print("SI STA CERCANO DI ESEGUIRE L'ALGORITMO CON UN NUMERO DI CAMPI ERRATO!")
            
# %%
