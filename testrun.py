def predict(recommender, current_run):
    train_list, test_list = rs.KFoldPartitioning(n_splits = 2).split_all(ratings)                         
    first_training_set = train_list[0]

    cbrs = rs.ContentBasedRS(recommender, first_training_set, (path + '/movies_codified'))
    first_test_set = test_list[0]
    rank = cbrs.fit_rank(first_test_set, user_id_list = ['8', '2', '1'], n_recs = 3, methodology=current_run['methodology'])

    result_list = []
    for train_set, test_set in zip(train_list, test_list):
        cbrs = rs.ContentBasedRS(recommender, train_set, (path + '/movies_codified'))
        rank_to_append = cbrs.fit_rank(test_set, methodology=current_run['methodology'])
        result_list.append(rank_to_append)

    for result in result_list:
        print(result)

    evaluation(result_list, test_list, current_run)

def evaluation(result_list, test_list, current_run):
    # determina il cutoff in base al candidate item
    if(current_run['methodologyID'] == 'All Items'):
        cutoff_list = [10,20]
    else:
        cutoff_list = [5,10]


    catalog = set(ratings.item_id_column)   # catalog per la catalog coverage
    user_groups = {'popular_users': 0.3, 'medium_popular_users': 0.2, 'low_popular_users': 0.5} # user group per il Delta GAP
    
    for cutoff in cutoff_list:
        em = eva.EvalModel(
            pred_list=result_list,
            truth_list=test_list,
            metric_list=[
                eva.PrecisionAtK(k=cutoff),
                eva.RecallAtK(k=cutoff),
                eva.FMeasureAtK(k=cutoff),
                eva.NDCGAtK(k=cutoff),
                eva.MRRAtK(k=cutoff),
                eva.GiniIndex(),
                eva.CatalogCoverage(catalog),
                eva.DeltaGap(user_groups)
            ]
        )

        sys_result, users_result =  em.fit()
        results_path = 'C:/Users/glamo/Desktop/Repository/RecSys-Algorithms-Evaluation/Eval Results/'
        sys_result.to_csv(
            results_path + 'SYS - ' + sys.argv[1] + ' - ' + current_run['representation'] + ' - '
            + current_run['algorithm'] + ' (' + current_run['methodologyID'] + '@' + str(cutoff) + ').csv')

    #users_result.to_csv(results_path + 'USER - ' + sys.argv[1] + ' - ' + algorithm + ' (' + candidateID + ').csv')

import sys

# import dei moduli per Content Analyzer, Recommender System e Evaluation come librerie
from clayrs import content_analyzer as ca
from clayrs import recsys as rs
from clayrs import evaluation as eva

path = 'C:/Users/glamo/Desktop/Repository/RecSys-Algorithms-Evaluation/'
dataset_path = 'C:/Users/glamo/Desktop/Repository/RecSys-Algorithms-Evaluation/Dataset/Movielens 100k/'

# acquisisce dai parametri l'attuale configurazione per la run di test
fields = sys.argv[1].split(',')

# apertura del file con i ratings
ratings = open(dataset_path + 'ratings.csv')

# configurazione del content analyzer
ca_config = ca.ItemAnalyzerConfig(
    source = ca.JSONFile(dataset_path + 'items_info.json'),
    id = 'movielens_id',
    output_directory = path + 'movies_codified/'
)

# rappresentazioni
ca_config.add_multiple_config(
    fields[0],
    [
        ca.FieldConfig(
        ca.WordEmbeddingTechnique(ca.GensimLatentSemanticAnalysis()),
        ca.NLTK(stopwords_removal=True, lemmatization=True),
        id='LSA'
        ),

        # Sentence2Doc-Sbert
        ca.FieldConfig(
            ca.Sentence2DocEmbedding(
                ca.Sbert('paraphrase-distilroberta-base-v1'),
                combining_technique=ca.Centroid()
            ),
            ca.NLTK(stopwords_removal=True, lemmatization=True),
            id='Sentence2Doc-Sbert'
        )
    ]
)

ca.ContentAnalyzer(config = ca_config).fit()

# lista delle rappresentazioni effettuate
representations =  [
    'LSA','Sentence2Doc-Sbert']

# lista di algoritmi per le previsioni
algorithms = ['Centroid Vector','Logistic Regression','Random Forest','SVC',
              'K-NN','Decision Tree','Gaussian Process','Gradient Descent','Ridge Regression']

# carica i ratings per le predizioni
ratings = ca.Ratings(ca.CSVFile(dataset_path + 'ratings.csv'))

current_run = {
    "fields_num" : len(fields),
    "fields" : "",
    "representation" : "",
    "algorithm" : "",
    "methodology" : "",
    "methodologyID" : ""
}

for rep in representations:
    current_run["representation"] = rep

    # Campo Singolo
    if(current_run["fields_num"] == 1):

        current_run["fields"] = fields[0]   # campo in analisi

        # Centroid Vector
        centroid_vector = rs.CentroidVector(
            {fields[0]:[rep]}, 
            similarity = rs.CosineSimilarity()
        )

        current_run['algorithm'] = "Centroid Vector"

        current_run['methodology'] = rs.TestItemsMethodology()
        current_run['methodologyID'] = "Test Items"
        predict(centroid_vector, current_run)

        #current_run['methodology'] = rs.AllItemsMethodology()
        #current_run['methodologyID'] = "All Items"
        #predict(centroid_vector, current_run)

        # Logistic Regression
        logistic_regression = rs.ClassifierRecommender(
            {fields[0]:[rep]}, 
            rs.SkLogisticRegression()
        )
        
        current_run['algorithm'] = "Logistic Regression"

        current_run['methodology'] = rs.TestItemsMethodology()
        current_run['methodologyID'] = "Test Items"
        predict(logistic_regression, current_run)

        current_run['methodology'] = rs.AllItemsMethodology(set(ratings.item_id_column))
        current_run['methodologyID'] = "All Items"
        predict(logistic_regression, current_run)

        # Random Forest
        random_forest = rs.ClassifierRecommender(
            {fields[0]:[rep]}, 
            rs.SkRandomForest(n_estimators = 145)
        )

        current_run['algorithm'] = "Random Forest"

        current_run['methodology'] = rs.TestItemsMethodology()
        current_run['methodologyID'] = "Test Items"
        predict(random_forest, current_run)

        #current_run['methodology'] = rs.AllItemsMethodology()
        #current_run['methodologyID'] = "All Items"
        #predict(random_forest, current_run)

        # SVC
        svc = rs.ClassifierRecommender(
            {fields[0]:[rep]}, 
            rs.SkSVC()
        )

        current_run['algorithm'] = "SVC"

        current_run['methodology'] = rs.TestItemsMethodology()
        current_run['methodologyID'] = "Test Items"
        predict(svc, current_run)

        #current_run['methodology'] = rs.AllItemsMethodology()
        #current_run['methodologyID'] = "All Items"        
        #predict(svc, current_run)

    # Tre Campi
    elif(current_run["fields_num"] == 3):
        
        current_run["fields"] = fields[0] + '+' + fields[1] + '+' + fields[2]

        centroid_vector = rs.CentroidVector(
            {
                fields[0]:[rep], 
                fields[1]:[rep], 
                fields[2]:[rep]
            }, 
            similarity = rs.CosineSimilarity()
        )
        predict(centroid_vector, 'Centroid Vector')

        logistic_regression = rs.ClassifierRecommender(
            {
                fields[0]:[rep], 
                fields[1]:[rep], 
                fields[2]:[rep]
            }, 
            rs.SkLogisticRegression()
        )
        predict(logistic_regression, 'Logistic Regression')

        random_forest = rs.ClassifierRecommender(
            {
                fields[0]:[rep], 
                fields[1]:[rep], 
                fields[2]:[rep]
            }, 
            rs.SkRandomForest(n_estimators = 145)
        )
        predict(random_forest, 'Random Forest')

        svc = rs.ClassifierRecommender(
            {
                fields[0]:[rep], 
                fields[1]:[rep], 
                fields[2]:[rep]
            }, 
            rs.SkSVC()
        )
        predict(svc, 'SVC')

    # Quattro Campi
    elif(current_run["fields_num"] == 4):

        current_run["fields"] = fields[0] + '+' + fields[1] + '+' + fields[2] + '+' + fields[3]

        centroid_vector = rs.CentroidVector(
            {
                fields[0]:[rep], 
                fields[1]:[rep], 
                fields[2]:[rep], 
                fields[3]:[rep]
            }, 
            similarity = rs.CosineSimilarity()
        )
        predict(centroid_vector, 'Centroid Vector')

        logistic_regression = rs.ClassifierRecommender(
            {
                fields[0]:[rep], 
                fields[1]:[rep], 
                fields[2]:[rep], 
                fields[3]:[rep]
            }, 
            rs.SkLogisticRegression()
        )
        predict(logistic_regression, 'Logistic Regression')

        random_forest = rs.ClassifierRecommender(
            {
                fields[0]:[rep], 
                fields[1]:[rep], 
                fields[2]:[rep], 
                fields[3]:[rep]
            }, 
            rs.SkRandomForest(n_estimators = 145)
        )
        predict(random_forest, 'Random Forest')

        svc = rs.ClassifierRecommender(
            {
                fields[0]:[rep], 
                fields[1]:[rep], 
                fields[2]:[rep], 
                fields[3]:[rep]
            }, 
            rs.SkSVC()
        )
        predict(svc, 'SVC')
    