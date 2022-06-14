# In[0]
import DatasetConversion as dc

dc.convertToCsv()

# In[1]:
# correzione dell'ordine di stampa
import functools
from operator import rshift
print = functools.partial(print, flush=True)

# import dei moduli per Content Analyzer, Recommender System e Evaluation come librerie
from clayrs import content_analyzer as ca
from clayrs import recsys as rs
from clayrs import evaluation as eva

def predict(recommender, current_setup):
    train_list, test_list = rs.KFoldPartitioning(n_splits = 2).split_all(ratings)                         
    first_training_set = train_list[0]

    cbrs = rs.ContentBasedRS(recommender, first_training_set, (path + '/movies_codified'))
    first_test_set = test_list[0]
    rank = cbrs.fit_rank(first_test_set, user_id_list = ['8', '2', '1'], n_recs = 3)

    result_list = []
    for train_set, test_set in zip(train_list, test_list):
        cbrs = rs.ContentBasedRS(recommender, train_set, (path + '/movies_codified'))
        rank_to_append = cbrs.fit_rank(test_set)
        result_list.append(rank_to_append)

    for result in result_list:
        print(result)

    evaluation(result_list, test_list, current_setup)
    
def evaluation(result_list, test_list, current_setup):
    em = eva.EvalModel(
        pred_list=result_list,
        truth_list=test_list,
        metric_list=[
            eva.NDCG(),
            eva.Precision(),
            eva.RecallAtK(k=5)
        ]
    )

    sys_result, users_result =  em.fit()

    results_path = 'C:/Users/glamo/Desktop/Repository/RecSys-Algorithms-Evaluation/Eval Results/'

    sys_result.to_csv(results_path + 'sys_results_' + current_setup[0] + '_' + current_setup[1] + '.csv')
    users_result.to_csv(results_path + 'user_results_' + current_setup[0] + '_' + current_setup[1] + '.csv')

def listRepresentations():
    print('Rappresentazioni:')
    for i in range(0, len(representations)):
        print(i+1, ': ', representations[i])

def listAlgorithms():
    print("Algoritmi: ")
    print('0: EXIT')
    for i in range(0, len(algorithms)):
        print(i+1, ': ', algorithms[i])

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
fields = ['plot']

# lista delle rappresentazioni effettuate
representations =  [
    'SK-TFIDF', 'Whoosh-TFIDF',
    'Word2Vec', 'Doc2Vec',
    'GensimLDA','GensimRandomIndexing', 'GensimFastText', 'GensimLatentSemanticAnalysis',
    'Word2Doc-GloVe','Sentence2Doc-Sbert']

# lista di algoritmi per le previsioni
algorithms = ['Centroid Vector','Logistic Regression','Random Forest','SVC',
              'K-NN','Decision Tree','Gaussian Process','Gradient Descent','Ridge Regression']

# rappresentazioni
ca_config.add_multiple_config(
    fields[0],
    [
        # SK-TFIDF 
        ca.FieldConfig(
            ca.SkLearnTfIdf(),
            ca.NLTK(stopwords_removal=True, lemmatization=True),
            id='SK-TFIDF'
        ),

        # Whoosh-TFIDF 
        ca.FieldConfig(
            ca.WhooshTfIdf(),
            ca.NLTK(stopwords_removal=True, lemmatization=True),
            id='Whoosh-TFIDF'
        ),

        # Word2Vec 
        ca.FieldConfig(
            ca.WordEmbeddingTechnique(ca.GensimWord2Vec()),
            ca.NLTK(stopwords_removal=True, lemmatization=True),
            id='Word2Vec'
        ),
        
        # Doc2Vec 
        ca.FieldConfig(
            ca.DocumentEmbeddingTechnique(ca.GensimDoc2Vec()),
            ca.NLTK(stopwords_removal=True, lemmatization=True),
            id='Doc2Vec'
        ),

        # GensimLDA 
        ca.FieldConfig(
            # Probabilmente LDA classifica DOCUMENTI interni e non parole (tenere conto anche del campo)
            ca.SentenceEmbeddingTechnique(ca.GensimLDA()),
            ca.NLTK(stopwords_removal=True, lemmatization=True),
            id='GensimLDA'
        ),

        # GensimRandomIndexing
        ca.FieldConfig(
            ca.DocumentEmbeddingTechnique(ca.GensimRandomIndexing()),
            ca.NLTK(stopwords_removal=True, lemmatization=True),
            id='GensimRandomIndexing'
        ),

        # GensimFastText
        ca.FieldConfig(
            ca.WordEmbeddingTechnique(ca.GensimFastText()),
            ca.NLTK(stopwords_removal=True, lemmatization=True),
            id='GensimFastText'
        ),

        # GensimLatentSemanticAnalysis
        ca.FieldConfig(
            ca.WordEmbeddingTechnique(ca.GensimLatentSemanticAnalysis()),
            ca.NLTK(stopwords_removal=True, lemmatization=True),
            id='GensimLatentSemanticAnalysis'
        ),

        # Word2Doc-GloVe
        ca.FieldConfig(
            ca.Word2DocEmbedding(
                ca.Gensim('glove-twitter-50'),
                combining_technique=ca.Centroid()
            ),
            ca.NLTK(stopwords_removal=True, lemmatization=True),
            id='Word2Doc-GloVe'
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
    

# %%
ratings = ca.Ratings(ca.CSVFile(path + 'ratings.csv'))

while(True):
    listRepresentations()
    choice = int(input('Su quale rappresentazione si desidera effettuare la previsione?: '))
    if(choice == 0): break

    listAlgorithms()
    alg_choice = int(input('Con quale algoritmo si desidera effettuare la predizione?: '))

    current_setup = [representations[choice-1], algorithms[alg_choice]]

    if(alg_choice == 1):
        centroid_vector = rs.CentroidVector({fields[0]: [representations[choice-1]]}, similarity = rs.CosineSimilarity())
        predict(centroid_vector, current_setup)

    elif(alg_choice == 2):
        logistic_regression = rs.ClassifierRecommender(
            {fields[0]: [representations[choice-1]]}, rs.SkLogisticRegression())
        predict(logistic_regression, current_setup)

    elif(alg_choice == 3):
        random_forest = rs.ClassifierRecommender({fields[0]: [representations[choice-1]]}, rs.SkRandomForest(n_estimators = 145))
        predict(random_forest, current_setup)

    elif(alg_choice == 4):
        svc = rs.ClassifierRecommender({fields[0]: [representations[choice-1]]}, rs.SkSVC())
        predict(svc, current_setup)

    elif(alg_choice == 5):
        knn = rs.ClassifierRecommender({fields[0]: [representations[choice-1]]}, rs.SkKNN())
        predict(knn, current_setup)

    elif(alg_choice == 6):
        decision_tree = rs.ClassifierRecommender({fields[0]: [representations[choice-1]]}, rs.SkDecisionTree())
        predict(decision_tree, current_setup)

    elif(alg_choice == 7):
        gaussian_process = rs.ClassifierRecommender({fields[0]: [representations[choice-1]]}, rs.SkGaussianProcess())
        predict(gaussian_process, current_setup)

    elif(alg_choice == 8):
        gradient_descent = rs.LinearPredictor({fields[0]: [representations[choice-1]]}, rs.SkSGDRegressor())
        predict(gradient_descent, current_setup)

    elif(alg_choice == 9):
        ridge_reg = rs.LinearPredictor({fields[0]: [representations[choice-1]]}, rs.SkRidge())
        predict(ridge_reg, current_setup)

# %%
