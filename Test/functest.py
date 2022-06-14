#In[]:
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
representation = 'Word2Vec'

# inserimento del campo rappresentato nella lista
fields.append('plot')



# In[]:
val = ca.Gensim()
print('Valore: ', val, '\tTipo: ', type(val))

val = ca.GensimDoc2Vec()
print('Valore: ', val, '\tTipo: ', type(val))


val[0] = ca.Sentence2DocEmbedding(ca.GensimLatentSemanticAnalysis(), combining_technique=ca.Centroid())
print('Valore: ', val, '\tTipo: ', type(val))

#val = ca.Sentence2DocEmbedding(ca.GensimFastText(), combining_technique=ca.Centroid())
#print('Valore: ', val, '\tTipo: ', type(val))



# %%

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

ca_config.add_single_config(
    'plot', 
    ca.FieldConfig(
        ca.WordEmbeddingTechnique(ca.GensimLatentSemanticAnalysis()),
        ca.NLTK(stopwords_removal=True, lemmatization=True),
        id='LSA'
    )
)

ca.ContentAnalyzer(config = ca_config).fit()

ratings = ca.Ratings(ca.CSVFile(path + 'ratings.csv'))

centroid_vector = rs.CentroidVector({'plot':'LSA'}, similarity = rs.CosineSimilarity())

train_list, test_list = rs.KFoldPartitioning(n_splits = 2).split_all(ratings)                         
first_training_set = train_list[0]

cbrs = rs.ContentBasedRS(centroid_vector, first_training_set, (path + '/movies_codified'))
first_test_set = test_list[0]
rank = cbrs.fit_rank(first_test_set, user_id_list = ['8', '2', '1'], n_recs = 3)

result_list = []
for train_set, test_set in zip(train_list, test_list):
    cbrs = rs.ContentBasedRS(centroid_vector, train_set, (path + '/movies_codified'))
    rank_to_append = cbrs.fit_rank(test_set, methodology=rs.AllItemsMethodology(items['movielens_id']))
    result_list.append(rank_to_append)

for result in result_list:
    print(result)


# In[]
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
    output_directory = path + 'movies_codified/LDA',
    export_json=True
)

ca_config.add_single_config(
    'plot', 
    ca.FieldConfig(
        #ca.WordEmbeddingTechnique(ca.GensimLatentSemanticAnalysis()),
        ca.WordEmbeddingTechnique(ca.Gensim('content.json')),
        ca.NLTK(stopwords_removal=True, lemmatization=True),
        id='LSA'
    )
)

ca.ContentAnalyzer(config = ca_config).fit()

# %%

current_run = {
    "fields_num" : 12,
    "fields" : "",
    "representation" : "",
    "algorithm" : "",
    "methodology" : "",
    "methodologyID" : ""
}

print(current_run["fields_num"])
