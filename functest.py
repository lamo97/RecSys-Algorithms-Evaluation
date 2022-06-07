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

value = ca.GensimLatentSemanticAnalysis()
print(type(value))

value = ca.SkLearnTfIdf()
print(type(value))

value = ca.WordEmbeddingTechnique(ca.GensimLatentSemanticAnalysis())
print(type(value))

value = ca.Gensim('glove-twitter-50')
print(type(value))

value = ca.Word2DocEmbedding(ca.SkLearnTfIdf(), combining_technique=ca.Centroid())
print(type(value))

value = ca.Sentence2DocEmbedding(ca.Gensim(), combining_technique=ca.Centroid())
print(type(value))

value = ca.Sentence2WordEmbedding(ca.Gensim())
print(type(value))

value = ca.GensimRandomIndexing()
print(type(value))

value = ca.GensimFastText()
print(type(value))

value = ca.GensimDoc2Vec()
print(type(value))

value = ca.WhooshTfIdf()
print(type(value))

value = ca.Sbert()
print(type(value))

value = ca.SentenceEmbeddingTechnique(ca.Gensim())
print(type(value))

# value = ca.DBPediaMappingTechnique(ca.GensimLatentSemanticAnalysis(), 'plot')
# print(type(value))

value = ca.GensimLDA()
print(type(value))

value = ca.Word2DocEmbedding(ca.GensimFastText(), combining_technique='None')
print(type(value))

# In[]:
ca_config.add_single_config(
    'plot',
    ca.FieldConfig(
        ca.GensimWord2Vec('glove-twitter-50', combining_technique=ca.Centroid())
    ),
    ca.NLTK(stopwords_removal=True, lemmatization=True),
    id = 'Word2Vec'
)

# In[]:

ca_config.add_multiple_config(
    'plot'
    [
        ca.FieldConfig(
            ca.latent_semantic_analysis(
                'glove-twitter-50',
                combining_technique=ca.Centroid()
            ),
        ca.NLTK(stopwords_removal=True, lemmatization=True),
        id = 'Word2Vec'
        )
    ],
    [
        ca.FieldConfig(
            ca.GensimWord2Vec(
                'glove-twitter-50',
                combining_technique=ca.Centroid()
            ),
        ca.NLTK(stopwords_removal=True, lemmatization=True),
        id = 'Word2Vec'
        )
    ]
)

# serializzazione degli item
ca.ContentAnalyzer(config = ca_config).fit()

# inserimento del campo rappresentato nella lista
fields.append('plot')

# %%
