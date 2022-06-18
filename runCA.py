# In[]: Imports
from clayrs import content_analyzer as ca
from clayrs import recsys as rs
import sys

path = 'D:/Repository/RecSys-Algorithms-Evaluation/'

# fields = []                           # nel caso si volesse lanciare lo script da codice
fields = sys.argv[1].split(",")         # lancio con args

# In[]: Scelta del dataset

if(sys.argv[2] == '100k'):
    print('\nDataset: Movielens 100k')
    dataset_path = path + 'Dataset/Movielens 100k/'
    
    # configurazione del content analyzer
    ca_config = ca.ItemAnalyzerConfig(
        source = ca.JSONFile(dataset_path + 'items_info.json'),
        id = 'movielens_id',
        output_directory = dataset_path + 'movies_codified/'
    )
else:
    print('\nDataset: Movielens 1M')
    dataset_path =  path + 'Dataset/Movielens 1M/'
    
    # configurazione del content analyzer
    ca_config = ca.ItemAnalyzerConfig(
        source = ca.CSVFile(dataset_path + 'movies-ml1m.csv'),
        id = 'item',
        output_directory = dataset_path + 'movies_codified/'
    )

# In[]: SK-TFIDF
for field_name in fields:
    ca_config.add_single_config(
        field_name,
        ca.FieldConfig(
            ca.SkLearnTfIdf(),
            ca.NLTK(stopwords_removal=True, lemmatization=True),
            id='SK-TFIDF'
        )
    )

# In[]: Word2Vec 
for field_name in fields:
    ca_config.add_single_config(
        field_name,
        ca.FieldConfig(
            ca.WordEmbeddingTechnique(ca.GensimWord2Vec()),
            ca.NLTK(stopwords_removal=True, lemmatization=True),
            id='Word2Vec'
        )
    ) 

# In[]: Doc2Vec
for field_name in fields:
    ca_config.add_single_config(
        field_name,
        ca.FieldConfig(
            ca.WordEmbeddingTechnique(ca.GensimDoc2Vec()),
            ca.NLTK(stopwords_removal=True, lemmatization=True),
            id='Doc2Vec'
        )
    ) 
    
# In[]: GensimLDA 
for field_name in fields:
    ca_config.add_single_config(
        field_name,
        ca.FieldConfig(
            ca.DocumentEmbeddingTechnique(ca.GensimLDA()),
            ca.NLTK(stopwords_removal=True, lemmatization=True),
            id='GensimLDA'
        )
    ) 

# In[]: GensimRandomIndexing
for field_name in fields:
    ca_config.add_single_config(
        field_name,
        ca.FieldConfig(
            ca.DocumentEmbeddingTechnique(ca.GensimRandomIndexing()),
            ca.NLTK(stopwords_removal=True, lemmatization=True),
            id='GensimRandomIndexing'
        )
    ) 

# In[]: GensimFastText
for field_name in fields:
    ca_config.add_single_config(
        field_name,
        ca.FieldConfig(
            ca.WordEmbeddingTechnique(ca.GensimFastText()),
            ca.NLTK(stopwords_removal=True, lemmatization=True),
            id='GensimFastText'
        )
    )

# In[]: GensimLatentSemanticAnalysis
for field_name in fields:
    ca_config.add_single_config(
        field_name,
        ca.FieldConfig(
            ca.DocumentEmbeddingTechnique(ca.GensimLatentSemanticAnalysis()),
            ca.NLTK(stopwords_removal=True, lemmatization=True),
            id='GensimLatentSemanticAnalysis'
        )
    )

# In[]: Word2Doc-GloVe
for field_name in fields:
    ca_config.add_single_config(
        field_name,
        ca.FieldConfig(
            ca.Word2DocEmbedding(
                ca.Gensim('glove-twitter-50'),
                combining_technique=ca.Centroid()
            ),
            ca.NLTK(stopwords_removal=True, lemmatization=True),
            id='Word2Doc-GloVe'
        )
    ) 

# In[]: Sentence2Doc-Sbert
for field_name in fields:
    ca_config.add_single_config(
        field_name,
        ca.FieldConfig(
            ca.Sentence2DocEmbedding(
                ca.Sbert('paraphrase-distilroberta-base-v1'),
                combining_technique=ca.Centroid()
            ),
            ca.NLTK(stopwords_removal=True, lemmatization=True),
            id='Sentence2Doc-Sbert'
        )
    )


# In[]: Serializzazione degli item delle rappresentazioni selezionate
ca.ContentAnalyzer(config = ca_config).fit()
