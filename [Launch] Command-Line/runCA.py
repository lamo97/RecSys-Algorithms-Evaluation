# In[]: Imports
from clayrs import content_analyzer as ca
from clayrs import recsys as rs
import sys

fields = sys.argv[1].split(",")
# fields = ['description','genres', 'tags']

path = 'D:/Repository/RecSys-Algorithms-Evaluation/'

# In[]: Scelta del dataset
def config(representation):
    if(sys.argv[2] == '100k'):
        dataset_path = path + 'Dataset/Movielens 100k/'
        
        # configurazione del content analyzer
        ca_config = ca.ItemAnalyzerConfig(
            source = ca.JSONFile(dataset_path + 'items_info.json'),
            id = 'movielens_id',
            output_directory = f'{dataset_path}movies_codified/{sys.argv[1]}/{representation}/'
        )
    else:
        dataset_path =  path + 'Dataset/Movielens 1M/'
        
        # configurazione del content analyzer
        ca_config = ca.ItemAnalyzerConfig(
            source = ca.CSVFile(dataset_path + 'movies-ml1m.csv'),
            id = 'item',
            output_directory = f'{dataset_path}movies_codified/{sys.argv[1]}/{representation}/'
        )

    return ca_config
# In[]: SK-TFIDF
# ca_config = config('SK-TFIDF')

# for field_name in fields:
#    print(f'Now adding: {field_name}')
#    ca_config.add_single_config(
#        field_name,
#        ca.FieldConfig(
#            ca.SkLearnTfIdf(),
#            ca.NLTK(stopwords_removal=True, lemmatization=True),
#            id='SK-TFIDF'
#        )
#    )

#ca.ContentAnalyzer(config = ca_config).fit()

# Word2Vec
ca_config = config('Word2Vec')

for field_name in fields:
    print(f'Now adding: {field_name}')
    ca_config.add_single_config(
        field_name,
        ca.FieldConfig(
            ca.WordEmbeddingTechnique(ca.GensimWord2Vec()),
            ca.NLTK(stopwords_removal=True, lemmatization=True),
            id='Word2Vec'
        )
    ) 

ca.ContentAnalyzer(config = ca_config).fit()

# Doc2Vec
ca_config = config('Doc2Vec')

for field_name in fields:
    print(f'Now adding: {field_name}')
    ca_config.add_single_config(
        field_name,
        ca.FieldConfig(
            ca.WordEmbeddingTechnique(ca.GensimDoc2Vec()),
            ca.NLTK(stopwords_removal=True, lemmatization=True),
            id='Doc2Vec'
        )
    )
 
ca.ContentAnalyzer(config = ca_config).fit()

# GensimLDA
ca_config = config('GensimLDA')

for field_name in fields:
    print(f'Now adding: {field_name}')
    ca_config.add_single_config(
        field_name,
        ca.FieldConfig(
            ca.DocumentEmbeddingTechnique(ca.GensimLDA()),
            ca.NLTK(stopwords_removal=True, lemmatization=True),
            id='GensimLDA'
        )
    )

ca.ContentAnalyzer(config = ca_config).fit()

# GensimRandomIndexing
ca_config = config('GensimRandomIndexing')

for field_name in fields:
    print(f'Now adding: {field_name}')
    ca_config.add_single_config(
        field_name,
        ca.FieldConfig(
            ca.DocumentEmbeddingTechnique(ca.GensimRandomIndexing()),
            ca.NLTK(stopwords_removal=True, lemmatization=True),
            id='GensimRandomIndexing'
        )
    ) 

ca.ContentAnalyzer(config = ca_config).fit()

# GensimFastText
ca_config = config('GensimFastText')

for field_name in fields:
    print(f'Now adding: {field_name}')    
    ca_config.add_single_config(
        field_name,
        ca.FieldConfig(
            ca.WordEmbeddingTechnique(ca.GensimFastText()),
            ca.NLTK(stopwords_removal=True, lemmatization=True),
            id='GensimFastText'
        )
    )

ca.ContentAnalyzer(config = ca_config).fit()

# GensimLSA
ca_config = config('GensimLSA')

for field_name in fields:
    print(f'Now adding: {field_name}')
    ca_config.add_single_config(
        field_name,
        ca.FieldConfig(
            ca.DocumentEmbeddingTechnique(ca.GensimLatentSemanticAnalysis()),
            ca.NLTK(stopwords_removal=True, lemmatization=True),
            id='GensimLSA'
        )
    )
ca.ContentAnalyzer(config = ca_config).fit()

# Word2Doc-GloVe
ca_config = config('Word2Doc-GloVe')

for field_name in fields:
    print(f'Now adding: {field_name}')
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

ca.ContentAnalyzer(config = ca_config).fit()

# Sentence2Doc-Sbert
ca_config = config('Sentence2Doc-Sbert')

for field_name in fields:
    print(f'Now adding: {field_name}')
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

ca.ContentAnalyzer(config = ca_config).fit()