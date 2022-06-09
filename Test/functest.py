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


val = ca.Sentence2DocEmbedding(ca.GensimLatentSemanticAnalysis(), combining_technique=ca.Centroid())
print('Valore: ', val, '\tTipo: ', type(val))

#val = ca.Sentence2DocEmbedding(ca.GensimFastText(), combining_technique=ca.Centroid())
#print('Valore: ', val, '\tTipo: ', type(val))



# %%
