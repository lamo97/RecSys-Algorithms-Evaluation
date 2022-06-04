import pandas as pd
import csv
import re

# correzione dell'ordine di stampa
import functools
from operator import rshift
print = functools.partial(print, flush=True)

# import dei moduli per Content Analyzer, Recommender System e Evaluation come librerie
from clayrs import content_analyzer as ca
from clayrs import recsys as rs
from clayrs import evaluation as eva

# elaborazione del file .dat relativo agli items
path = 'C:/Users/glamo/Desktop/Repository/RecSys-Algorithms-Evaluation/Dataset/'

moviesCsv = open(path + 'movies.csv', 'w')
moviesCsv.write("id,title,year,genres\n")

with open(path + 'movies.dat', 'r') as moviesDat:
    for entry in moviesDat:
        entry = entry.split("::")
        moviesCsv.write(entry[0] + ',')
        year = re.findall('\d\d\d\d', entry[1])
        moviesCsv.write('"' + re.split(' \(\d\d\d\d\)', entry[1])[0] + '"' + ',')
        moviesCsv.write(year[0] + ',')
        entry[2] = re.sub('\|', ',', entry[2])
        entry[2] = re.sub('\n', '', entry[2])
        moviesCsv.write('"' + entry[2] + '"' + '\n')
        
moviesDat.close()

# elaborazione del file .dat relativo agli utenti
usersCsv = open(path + 'users.csv', 'w')
usersCsv.write('user_id,gender,age,occupation,zip_code\n')

with open(path + 'users.dat', 'r') as usersDat:
    for entry in usersDat:
        entry = entry.split("::")
        usersCsv.write(entry[0] + ',')
        usersCsv.write(entry[1] + ',')
        usersCsv.write(entry[2] + ',')
        usersCsv.write(entry[3] + ',')
        usersCsv.write(entry[4])
        
usersDat.close()

# elaborazione del file .dat relativo ai ratings
ratingsCsv = open(path + 'ratings.csv', 'w')
ratingsCsv.write("user_id,movie_id,rating,timestamp\n")

with open(path + 'ratings.dat', 'r') as ratingsDat:
    for entry in ratingsDat:
        entry = entry.split("::")
        ratingsCsv.write(entry[0] + ',')
        ratingsCsv.write(entry[1] + ',')
        ratingsCsv.write(entry[2] + ',')
        ratingsCsv.write(entry[3])
        
ratingsDat.close()
ratingsCsv.close()

# rappresentazione items con TF-IDF
movies_ca_config = ca.ItemAnalyzerConfig(
    source = ca.CSVFile(path + 'movies.csv'),
    id = 'id',
    output_directory = 'C:/Users/glamo/Desktop/Repository/RecSys-Algorithms-Evaluation/movies_codified/'
)

movies_ca_config.add_single_config(
    'plot',
    ca.FieldConfig(ca.SkLearnTfIdf(),
    preprocessing = ca.NLTK(stopwords_removal=True, lemmatization=True),
    id = 'tfidf'              
    )
)

mv = pd.read_csv(path + 'movies.csv', encoding='latin1')
ca.ContentAnalyzer(config = movies_ca_config).fit()

