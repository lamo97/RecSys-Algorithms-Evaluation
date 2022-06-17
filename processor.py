
import pandas as pd
import csv
import os

# directory dei risultati: Eval Results
# sottodirectory per rappresentazione: TF-IDF, Word2Vec...
# Nomi dei File: Campo - Rappresentazione - Algoritmo (Candidate Item@Cutoff)

# Tabelle Riassuntive:
    # Titolo:   Campo + Metrica
    # Colonne:  TestItems: Precision@5 | TestItems: Precision@5 | AllItems: Precision@10 | AllItems: Precision@20
    #           TestItems: Recall@5 | TestItems: Recall@5 | AllItems: Recalln@10 | AllItems: Recall@20
    #           TestItems: F15@5 | TestItems: F15@5 | AllItems: F15@10 | AllItems: F15@20
    #           TestItems: NDCG@5 | TestItems: NDCG@5 | AllItems: NDCG@10 | AllItems: NDCG@20
    #           TestItems: MRR@5 | TestItems: MRR@5 | AllItems: MRR@10 | AllItems: MRR@20
    #           TestItems: Gini | TestItems: CatalogCoverage | AllItems: Gini | AllItems: CatalogCoverage
    #           TestItems: DeltaGAP - Niche | TestItems: DeltaGAP - Diverse | TestItems: DeltaGAP - Blockbuster Focused |  AllItems: DeltaGAP - Niche | AllItems: DeltaGAP - Diverse | AllItems: DeltaGAP - Blockbuster Focused
    #           Test statistici (2) 
    # Righe:    Algoritmi di predizione + Rappresentazione

    # Es.       File: Description - Precision
    #           Colonne: ID, TestItems: Precision@5 | TestItems: Precision@5 | AllItems: Precision@10 | AllItems: Precision@20
    #           Righe: Centroid Vector, Logistic Regression, Random Forests, SVC           

def retrieveInfo(filename, metric):
    dir = 'C:/Users/glamo/Desktop/Repository/RecSys-Algorithms-Evaluation/Eval Results/'

    df = pd.read_csv(dir + filename)
    print(round(df[metric][2], 4))
    
#retrieveInfo('SYS - plot - Doc2Vec - SVC (All Items@10).csv','Precision@10 - macro')

fields = [
    'description',
    'genres',
    'tags',
    'reviews',
    'description, genres, tags',
    'description, genres, reviews',
    'description, tags, reviews',
    'genres, tags, reviews',
    'description, genres, tags, reviews'
]

metricsID = [
    ['Precision@5 - Macro','Precision@10 - Macro', 'Precision@10 - Macro', 'Precision@20 - Macro'],
    ['Recall@5 - Macro', 'Recall@10 - Macro', 'Recall@10 - Macro', 'Recall@20 - Macro'],
    ['F1@5 - Macro', 'F1@10 - Macro', 'F1@10 - Macro', 'F1@20 - Macro'],
    ['nDCG@5 - Macro', 'nDCG@10 - Macro', 'nDCG@10 - Macro', 'nDCG@20 - Macro'],
    ['MRR@5 - Macro', 'MRR@10 - Macro', 'MRR@10 - Macro', 'MRR@20 - Macro'],
    ['Gini', 'Catalog Coverage'],
    ['Delta GAP - Niche', 'DeltaGAP - Diverse', 'DeltaGAP - Blockbuster Focused']
    #['T-Test','Wilcoxon']
]

representations_list =  [
        'SK-TFIDF',
        'Word2Vec', 'Doc2Vec',
        'GensimLDA','GensimRandomIndexing', 'GensimFastText', 'GensimLatentSemanticAnalysis',
        'Word2Doc-GloVe','Sentence2Doc-Sbert']

metrics = ['Precision', 'Recall', 'F-15', 'nDCG', 'Gini', 'Catalog Coverage', 'DeltaGAP']

dir = 'C:/Users/glamo/Desktop/Repository/RecSys-Algorithms-Evaluation/Eval Results - 100k/'

candidate_items = ['Test Items','All Items']
cutoffs = []

def checkFilename(field, candidate, cutoff, filename):
    fn = filename.split(" - ")
    # ottiene la combinazione candidate@cutoff
    fn[2] = fn[2].split("(")[1].replace(")", "")
    
    if(fn[0] == field and fn[2] == (candidate + '@' + str(cutoff))):
        return True
    else:
        return False

def getValue(path, filename, metric):
    # divide in nomde del file
    fn = filename.split(" - ")
    # combina rappresentazione e algoritmo
    id = fn[1] + " + " + fn[2].split("(")[0]

    df = pd.read_csv(path + filename)
    print(df[str(metric)])
    value = df[metric][1]

    return id,value

index = 0

for metric_name in metrics:
    for field in fields:
        df = pd.DataFrame()
        df.insert(0, 'id', value=None)
        for metricID in metricsID[index]:
            df.insert(index+1, metricID, value=None)
            for representation in representations_list:
                path = (dir + representation + '/')
                for candiate in candidate_items:
                    if(candiate == 'Test Items'):
                        cutoffs = [5,10]
                    else:
                        cutoffs = [10,20]
                    for cutoff in cutoffs:
                        for filename in os.listdir(path):
                            if(checkFilename(field, candiate, cutoff, filename) == True):
                                id, value = getValue(path, filename, metricID)
                                df['id'] = id
                                df[metricID] = value

                name = metricID.replace(" - macro", "")
                name = name + ' | ' + candiate
                df.rename(columns={metricID : (name)})
                print(df)
        index += 1          
index = 0



