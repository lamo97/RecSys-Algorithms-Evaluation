from operator import index
import pandas as pd
import csv
import os

path = 'D:/Repository/RecSys-Algorithms-Evaluation/'

def check_filename(filename, field, candidate, cutoff):
    filename = filename.split(' - ')
    filename[2] = filename[2].split(' (')
    filename[2] = filename[2][1].replace(").csv","")
    
    return (filename[0] == field and filename[2] == (candidate + '@' + str(cutoff)))

def filename_info(filename):
    filename = filename.split(' - ')
    filename[2] = filename[2].split(' (')
    return filename[2][0] + " + " + filename[1]

def retrieve_tuple(path, filename, metric):
    dataframe = pd.read_csv(path + filename)
    return filename_info(filename), dataframe[metric][1]

def getPrecision(field, dataset):
    dir =  path + (f'Eval Results - {dataset}/SYS/')
    result_path = path + 'Metrics/'

    data = {'ID' : [], 'val' : []}

    candidate_items = ['Test Ratings', 'All Items']

    representations_list =  [
            'SK-TFIDF',
            'Word2Vec', 'Doc2Vec',
            'GensimLDA','GensimRandomIndexing', 'GensimFastText', 'GensimLatentSemanticAnalysis',
            'Word2Doc-GloVe','Sentence2Doc-Sbert']

    metric_prefix = 'Precision@'

    frames = []
    # itera tra le rappresentazioni per la combinazione metrica-campo in considerazione
    for representation in representations_list:
        # itera tra le directories in base alla rappresentazione
        current_dir = dir + representation + '/'
        # itera tra i candidate items
        for candidate in candidate_items:
            # sceglie il cutoff in base al candidate item considerato
            if(candidate == 'Test Ratings'):
                cutoffs = [5,10]
            else:
                cutoffs = [10,20]
            # itera sui due cutoffs scelti
            for cutoff in cutoffs:
                # inserisce cutoff e suffisso all'ID della metrica
                metric = metric_prefix + str(cutoff) + ' - macro'
                # itera tra tutti i file di una data rappresentazione
                for filename in os.listdir(current_dir):
                    # controlla se il nome del file soddisfa il campo in considerazione, candidate e cutoff
                    if(check_filename(filename, field, candidate, cutoff) == True):
                        # prende valore e combinazione algoritmo/rappresentazione
                        id, val = retrieve_tuple(current_dir, filename, metric)
                        # inserisce quanto appena ottenuto in un dizionario
                        data['ID'].append(id)
                        data['val'].append(round(val,5))
                # crea un dataframe dal dizionario
                df = pd.DataFrame.from_dict(data)
                # costruisce il nuovo nome della colonna della metrica
                col_name = metric.split(" - ")[0]
                col_name = col_name + ' | ' + candidate
                # rinomina la colonna relativa alla metrica in esame
                df.rename(columns={'val': col_name}, inplace=True)
                # aggiunge il dataframe alla lista dei dataframe da concatenare
                frames.append(df)
                # svuota il dizionario
                data['ID'] = []
                data['val'] = []

    # concatena tutti i dataframe relativi alla combinazione metrica-campo
    metric_df = pd.concat(frames)

    # raggruppa e somma le righe per id per non avere 4 ripetizioni dello stesso id con colonne vuote alternate
    metric_df = metric_df.groupby('ID').agg({
        'ID': 'first',
        'Precision@5 | Test Ratings': sum,  
        'Precision@10 | Test Ratings': sum,
        'Precision@10 | All Items': sum,  
        'Precision@20 | All Items': sum
    })

    # salva il dataframe in un csv
    metric_df.to_csv(result_path + f'[{dataset}] Precision - {field}.csv', index=False)


###########################################################################################################
def set_prefix(metric):
    if(metric == 'Precision' or metric == 'Recall' or metric == 'F1' or metric == 'NDCG' or metric == 'MRR'):
        return f'{metric}@'
    elif(metric == 'Gini' or metric == 'CatalogCoverage' or metric == 'DeltaGap'):
        return f'{metric} - Top '

def set_suffix(metric, cutoff):
    if(metric == 'Precision@' or metric == 'Recall@' or metric == 'F1@'):
        return f'{metric}{cutoff} - macro'
    elif(metric == 'NDCG@' or metric == 'MRR@' or metric == 'Gini - Top ' or metric == 'CatalogCoverage - Top '):
        return f'{metric}{cutoff}'

def set_column(original_metric, metric, candidate):
    if(original_metric != 'Gini' and original_metric != 'CatalogCoverage'):
        col_name = str(metric).split(" - ")[0]
    else:
        col_name = metric

    return col_name + ' | ' + candidate


def getPrecision(metric, field, dataset):
    dir =  path + (f'Eval Results - {dataset}/SYS/')
    result_path = path + 'Metrics/'

    data = {'ID' : [], 'val' : []}

    candidate_items = ['Test Ratings', 'All Items']

    representations_list =  [
            'SK-TFIDF',
            'Word2Vec', 'Doc2Vec',
            'GensimLDA','GensimRandomIndexing', 'GensimFastText', 'GensimLatentSemanticAnalysis',
            'Word2Doc-GloVe','Sentence2Doc-Sbert']

    or_metric = metric
    metric = set_prefix(metric)
    
    frames = []
    # itera tra le rappresentazioni per la combinazione metrica-campo in considerazione
    for representation in representations_list:
        # itera tra le directories in base alla rappresentazione
        current_dir = dir + representation + '/'
        # itera tra i candidate items
        for candidate in candidate_items:
            # sceglie il cutoff in base al candidate item considerato
            if(candidate == 'Test Ratings'):
                cutoffs = [5,10]
            else:
                cutoffs = [10,20]
            # itera sui due cutoffs scelti
            for cutoff in cutoffs:
                # inserisce cutoff e suffisso all'ID della metrica
                metric = set_prefix(or_metric)
                metric = set_suffix(metric, cutoff)
                # itera tra tutti i file di una data rappresentazione
                for filename in os.listdir(current_dir):
                    # controlla se il nome del file soddisfa il campo in considerazione, candidate e cutoff
                    if(check_filename(filename, field, candidate, cutoff) == True):
                        # prende valore e combinazione algoritmo/rappresentazione
                        id, val = retrieve_tuple(current_dir, filename, metric)
                        # inserisce quanto appena ottenuto in un dizionario
                        data['ID'].append(id)
                        data['val'].append(round(val,5))

                # crea un dataframe dal dizionario
                df = pd.DataFrame.from_dict(data)
                # costruisce il nuovo nome della colonna della metrica
                col_name = set_column(or_metric, metric, candidate)
                # rinomina la colonna relativa alla metrica in esame
                df.rename(columns={'val': col_name}, inplace=True)
                # aggiunge il dataframe alla lista dei dataframe da concatenare
                frames.append(df)
                # svuota il dizionario
                data['ID'] = []
                data['val'] = []
                

    # concatena tutti i dataframe relativi alla combinazione metrica-campo
    metric_df = pd.concat(frames)

    # raggruppa e somma le righe per id per non avere 4 ripetizioni dello stesso id con colonne vuote alternate
    metric_df = metric_df.groupby('ID').agg({
        'ID': 'first',
        f'{set_prefix(or_metric)}5 | Test Ratings': sum,  
        f'{set_prefix(or_metric)}10 | Test Ratings': sum,
        f'{set_prefix(or_metric)}10 | All Items': sum,  
        f'{set_prefix(or_metric)}20 | All Items': sum
    })

    # salva il dataframe in un csv
    metric_df.to_csv(result_path + f'[{dataset}] {or_metric} - {field}.csv', index=False)

#getPrecision('plot', '100k')
getPrecision('Recall','plot', '100k')
getPrecision('F1','plot', '100k')
getPrecision('NDCG','plot', '100k')
getPrecision('Gini','genres', '100k')