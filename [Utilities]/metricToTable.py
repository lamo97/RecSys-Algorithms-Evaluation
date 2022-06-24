import pandas as pd
import os
import dataframe_image as dfi

path = 'D:/Repository/RecSys-Algorithms-Evaluation/'
dir = f'{path}/Eval Results - 1M/SYS/'
result_path = f'{path}/Metrics/'

def getFileInfo(filename):
    file_info = {'fields' : '', 'representation': '', 'algorithm': '', 'candidate': '', 'cutoff': 0}
    
    filename = filename.split(' - ')
    filename[2] = filename[2].split(' (')
    filename[2][1] = filename[2][1].replace(').csv','')
    filename[2][1] = filename[2][1].split('@')
    
    # es. 'description - Doc2Vec - Centroid Vector (All Items@10).csv' dopo gli split e replace diventa:
    # ['description', 'Doc2Vec', ['Centroid Vector', ['All Items', '10']]]
    
    file_info['fields'] = filename[0]
    file_info['representation'] = filename[1]
    file_info['algorithm'] = filename[2][0]
    file_info['candidate'] = filename[2][1][0] # 'All Items'
    file_info['cutoff'] = int(filename[2][1][1]) # 10
    
    return file_info

def checkFilename(filename, run):
    file_info = getFileInfo(filename)
    
    return(
        file_info['fields'] == run['fields'] and 
        file_info['candidate'] == run['candidate'] and
        file_info['cutoff'] == run['cutoff']
    )

def setPrefix(metric):
    if(metric == 'Precision' or metric == 'Recall' or metric == 'F1' or metric == 'NDCG' or metric == 'MRR'):
        return f'{metric}@'
    elif(metric == 'Gini' or metric == 'CatalogCoverage' or metric == 'DeltaGap'):
        return f'{metric} - Top '

def setSuffix(metric, cutoff):
    if(metric == 'Precision@' or metric == 'Recall@' or metric == 'F1@'):
        return f'{metric}{cutoff} - macro'
    elif(metric == 'NDCG@' or metric == 'MRR@' or metric == 'Gini - Top ' or metric == 'CatalogCoverage - Top '):
        return f'{metric}{cutoff}'

def getMetricValue(current_dir, filename, metric_string, candidate):
    dataframe = pd.read_csv(current_dir + filename)
    
    file_info = getFileInfo(filename)
    ID = f'{file_info["algorithm"]} + {file_info["representation"]}'

    return ID, dataframe[metric_string][1]

def shortenColumn(run):
    if(run['metric'] == 'Catalog Coverage'):
        metric = 'Coverage'
    else:
        metric = run['metric']

    if (run["candidate"] == 'Test Ratings'):
        candidate = '(TR)'
    elif (run["candidate"] == 'All Items'):
        candidate = '(AI)'
    return f'{run["metric"]}@{run["cutoff"]} {candidate}'

candidate_items = ['Test Ratings', 'All Items']

representations_list =  [
            'SK-TFIDF',
            'Word2Vec', 'Doc2Vec',
            'GensimLDA','GensimRandomIndexing', 'GensimFastText', 'GensimLSA',
            'Word2Doc-GloVe','Sentence2Doc-Sbert']

def metricToTable(metric, field):
    run = {'fields' : field, 'metric': metric, 'candidate': '', 'cutoff': []}
    data = {'ID':[]}
    frames = []
    metric_string = ''
    
    # itera tra le rappresentazioni per la combinazione metrica-campo in considerazione
    for representation in representations_list:
        # itera tra le directories in base alla rappresentazione
        current_dir = f'{dir}{representation}/'
        # itera tra i candidate items
        for candidate in candidate_items:
            run['candidate'] = candidate
            # sceglie il cutoff in base al candidate item considerato
            if(run['candidate'] == 'Test Ratings'):
                cutoffs = [5,10]
            else:
                cutoffs = [10,20]
            # itera sui due cutoffs scelti
            for cutoff in cutoffs:
                run['cutoff'] = cutoff
                # inserisce cutoff e suffisso all'ID della metrica
                metric_string = setPrefix(run['metric'])
                metric_string = setSuffix(metric_string, run['cutoff'])
                # itera tra tutti i file di una data rappresentazione
                for filename in os.listdir(current_dir):
                    # controlla se il nome del file soddisfa il campo in considerazione, candidate e cutoff
                    if(checkFilename(filename, run)):
                    # prende valore e combinazione algoritmo/rappresentazione
                        id, val = getMetricValue(current_dir, filename, metric_string, run['candidate'])
                        column_name = shortenColumn(run)
                        
                        data['ID'].append(id)
                        
                        if column_name in data:
                            data[column_name].append(round(val,4))
                        else:
                            data[column_name] = []
                            data[column_name].append(round(val,4))
                
                # crea un dataframe dal dizionario
                df = pd.DataFrame.from_dict(data)
                # aggiunge il dataframe alla lista dei dataframe da concatenare
                frames.append(df)
                # svuota il dizionario
                data = {'ID':[]}
                
                
    metric_df = pd.concat(frames)

    metric_df = metric_df.groupby('ID').agg({
        'ID': 'first',
        f'{metric}@5 (TR)': sum,  
        f'{metric}@10 (TR)': sum,
        f'{metric}@10 (AI)': sum,  
        f'{metric}@20 (AI)': sum
    })
    
    df_styled = metric_df.style.background_gradient()
    df_styled.set_precision(3)
    df_styled.hide_index()
    
    output_name = f'{result_path}[1M] {run["metric"]} - {run["fields"]}.png'
    dfi.export(df_styled, output_name)

    
metrics = ['Precision', 'Recall', 'MRR', 'NDCG', 'Gini', 'CatalogCoverage', 'F1']

for metric in metrics:
    metricToTable(metric, 'description')