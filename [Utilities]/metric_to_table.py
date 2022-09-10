import pandas as pd
import os
import dataframe_image as dfi

PATH = 'D:/Repository/RecSys-Algorithms-Evaluation/'
DIR = f'{PATH}/Eval Results - 1M/SYS/'
RESULT_PATH = f'{PATH}/Metrics/'

def get_file_info(filename):
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

def check_filename(filename, run):
    file_info = get_file_info(filename)
    
    return(
        file_info['fields'] == run['fields'] and 
        file_info['candidate'] == run['candidate'] and
        file_info['cutoff'] == run['cutoff']
    )

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

def get_metric_value(current_dir, filename, metric_string, candidate):
    dataframe = pd.read_csv(current_dir + filename)
    
    file_info = get_file_info(filename)
    ID = f'{file_info["algorithm"]} + {file_info["representation"]}'

    return ID, dataframe[metric_string][1]

def shorten_column(run):
    if(run['metric'] == 'Catalog Coverage'):
        metric = 'Coverage'
    else:
        metric = run['metric']

    if (run["candidate"] == 'Test Ratings'):
        candidate = '(TR)'
    elif (run["candidate"] == 'All Items'):
        candidate = '(AI)'
    return f'{run["metric"]}@{run["cutoff"]} {candidate}'

def getDeltaValue(current_dir, filename, metric_string, candidate):
    dataframe = pd.read_csv(current_dir + filename)
    
    file_info = get_file_info(filename)
    ID = f'{file_info["algorithm"]} + {file_info["representation"]}'

    return ID, dataframe[metric_string][1]


candidate_items = ['Test Ratings', 'All Items']

representations_list =  [
            'SK-TFIDF',
            'Word2Vec', 'Doc2Vec',
            'GensimLDA','GensimRandomIndexing', 'GensimFastText', 'GensimLSA',
            'Word2Doc-GloVe','Sentence2Doc-Sbert']

def metric_to_table(metric, field):
    run = {'fields' : field, 'metric': metric, 'candidate': '', 'cutoff': []}
    data = {'ID':[]}
    frames = []
    metric_string = ''
    
    # itera tra le rappresentazioni per la combinazione metrica-campo in considerazione
    for representation in representations_list:
        # itera tra le DIRectories in base alla rappresentazione
        current_dir = f'{DIR}{representation}/'
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
                metric_string = set_prefix(run['metric'])
                metric_string = set_suffix(metric_string, run['cutoff'])
                # itera tra tutti i file di una data rappresentazione
                for filename in os.listdir(current_dir):
                    # controlla se il nome del file soddisfa il campo in considerazione, candidate e cutoff
                    if(check_filename(filename, run)):
                    # prende valore e combinazione algoritmo/rappresentazione
                        id, val = get_metric_value(current_dir, filename, metric_string, run['candidate'])
                        column_name = shorten_column(run)
                        
                        #inversione ID
                        id = id.split("+")
                        id_inv = f'{id[1]}+{id[0]}'
                        
                        data['ID'].append(id_inv)
                        #data['ID'].append(id)

                        if run['metric'] == 'CatalogCoverage':
                            rounding = 1
                        else:
                            rounding = 3

                        if column_name in data:
                            data[column_name].append(round(val,rounding))
                        else:
                            data[column_name] = []
                            data[column_name].append(round(val,rounding))
                
                # crea un dataframe dal dizionario
                df = pd.DataFrame.from_dict(data)
                # aggiunge il dataframe alla lista dei dataframe da concatenare
                frames.append(df)
                # svuota il dizionario
                data = {'ID':[]}
                
                
    metric_df = pd.concat(frames)

    # ragrauppamento righe con stesso ID
    metric_df = metric_df.groupby('ID').agg({
        'ID': 'first',
        f'{metric}@5 (TR)': sum,  
        f'{metric}@10 (TR)': sum
        #f'{metric}@10 (AI)': sum,  
        #f'{metric}@20 (AI)': sum
    })
    
    metric_df.rename(columns={'ID': f'ID (Field: {field})'}, inplace = True)
    
    # styling del dataframe
    df_styled = metric_df.style.background_gradient().set_properties(**{'text-align': 'left'})
    
    if metric == 'CatalogCoverage':
        rounding = 1
    else:
        rounding = 3
        
    df_styled.set_precision(rounding)
    df_styled.hide_index()

    # percorsi per il salvataggio dei diversi files
    csv_output = f'{RESULT_PATH}/CSVs/[1M] {run["metric"]} - {run["fields"]}.csv'
    excel_output = f'{RESULT_PATH}/Excel/[1M] {run["metric"]} - {run["fields"]}.xlsx'
    png_output = f'{RESULT_PATH}/PNGs/{metric}/[1M] {run["metric"]} - {run["fields"]}.png'
    
    # salvataggio su file
    #metric_df.to_csv(csv_output, index = False)
    metric_df.to_excel(excel_output, index = False)
    #dfi.export(df_styled, png_output)

def delta_to_table(metric, field, cf):
    run = {'fields' : field, 'metric': metric, 'candidate': '', 'cutoff': []}
    data = {'ID':[]}
    suffixs = [' | Blockbuster', ' | Niche', ' | Diverse']
    frames = []
    delta_frames = []
    metric_string = ''
    cutoffs = []
    cutoffs.append(cf)
    
    # itera tra le rappresentazioni per la combinazione metrica-campo in considerazione
   
    for representation in representations_list:
        # itera tra le directories in base alla rappresentazione
        current_dir = f'{DIR}{representation}/'
        # itera tra i candidate items
        for candidate in candidate_items:
            run['candidate'] = candidate
            # sceglie il cutoff in base al candidate item considerato
            if(run['candidate'] == 'Test Ratings'):
                cutoffs.append(cf)
            #else:
                #cutoffs = [10,20]
            # itera sui due cutoffs scelti
            for cutoff in cutoffs:
                run['cutoff'] = cutoff
                data[f'DeltaGAP@{cf} | Blockbuster'] = []
                data[f'DeltaGAP@{cf} | Niche'] = []
                data[f'DeltaGAP@{cf} | Diverse'] = []
                # itera tra tutti i file di una data rappresentazione
                for filename in os.listdir(current_dir):
                    # controlla se il nome del file soddisfa il campo in considerazione, candidate e cutoff
                    if(check_filename(filename, run)):
                        # prende valore e combinazione algoritmo/rappresentazione
                        for suffix in suffixs:
                            metric_string = f'DeltaGap - Top {cutoff}{suffix}'
                            id, val = get_metric_value(current_dir, filename, metric_string, run['candidate'])
                            column_name = f'DeltaGAP@{cutoff}{suffix}'
                            
                            #inversione ID
                            id = id.split("+")
                            id_inv = f'{id[1]}+{id[0]}'
                        
                            data['ID'].append(id_inv)
                            # data['ID'].append(id)
                            
                            data[column_name].append(round(val,3))
        
                            keys = data.keys()
                            for key in keys:
                                if(key != column_name and key!='ID'):
                                    data[key].append(0)

                # crea un dataframe dal dizionario
                df = pd.DataFrame.from_dict(data)
                # aggiunge il dataframe alla lista dei dataframe da concatenare
                frames.append(df)
                # svuota il dizionario
                data = {'ID':[]}

    metric_df = pd.concat(frames)
    
    # modificare questa (e aggiungere 0 nei campi non interessati in fase di 
    # retrieve delle tuple) se si vogliono entrambi i cutoff in tabella
    metric_df = metric_df.groupby('ID').agg({
        'ID': 'first',
        f'DeltaGAP@{cutoff} | Blockbuster': sum,
        f'DeltaGAP@{cutoff} | Niche': sum,  
        f'DeltaGAP@{cutoff} | Diverse': sum,   
    })
    
    metric_df.rename(columns={'ID': f'ID (Field: {field})'}, inplace = True)
    
    
    #df_styled = metric_df.style.background_gradient(axis=0, gmap=(metric_df[f'DeltaGAP@{cutoff} | Blockbuster']), cmap='YlOrRd')
    df_styled = metric_df.style.background_gradient().set_properties(**{'text-align': 'left'})

    df_styled.set_precision(3)
    df_styled.hide_index()

    csv_output = f'{RESULT_PATH}/CSVs/[1M] {run["metric"]}@{cf} - {run["fields"]}.csv'
    excel_output = f'{RESULT_PATH}/Excel/[1M] {run["metric"]}@{cf} - {run["fields"]}.xlsx'
    png_output = f'{RESULT_PATH}/PNGs/DeltaGAP/[1M] {run["metric"]}@{cf} - {run["fields"]}.png'
    
    metric_df.to_excel(excel_output, index = False)
    #metric_df.to_csv(csv_output, index = False)
    #dfi.export(df_styled, png_output)