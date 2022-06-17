from operator import index
import pandas as pd
import csv
import os

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

dir = 'C:/Users/glamo/Desktop/Repository/RecSys-Algorithms-Evaluation/Eval Results - 100k/'
result_path = 'C:/Users/glamo/Desktop/Repository/RecSys-Algorithms-Evaluation/Metrics/'

# print(check_filename('plot - SK-TFIDF - Centroid Vector (All Items@5).csv','plot','All Items',5))
# print(retrieve_tuple(dir, 'plot - SK-TFIDF - Centroid Vector (All Items@5).csv', 'Precision@5 - macro'))

data = {
    'ID' : [],
    'val' : []
}

candidate_items = ['Test Items', 'All Items']

representations_list =  [
        'SK-TFIDF',
        'Word2Vec', 'Doc2Vec',
        'GensimLDA','GensimRandomIndexing', 'GensimFastText', 'GensimLatentSemanticAnalysis',
        'Word2Doc-GloVe','Sentence2Doc-Sbert']

metric_prefix = 'Precision@'

# metric_df = pd.DataFrame()
# metric_df.insert(0, column='ID', value=None)
# metric_df['ID'] = metric_df['ID'].astype('float64')
# print(metric_df.dtypes)


frames = []
# itera tra le rappresentazioni per la combinazione metrica-campo in considerazione
for representation in representations_list:
    # itera tra le directories in base alla rappresentazione
    current_dir = dir + representation + '/'
    # itera tra i candidate items
    for candidate in candidate_items:
        # sceglie il cutoff in base al candidate item considerato
        if(candidate == 'Test Items'):
            cutoffs = [5,10]
        else:
            cutoffs = [10,20]
        # itera sui due cutoffs scelti
        for cutoff in cutoffs:
            metric = metric_prefix + str(cutoff) + ' - macro'
            for filename in os.listdir(current_dir):
                if(check_filename(filename, 'plot', candidate, cutoff) == True):
                    #print(filename)
                    #print("Candidate: ", candidate, "\tCutoff: ", cutoff, "\tMetric", metric)
                    #input()
                    id, val = retrieve_tuple(current_dir, filename, metric)
                    data['ID'].append(id)
                    data['val'].append(round(val,5))
            # print(data)
            df = pd.DataFrame.from_dict(data)
            col_name = metric.split(" - ")[0]
            col_name = col_name + ' | ' + candidate
            df.rename(columns={'val': col_name}, inplace=True)
            # print(df)
            frames.append(df)
            # pd.merge(metric_df, df, on='ID', how='right')
            data['ID'] = []
            data['val'] = []

metric_df = pd.concat(frames)

#metric_df.set_index('ID', drop=False, inplace=True)

#metric_df.groupby(df.index)


#metric_df = pd.merge(metric_df, metric_df, on='ID', how='outer')

#metric_df.set_index('ID')
metric_df = metric_df.groupby('ID').agg({
    'ID': 'first',
    'Precision@5 | Test Items':sum,  
    'Precision@10 | Test Items':sum,
    'Precision@10 | All Items': sum,  
    'Precision@20 | All Items': sum
})

metric_df.to_csv(result_path + 'Precision - Plotsss.csv', index=False)
print(metric_df)

#1 = pd.DataFrame.from_dict(data)
#            
#            print(df1)
