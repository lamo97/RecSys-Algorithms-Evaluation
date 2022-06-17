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

dir = 'C:/Users/glamo/Desktop/Repository/RecSys-Algorithms-Evaluation/Eval Results - 100k/SK-TFIDF/'

# print(check_filename('plot - SK-TFIDF - Centroid Vector (All Items@5).csv','plot','All Items',5))
# print(retrieve_tuple(dir, 'plot - SK-TFIDF - Centroid Vector (All Items@5).csv', 'Precision@5 - macro'))

data = {
    'ID' : [],
    'val' : []
}

candidate_items = ['Test Items', 'All Items']

or_metric = 'Precision@'

frames = []

for candidate in candidate_items:
    if(candidate == 'Test Items'):
        cutoffs = [5,10]
    else:
        cutoffs = [10,20]
    for cutoff in cutoffs:
        metric = or_metric + str(cutoff) + ' - macro'
        for filename in os.listdir(dir):
            if(check_filename(filename, 'plot', candidate, cutoff) == True):
                print(filename)
                print("Candidate: ", candidate, "\tCutoff: ", cutoff, "\tMetric", metric)
                input()
                id, val = retrieve_tuple(dir, filename, metric)
                data['ID'].append(id)
                data['val'].append(val)
        print(data)
        df = pd.DataFrame.from_dict(data)
        col_name = metric.split(" - ")[0]
        col_name = col_name + ' | ' + candidate
        df.rename(columns={'val': col_name}, inplace=True)
        print(df)
        frames.append(df)
        data['ID'] = []
        data['val'] = []

for entry in frames:
    print(entry)

#1 = pd.DataFrame.from_dict(data)
#            
#            print(df1)
