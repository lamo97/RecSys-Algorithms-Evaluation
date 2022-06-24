import os
import pandas as pd
from numpy import sign, nan

path = 'D:/Repository/RecSys-Algorithms-Evaluation/'
dir = f'{path}Metrics/'
output_dir = f'{path}Metrics%/'
threshold = 5

def getPercentage(average, value):
    percentage = ((value - average)/average)*100

    if(percentage >= threshold) or (percentage <= -threshold): return int(percentage)
    else : return 0

def percentageChange(threshold = 5):
    for file in os.listdir(dir):
        edited = {}
        df = pd.read_csv(dir + file)
        edited['ID'] = df['ID']
        
        columns = list(df.columns)
        columns.pop(0)

        for column in columns:
            average = 0
            entry_number = 0
            sum = 0

            for row in df[column]:
                sum += row
                entry_number += 1

            average = (sum/entry_number)
            edited[column] = []

            for row in df[column]:
                if(row != 0):
                    edited[column].append(int(getPercentage(average, row)))
                else:
                    edited[column].append(0)

        edited_df = pd.DataFrame.from_dict(edited)
        edited_df.replace(0, nan, inplace=True)
        
        edited_df.to_csv(f'{output_dir}[%] {file}', index=False)

percentageChange()