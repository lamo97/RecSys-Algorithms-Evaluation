import os

dir = 'C:/Users/glamo/Desktop/Repository/RecSys-Algorithms-Evaluation/Eval Results/'

parameters = {
    "fields" : [],
    "representations" : [],
    "algorithms" : []
}

for filename in os.listdir(dir):
    filename = filename.split(" - ")
    
    if((filename[1] in parameters['fields']) == False):
        parameters['fields'].append(filename[1])
    
    if((filename[2] in parameters['representations']) == False):
        parameters['representations'].append(filename[2])

    if((filename[3] in parameters['algorithms']) == False):
        parameters['algorithms'].append(filename[3])

print(parameters)

