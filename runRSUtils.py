from clayrs import content_analyzer as ca
from clayrs import recsys as rs
from clayrs import evaluation as eva
import sys
import runEV

def predict(algorithm, run):
    # sceglie il dataset in base al parametro passato
    if (run["dataset"] == '100k'):
        dataset_path = 'C:/Users/glamo/Desktop/Repository/RecSys-Algorithms-Evaluation/Dataset/Movielens 100k/'
        rank_path = 'C:/Users/glamo/Desktop/Repository/RecSys-Algorithms-Evaluation/Ranks-100k/'
    else:
        dataset_path = 'C:/Users/glamo/Desktop/Repository/RecSys-Algorithms-Evaluation/Dataset/Movielens 1M/'
        rank_path = 'C:/Users/glamo/Desktop/Repository/RecSys-Algorithms-Evaluation/Ranks-1M/'

    #apretura file dei ratings
    ratings = ca.Ratings(ca.CSVFile(dataset_path + 'ratings.csv'))
    
    train_list, test_list =  rs.HoldOutPartitioning(train_set_size=0.8).split_all(ratings)                         
    train_set = train_list[0]

    cbrs = rs.ContentBasedRS(algorithm, train_set, (dataset_path + '/movies_codified'))
    cbrs.fit()

    test_set = test_list[0]
    rank = cbrs.rank(test_set, user_id_list = ['8', '2', '1'], n_recs = 3)

    # Risultati con TestItems
    run['methodology'] = "Test Items"
    result_list = []
    result_rank = cbrs.rank(test_set,methodology=rs.TestItemsMethodology())
    result_list.append(result_rank)

    # salva il rank per gli utenti in un CSV
    filename = (run['fields'] + ' - '+ run['representation'] + ' - ' +  run['algorithm'] + ' - ' +  run['methodology'])
    result_rank.to_csv((rank_path + run['representation'] + '/'), filename)

    runEV.evaluate(result_list, test_list, run)

    # Risultati con AllItems
    run['methodology'] = "All Items"
    result_list = []
    result_rank = cbrs.rank(test_set,methodology=rs.AllItemsMethodology(set(ratings.item_id_column)))
    result_list.append(result_rank)
    
    # salva il rank per gli utenti in un CSV
    filename = (run['fields'] + ' - '+ run['representation'] + ' - ' +  run['algorithm'] + ' - ' +  run['methodology'])
    result_rank.to_csv((rank_path + run['representation'] + '/'), filename)

    runEV.evaluate(result_list, test_list, run)