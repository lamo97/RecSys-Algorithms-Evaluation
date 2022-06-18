from clayrs import content_analyzer as ca
from clayrs import recsys as rs
from clayrs import evaluation as eva
import sys
import runEV

def predict(algorithm, run, train_list, test_list, ratings):
    path = 'D:/Repository/RecSys-Algorithms-Evaluation/'

    # sceglie il dataset in base al parametro passato
    if (run["dataset"] == '100k'):
        dataset_path = path + 'Dataset/Movielens 100k/'
    else:
        dataset_path = path + 'Dataset/Movielens 1M/'

    rs_path = path + f'RS Results {run["dataset"]}/'

    # l'Hold Out restituisce un singolo training set e test set
    training_set = train_list[0]
    test_set = test_list[0]

    cbrs = rs.ContentBasedRS(algorithm, training_set, (dataset_path + '/movies_codified'))
    cbrs.fit()


    # Risultati con Test Ratings
    print("Running: Test Ratings")
    run['methodology'] = "Test Ratings"
    result_list = []

    result_rank = cbrs.rank(test_set, methodology=rs.TestRatingsMethodology(), n_recs=10)
    result_list.append(result_rank)
    
    # salva il rank per gli utenti in un CSV
    filename = (run['fields'] + ' - '+ run['representation'] + ' - ' +  run['algorithm'] + ' - ' +  run['methodology'])
    result_rank.to_csv((rs_path + run['representation'] + '/'), filename, overwrite=True)

    runEV.evaluate(result_list, test_list, run)

    # Risultati con AllItems
    print("Running: All Items")
    run['methodology'] = "All Items"
    result_list = []
    result_rank = cbrs.rank(test_set,methodology=rs.AllItemsMethodology(set(ratings.item_id_column)), n_recs=20)
    result_list.append(result_rank)
    
    # salva il rank per gli utenti in un CSV
    filename = (run['fields'] + ' - '+ run['representation'] + ' - ' +  run['algorithm'] + ' - ' +  run['methodology'])
    result_rank.to_csv((rs_path + run['representation'] + '/'), filename, overwrite=True)

    runEV.evaluate(result_list, test_list, run)