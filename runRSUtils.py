from clayrs import content_analyzer as ca
from clayrs import recsys as rs
from clayrs import evaluation as eva
import sys
import runEV

def predict(algorithm, run, train_list, test_list, ratings):
    # sceglie il dataset in base al parametro passato
    if (run["dataset"] == '100k'):
        dataset_path = 'C:/Users/glamo/Desktop/Repository/RecSys-Algorithms-Evaluation/Dataset/Movielens 100k/'
        rank_path = 'C:/Users/glamo/Desktop/Repository/RecSys-Algorithms-Evaluation/Ranks-100k/'
    else:
        dataset_path = 'C:/Users/glamo/Desktop/Repository/RecSys-Algorithms-Evaluation/Dataset/Movielens 1M/'
        rank_path = 'C:/Users/glamo/Desktop/Repository/RecSys-Algorithms-Evaluation/Ranks-1M/'

    train_set = train_list[0]
    test_set = test_list[0]

    cbrs = rs.ContentBasedRS(algorithm, train_set, (dataset_path + '/movies_codified'))
    cbrs.fit()

    test_set = test_list[0]
    # rank = cbrs.rank(test_set, user_id_list = ['8', '2', '1'], n_recs = 3)

    # Risultati con Test Ratings
    print("Running: Test Ratings")
    run['methodology'] = "Test Items"
    result_list = []
    #salvare e riusare
    result_rank = cbrs.rank(test_set, methodology=rs.TestRatingsMethodology(), n_recs=10)
    result_list.append(result_rank)

    # salva il rank per gli utenti in un CSV
    filename = (run['fields'] + ' - '+ run['representation'] + ' - ' +  run['algorithm'] + ' - ' +  run['methodology'])
    result_rank.to_csv((rank_path + run['representation'] + '/'), filename)

    # spostare
    runEV.evaluate(result_list, test_list, run)

    # Risultati con AllItems
    print("Running: All Items")
    run['methodology'] = "All Items"
    result_list = []
    result_rank = cbrs.rank(test_set,methodology=rs.AllItemsMethodology(set(ratings.item_id_column)), n_recs=20)
    result_list.append(result_rank)
    
    # salva il rank per gli utenti in un CSV
    filename = (run['fields'] + ' - '+ run['representation'] + ' - ' +  run['algorithm'] + ' - ' +  run['methodology'])
    
    result_rank.to_csv((rank_path + run['representation'] + '/'), filename)

    # runEV.evaluate(result_list, test_list, run)