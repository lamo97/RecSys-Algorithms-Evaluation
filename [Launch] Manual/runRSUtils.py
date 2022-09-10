from clayrs import recsys as rs
import runEV

def predict(algorithm, run, train_list, test_list, ratings):
    path = 'D:/Repository/RecSys-Algorithms-Evaluation/'

    dataset_path = path + 'Dataset/Movielens 1M/'

    # Result Ranks' path
    rs_path = path + f'RS Results {run["dataset"]}/'

    # Hold Out Partitioning (returns a single train/test set)
    training_set = train_list[0]
    test_set = test_list[0]

    # Serialized items' path
    items_path = f'{dataset_path}movies_codified/{run["fields"]}/{run["representation"]}/'

    # Recommender System configuration
    print(f'### Loading items from {items_path}')

    cbrs = rs.ContentBasedRS(algorithm, training_set, items_path)
    cbrs.fit()

    # Results with Test Ratings methodology
    print(f'Running: {run["algorithm"]} w/Test Ratings')
    run['methodology'] = "Test Ratings"
    result_list = []

    result_rank = cbrs.rank(test_set, methodology=rs.TestRatingsMethodology(), n_recs=10)
    result_list.append(result_rank)
    
    # salva il rank per gli utenti in un CSV
    filename = (run['fields'] + ' - '+ run['representation'] + ' - ' +  run['algorithm'] + ' - ' +  run['methodology'])
    result_rank.to_csv((rs_path + run['representation'] + '/'), filename, overwrite=True)

    runEV.evaluate(result_list, test_list, run)

    # Results with All Items methodology
    print("Running: All Items")
    run['methodology'] = "All Items"
    result_list = []

    result_rank = cbrs.rank(test_set,methodology=rs.AllItemsMethodology(set(ratings.item_id_column)), n_recs=20)
    result_list.append(result_rank)
    
    # Result Rank is saved on file
    filename = (run['fields'] + ' - '+ run['representation'] + ' - ' +  run['algorithm'] + ' - ' +  run['methodology'])
    result_rank.to_csv((rs_path + run['representation'] + '/'), filename, overwrite=True)

    runEV.evaluate(result_list, test_list, run)