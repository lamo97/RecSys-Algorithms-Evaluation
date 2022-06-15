from clayrs import content_analyzer as ca
from clayrs import recsys as rs
from clayrs import evaluation as eva
import sys
import runEV

def predict(algorithm, run):
    if (run["dataset"] == '100k'):
        dataset_path = 'C:/Users/glamo/Desktop/Repository/RecSys-Algorithms-Evaluation/Dataset/Movielens 100k/'
    else:
        dataset_path = 'C:/Users/glamo/Desktop/Repository/RecSys-Algorithms-Evaluation/Dataset/Movielens 1M/'

    ratings = ca.Ratings(ca.CSVFile(dataset_path + 'ratings.csv'))
    
    if(run["split"] == 'ho'):
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

        runEV.evaluate(result_list, test_list, run)

        # Risultati con AllItems
        run['methodology'] = "All Items"
        result_list = []
        result_rank = cbrs.rank(test_set,methodology=rs.AllItemsMethodology(set(ratings.item_id_column)))
        result_list.append(result_rank)

        runEV.evaluate(result_list, test_list, run)

    elif(run["split"] == 'kf'):
        kf = rs.KFoldPartitioning(n_splits=2)
        train_list, test_list = kf.split_all(ratings)

        # Risultati con TestItems
        run['methodology'] = "Test Items"
        result_list = []

        for train_set, test_set in zip(train_list, test_list):
            cbrs = rs.ContentBasedRS(algorithm, train_set, (dataset_path + '/movies_codified'))
            rank_to_append = cbrs.fit_rank(test_set, methodology=rs.TestItemsMethodology())
            result_list.append(rank_to_append)
            #####################
            rank_to_append.to_csv('C:/Users/glamo/Desktop/Repository/RecSys-Algorithms-Evaluation', ('score'+ run['representation'] + ' - ' +  run['algorithm'] + '.csv'))

        runEV.evaluate(result_list, test_list, run)

        # Risultati con AllItems
        run['methodology'] = "All Items"
        result_list = []

        for train_set, test_set in zip(train_list, test_list):
            rank_to_append = cbrs.fit_rank(test_set, methodology=rs.TestItemsMethodology())
            result_list.append(rank_to_append)
            
        runEV.evaluate(result_list, test_list, run)

    