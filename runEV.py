from clayrs import content_analyzer as ca
from clayrs import evaluation as eva

def filename(fields, representation, algorithm, methodology, cutoff):
    results_path = 'C:/Users/glamo/Desktop/Repository/RecSys-Algorithms-Evaluation/Eval Results/'
    return (results_path + 'SYS - ' + fields + ' - ' + representation + ' - '
    + algorithm + ' (' + methodology + '@' + str(cutoff) + ').csv')

def evaluate(result_list, test_list, run):
    if (run["dataset"] == '100k'):
        dataset_path = 'C:/Users/glamo/Desktop/Repository/RecSys-Algorithms-Evaluation/Dataset/Movielens 100k/'
    else:
        dataset_path = 'C:/Users/glamo/Desktop/Repository/RecSys-Algorithms-Evaluation/Dataset/Movielens 1M/'

    ratings = ca.Ratings(ca.CSVFile(dataset_path + 'ratings.csv'))

    TestItems_cutoffs = [5,10]
    AllItems_cutoffs = [5,10]

    # catalog per la catalog coverage
    catalog = set(ratings.item_id_column)   

    # user group per il Delta GAP
    user_groups = {'popular_users': 0.3, 'medium_popular_users': 0.2, 'low_popular_users': 0.5} 

    if(run["methodology"] == 'Test Items'):
        for cutoff in TestItems_cutoffs:
            em = eva.EvalModel(
                result_list,
                test_list,
                metric_list=[
                    eva.PrecisionAtK(k=cutoff),
                    eva.RecallAtK(k=cutoff),
                    eva.FMeasureAtK(k=cutoff),
                    eva.NDCGAtK(k=cutoff),
                    eva.MRRAtK(k=cutoff),
                    eva.GiniIndex(),
                    eva.CatalogCoverage(catalog),
                    eva.DeltaGap(user_groups)
                ]
            )
            sys_result, users_result = em.fit()
            sys_result.to_csv(filename(run['fields'], run['representation'], run['algorithm'], run['methodology'], cutoff))

    elif(run["methodology"] == 'All Items'):
        for cutoff in AllItems_cutoffs:
            em = eva.EvalModel(
                result_list,
                test_list,
                metric_list=[
                    eva.PrecisionAtK(k=cutoff),
                    eva.RecallAtK(k=cutoff),
                    eva.FMeasureAtK(k=cutoff),
                    eva.NDCGAtK(k=cutoff),
                    eva.MRRAtK(k=cutoff),
                    eva.GiniIndex(),
                    eva.CatalogCoverage(catalog),
                    eva.DeltaGap(user_groups)
                ]
            )
            sys_result, users_result = em.fit()
            sys_result.to_csv(filename(run['fields'], run['representation'], run['algorithm'], run['methodology'], cutoff))