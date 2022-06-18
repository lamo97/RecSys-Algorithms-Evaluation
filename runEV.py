from clayrs import content_analyzer as ca
from clayrs import evaluation as eva

path = 'D:/Repository/RecSys-Algorithms-Evaluation/'

def filename(fields, representation, algorithm, methodology, cutoff):
    return (representation + '/' + fields + ' - ' + representation + ' - '
    + algorithm + ' (' + methodology + '@' + str(cutoff) + ').csv')

def evaluate(result_list, test_list, run):
    if (run["dataset"] == '100k'):
        dataset_path = path + 'Dataset/Movielens 100k/'
        results_path = path + 'Eval Results - 100k/'
    else:
        dataset_path = path + 'Dataset/Movielens 1M/'
        results_path = path + 'Eval Results - 1M/'

    ratings = ca.Ratings(ca.CSVFile(dataset_path + 'ratings.csv'))

    TestRatings_cutoffs = [5,10]
    AllItems_cutoffs = [10,20]

    # catalog per la catalog coverage
    catalog = set(ratings.item_id_column)   

    # user group per il Delta GAP
    user_groups = {'Blockbuster': 0.2, 'Niche': 0.2, 'Diverse': 0.6} 

    if(run["methodology"] == 'Test Ratings'):
        for cutoff in TestRatings_cutoffs:
            em = eva.EvalModel(
                result_list,
                test_list,
                metric_list=[
                    eva.PrecisionAtK(k=cutoff),
                    eva.RecallAtK(k=cutoff),
                    eva.FMeasureAtK(k=cutoff),
                    eva.NDCGAtK(k=cutoff),
                    eva.MRRAtK(k=cutoff),
                    eva.GiniIndex(top_n=cutoff),
                    eva.CatalogCoverage(catalog, top_n=cutoff),
                    eva.DeltaGap(user_groups, top_n=cutoff)
                ]
            )
            sys_result, users_result = em.fit()
           
            sys_result.to_csv(
                results_path + 'SYS/' +
                filename(run['fields'], run['representation'], run['algorithm'], run['methodology'], cutoff))

            users_result.to_csv(
                results_path + 'USER/' +
                filename(run['fields'], run['representation'], run['algorithm'], run['methodology'], cutoff))
            

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
                    eva.GiniIndex(top_n=cutoff),
                    eva.CatalogCoverage(catalog, top_n=cutoff),
                    eva.DeltaGap(user_groups, top_n=cutoff)
                ]
            )
            sys_result, users_result = em.fit()
                       
            sys_result.to_csv(
                results_path + 'SYS/' +
                filename(run['fields'], run['representation'], run['algorithm'], run['methodology'], cutoff))

            users_result.to_csv(
                results_path + 'USER/' +
                filename(run['fields'], run['representation'], run['algorithm'], run['methodology'], cutoff))