import pandas as pd

def convertToCsv(path = 'C:/Users/glamo/Desktop/Repository/RecSys-Algorithms-Evaluation/Dataset/Movielens 1M/'):
    ratings_data = {
        'user_id': [],
        'movie_id': [],
        'rating': [],
        'timestamp': []
    }

    with open(path + 'ratings.dat') as ratings:
        for entry in ratings:
            entry = entry.split("::")
            ratings_data['user_id'].append(entry[0])
            ratings_data['movie_id'].append(entry[1])
            ratings_data['rating'].append(entry[2])
            ratings_data['timestamp'].append(entry[3].split("\n")[0])

    ratings.close()

    ratings_df = pd.DataFrame(ratings_data, columns=['user_id', 'movie_id', 'rating', 'timestamp'])
    ratings_df.to_csv(path + "ratings.csv", index=False)