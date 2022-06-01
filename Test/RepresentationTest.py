import pandas as pd

# correzione dell'ordine di stampa
import functools
from operator import rshift
print = functools.partial(print, flush=True)

# import dei moduli per Content Analyzer, Recommender System e Evaluation come librerie
from clayrs import content_analyzer as ca
from clayrs import recsys as rs
from clayrs import evaluation as eva

with open("C:/Users/glamo/Desktop/Repository/RecSys-Algorithms-Evaluation/Dataset/movies.dat", "r") as movies:
    for entry in movies:
        print(movies.readline())
