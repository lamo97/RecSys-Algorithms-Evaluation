# RecSys-Algorithms-Evaluation
Repository for my thesis on Study of the Popularity Bias of Content-Based Recommendation Algorithms (Studio del Popularity Bias degli Algoritmi di Content-Based Recommendation).

These scripts were meant to be used just to run experiments for my thesis and hold just a demonstrative value.

These scripts/notebooks are structured in such way in order to execute just the methods you need at the moment (the execution can, sometimes, take quite some time, due to the complexity of some algorithms and operations).
A code editor (like VS Code) with and extension that allows a notebook-like execution is needed.
The cells ran one after another will be queued.

## Content Analyzer: runCA - Manual.py
This script, structured in a "Jupiter Notebook" fashion, allows the execution of the Content Analyzer, and therefor, the serialization of items.
### Usage
In order to run this script on your machine, you need to:
* run the first cell `# In[1]: Setup` with the appropriate values to config the CA
* run any other cell (depending on which representation you wish to generate)

## Recommender System: runRS - Manual.py
The content represented by the Content Analyzer can be now used to generate predictions through the Recommender System. This script allows you to do so with 4 different algorithms.   

This script uses the methods defined in `runRSUtils.py` and `runEV.py`.  

By default, predictions will be generated with both "Test Ratings" and "All Items" methodology (the latter requires more computation time).
Once the prediction phase is over, the script will invoke a few methods in from runEV.py, in order to evaluate the results (it should take just a few seconds).

### Usage
In order to run this script on your machine, you need to:
* customize the required values in `runRSUtils.py` (`path`, mostly)
* run the first cell `# In[1]: Setup` with the appropriate values to config the RS
* choose the cell corrisponding to the algorithm you wish to generate predictions with (mind the number of features)

## Evaluation Module: runEV.py
This script contains methods used to evaluate the results obtained from the Recommender System.
The evaluation metrics are:
* Precision
* Recall
* F-Measure
* NDCG
* MRR
* Gini Index
* Catalog Coverage
* Delta GAP

### Usage
Customize the `path` and any other value concerning the metrics you wish; the methods will be invoked by the other scripts.
