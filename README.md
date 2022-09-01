# RecSys-Algorithms-Evaluation
Repository per la tesi su: "Studio del Popularity Bias degli Algoritmi di Content-Based Recommendation"

Notebooks di esempio:

1o esempio
https://colab.research.google.com/drive/1oy-0zX5udiTNJ8mk3orimwK01R-6rjY1?usp=sharing

3o esempio
https://colab.research.google.com/drive/11OoyVLVpw7lfRHtCzAT6l-rWFa4F77AD?usp=sharing

4o esempio
https://colab.research.google.com/drive/1ym8pspwGS-8vxAUu9xsdpTcV7LjHtzEe?usp=sharing

5o esempio
https://colab.research.google.com/drive/1pOXvRY8lW9PQ2VmhQUmtZX4Y-MJfs1Yi?usp=sharing

## Args
### runCA.py:
	[1]: fields
		- plot | genres		 		-> (dataset 100k)
		- description | tags | genres | reviews -> (dataset 1M)
	[2]: dataset
		- 100k
		- 1M
		
### runRS.py:
	[1]: fields
		- plot | genres		 		-> (dataset 100k)
		- description | tags | genres | reviews -> (dataset 1M)
	[2]: representations
		- all
		- SK-TFIDF
    	- Word2Vec
		- Doc2Vec 
    	- GensimLDA
		- GensimRandomIndexing
		- GensimFastText
		- GensimLSA
    	- Word2Doc-GloVe
		- Sentence2Doc-Sbert
	[3]: dataset
		- 100k
		- 1M

### Ex.
	python runCA.py plot,genres 100k
	python runRS.py plot,genres all 100k
	python runRS.py description Word2Vec 1M

### Combinazioni
	description
    genres
    tags
    reviews
    description,genres,tags
    description,genres,reviews
    description,tags,reviews
    genres,tags,reviews
    description,genres,tags,reviews
