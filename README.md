# RecSys-Algorithms-Evaluation
Repository per la tesi su: "Studio del Popularity Bias degli Algoritmi di Content-Based Recommendation"

Classi, Parametri e Metodi per le Rappresentazioni:<br />
https://various-tax-737.notion.site/ClayRS-Representations-Classes-a31d730f141342bd8be5f9ac765e92ea

Classi, Parametri e Metodi per le Raccomandazioni: <br />
https://various-tax-737.notion.site/ClayRS-Modulo-recsys-439da14c12cf465dbafbd74457e9f8f5

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
**runCA.py:** <br />
  **[1]:** **fields** <br />
		- plot genres (dataset 100k) <br />
		- description tags genres reviews (dataset 1M) <br /><br />
	**[2]:** **dataset** <br />
		- 100k <br />
		- 1M <br /> <br />
 <br />
**runRS.py:** <br />
	**[1]:** **fields** <br />
		- plot genres (dataset 100k) <br />
		- description tags genres reviews (dataset 1M) <br /><br />
	**[2]:** **representations** <br />
		- all <br />
		- SK-TFIDF <br />
    - Word2Vec <br />
		- Doc2Vec <br />
    - GensimLDA <br />
		- GensimRandomIndexing <br />
		- GensimFastText <br />
		- GensimLatentSemanticAnalysis <br />
    - Word2Doc-GloVe <br />
		- Sentence2Doc-Sbert <br /><br />
	**[3]:** **dataset** <br />
		- 100k <br />
		- 1M <br />

<br />
es. <br />
python runCA.py plot,genres 1M <br />
python runRS.py plot,genres all 1M
