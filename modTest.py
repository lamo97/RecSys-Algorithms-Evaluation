def addRepresentation(field):
    ca_config.add_multiple_config(
    field,
    [
        # SK-TFIDF 
        ca.FieldConfig(
            ca.SkLearnTfIdf(),
            ca.NLTK(stopwords_removal=True, lemmatization=True),
            id='SK-TFIDF'
        ),

        # Whoosh-TFIDF 
        ca.FieldConfig(
            ca.WhooshTfIdf(),
            ca.NLTK(stopwords_removal=True, lemmatization=True),
            id='Whoosh-TFIDF'
        ),

        # Word2Vec 
        ca.FieldConfig(
            ca.WordEmbeddingTechnique(ca.GensimWord2Vec()),
            ca.NLTK(stopwords_removal=True, lemmatization=True),
            id='Word2Vec'
        ),
        
        # Doc2Vec 
        ca.FieldConfig(
            ca.DocumentEmbeddingTechnique(ca.GensimDoc2Vec()),
            ca.NLTK(stopwords_removal=True, lemmatization=True),
            id='Doc2Vec'
        ),

        # GensimLDA 
        ca.FieldConfig(
            # Probabilmente LDA classifica DOCUMENTI interni e non parole (tenere conto anche del campo)
            ca.SentenceEmbeddingTechnique(ca.GensimLDA()),
            ca.NLTK(stopwords_removal=True, lemmatization=True),
            id='GensimLDA'
        ),

        # GensimRandomIndexing
        ca.FieldConfig(
            ca.DocumentEmbeddingTechnique(ca.GensimRandomIndexing()),
            ca.NLTK(stopwords_removal=True, lemmatization=True),
            id='GensimRandomIndexing'
        ),

        # GensimFastText
        ca.FieldConfig(
            ca.WordEmbeddingTechnique(ca.GensimFastText()),
            ca.NLTK(stopwords_removal=True, lemmatization=True),
            id='GensimFastText'
        ),

        # GensimLatentSemanticAnalysis
        ca.FieldConfig(
            ca.WordEmbeddingTechnique(ca.GensimLatentSemanticAnalysis()),
            ca.NLTK(stopwords_removal=True, lemmatization=True),
            id='GensimLatentSemanticAnalysis'
        ),

        # Word2Doc-GloVe
        ca.FieldConfig(
            ca.Word2DocEmbedding(
                ca.Gensim('glove-twitter-50'),
                combining_technique=ca.Centroid()
            ),
            ca.NLTK(stopwords_removal=True, lemmatization=True),
            id='Word2Doc-GloVe'
        ),

        # Sentence2Doc-Sbert
        ca.FieldConfig(
            ca.Sentence2DocEmbedding(
                ca.Sbert('paraphrase-distilroberta-base-v1'),
                combining_technique=ca.Centroid()
            ),
            ca.NLTK(stopwords_removal=True, lemmatization=True),
            id='Sentence2Doc-Sbert'
        )
    ]
)