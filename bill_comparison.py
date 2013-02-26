import os
import json
import gensim
import logging
import csv
import collections
import fastcluster
import MySQLdb
from scipy import cluster
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
from matplotlib.pyplot import show

    

def create_tfidf_corpus(dict, texts):
    dictionary = dict
    print dictionary
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    # uncomment the below to save the corpus
    #gensim.corpora.MmCorpus.serialize('bill_corpus.mm', corpus)
    
    tfidf = gensim.models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    
    return corpus_tfidf
    
def create_lsi(dictionary, corpus_tfidf):
    num_topics = 250
    lsi = gensim.models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=num_topics)
    corpus_lsi = lsi[corpus_tfidf]
    index = gensim.similarities.docsim.MatrixSimilarity(lsi[corpus_tfidf]) 
    return index


def make_stoplist():
    f = csv.reader(open("./Legislative_Stopwords.csv", "rU"), dialect = "excel")

    stoplist = []
    print "making stoplist"
    for row in f:
        stoplist.append(row[0])
    return stoplist
				
if __name__=='__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    
    #This needs to be pointed at a database with the relevent info in it. Database architecture being queried here was put together by George Stephanis
    
    db = MySQLdb.connect("localhost","root","","test_bills" )
    cursor = db.cursor()
    sql = """SELECT id, bill_id, text FROM resolution_text WHERE LENGTH(text) > 15;
    """
    cursor.execute(sql)
    results = cursor.fetchall()
    
    bills = []
    for row in results:
        text_id = row[0]
        bill_id = row[1]
        text = row[2]
        processed = gensim.utils.simple_preprocess(text)
        bills.append(processed)
        data = (text_id, bill_id, processed)
        print data

    
    
    #point this at the included stopwords list
    stoplist = make_stoplist()
    
    print bills

    texts = [[word for word in document if word not in stoplist] for document in bills]
    
    dictionary = gensim.corpora.dictionary.Dictionary(texts)
    corpus_tfidf = create_tfidf_corpus(dictionary, texts)
    
    index = create_lsi(dictionary, corpus_tfidf)
    
    lsi_matrix = []
    
    for similarities in index:
        lsi_matrix.append(similarities)

        
    # Below produces HAC Clustering and Dendrogram
       
    clustering = fastcluster.ward(lsi_matrix)
    leaves = cluster.hierarchy.leaves_list(clustering)
    cluster.hierarchy.dendrogram(clustering, p=30, truncate_mode=None, color_threshold=3.5, get_leaves=True, orientation='top', labels=bill_names, count_sort=True, distance_sort=False, show_leaf_counts=True, no_plot=False, no_labels=False, color_list=None, leaf_font_size=12, leaf_rotation=None, leaf_label_func=None, no_leaves=False, show_contracted=False, link_color_func=None)
    show()
      
 