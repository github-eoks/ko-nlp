
#20190503_kr_lda.py
from konlpy.tag import Okt
# from konlpy.tag import Twitter
import pandas as pd
import nltk
from openpyxl import load_workbook


import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)

import re

#######################
####data import #######
#######################
data_ = pd.read_excel('br.xlsx', engine='openpyxl')

data_ = data_[:100]
print(data_.shape)
part = 'BR'
data_['index'] = data_.index

data = data_
documents = data
print(data.shape)
print(data.head())

data[part][:10]
data = data[part]

########################
#######Konlpy Okt ######
########################

corpus = []
twitter = Okt()
len(data)
data = data.fillna("\n")

for i in range(0, len(data), 1):
    if i % 100 == 0:
        print("Processing...: ", i)

    words = []
    pos = twitter.pos(data.iloc[i])
    for i in range(0,len(pos)):
        if pos[i][1] in ["Noun"]:
            words.append(pos[i][0])

    join_words = ','.join(words)
    corpus.append(join_words)

s_corpus = pd.Series(corpus)
data['corpus'] = s_corpus
documents = data
documents = data.fillna('')
documents = documents.dropna(axis=0)


def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 1:
            result.append(token)
    return result



processed_docs = documents['corpus'].map(preprocess)
print(processed_docs[:100])

tokenized_doc = processed_docs

from gensim import corpora
dictionary = corpora.Dictionary(tokenized_doc)
dictionary.filter_extremes(no_below=3, no_above=0.5)


corpus = [dictionary.doc2bow(text) for text in tokenized_doc]
print(corpus[1])

print(dictionary[66])
len(dictionary)



import gensim


num_topics = 120
chunksize = 2000
passes = 50
iterations = 400
eval_every = None

import time
from gensim.models.coherencemodel import CoherenceModel

# 'u_mass', 'c_v', 'c_uci', 'c_npmi'


#######################
####최종 토픽 출력#####
#######################

tic = time.time()
lda4 = gensim.models.ldamodel.LdaModel(corpus, num_topics = 5, id2word=dictionary, passes=20)
# print('ntopics:',ntopics,time.time() - tic)

cm = CoherenceModel(model=lda4, texts=tokenized_doc, coherence='u_mass')
coherence = cm.get_coherence()
print("Coherence:",coherence)
# coherencesT.append(coherence)
print('Perplexity: ', lda4.log_perplexity(corpus),'\n\n')
# perplexitiesT.append(lda4.log_perplexity(corpus))

li_topic =[]
topics = lda4.print_topics(num_words=10)
for topic in topics:
    print(topic)
#     li_topic.append(topic)
#



print(lda4.print_topics())




texts = tokenized_doc
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    print(compute_coherence_values)
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model=gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_npmi')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus, texts=texts, start=2, limit=40, step=6)



# Show graph
import matplotlib.pyplot as plt
limit=40; start=2; step=6;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()

