
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

# nltk.download('wordnet')


#nltk.download('words')
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import re


#######################
####data import #######
#######################


data_ = pd.read_excel('br.xlsx', engine='openpyxl')
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


####Konlpy_Okt############


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
documents = data.fillna('')
documents = documents.dropna(axis=0)



############################
#####tf-ifd trasform#########
#############################

vectorizer = CountVectorizer(analyzer='word',
                             tokenizer=None,
                             preprocessor=None,
                             stop_words=None,
                             max_df=0.5,
                             min_df=3,  # 토큰이 나타날 최소 문서 개수
                             ngram_range=(1, 1),
                             max_features=10000
                             )
vectorizer


pipeline = Pipeline([
    ('vect', vectorizer),
])


data_features = pipeline.fit_transform(data['corpus'])
data_features

print(data_features.shape)

vocab = vectorizer.get_feature_names()
print(len(vocab))
print(vocab[:10])


import numpy as np

dist = np.sum(data_features, axis=0)

for tag, count in zip(vocab, dist):
    print(count, tag)

print(pd.DataFrame(dist, columns=vocab))

print(pd.DataFrame(data_features[:10].toarray(), columns=vocab).head())

print(data_features)


df = pd.DataFrame(data_features.toarray(), columns=vocab)
df


data__ = data_
data__ = data__.drop(['text'], axis=1)



import xlrd
from openpyxl import load_workbook
data2 = pd.read_csv("../data/re_topic.csv", encoding='utf8')
data2 = data2.head(10)


topic = '토픽1'
wordlist = data2[topic].dropna().tolist()
cols = df.columns.tolist()
intersect = []

for c in cols:
    if c in wordlist:
        intersect.append(c)

df_cp = df[intersect]
df_= df_cp.sum(axis=1)
data__[topic] = df_
data__[topic]

data__
###################################
##########Save DF to CSV###########
##############################

# df.to_excel('br_tf.xlsx', engine='openpyxl')



##############