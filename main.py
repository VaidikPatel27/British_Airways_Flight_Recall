import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import nltk
nltk.download('stopwords')
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from imblearn.over_sampling import RandomOverSampler
from xgboost import XGBClassifier as xgb
from sklearn.metrics import f1_score, precision_score, recall_score

df = pd.read_csv('/content/drive/MyDrive/Colab_Notebooks/Projects/Sentiment Analysis/sentiment.csv')
df['date'] = pd.to_datetime(df['date'])
df.dropna(inplace=True)
df.sample(3)

df_final = df.drop('customer_name', axis=1).dropna()

port_stemmer = PorterStemmer()

def stemming(content):
  stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
  stemmed_content = stemmed_content.lower()
  stemmed_content = stemmed_content.split(' ')
  stemmed_content = [port_stemmer.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
  stemmed_content = ' '.join(stemmed_content)
  return stemmed_content

df_final['stemmed_review_title'] = df_final['review_title'].apply(stemming)
df_final['stemmed_review'] = df_final['review'].apply(stemming)
df_final['stemmed_route'] = df_final['route'].apply(stemming)

X = df_final.iloc[:,[0,1,2,3,4,5,6,8,9,10]]
y = df_final.iloc[:,7]

vect = TfidfVectorizer()
title = vect.fit_transform(X['stemmed_review_title']).toarray()
review = vect.fit_transform(X['stemmed_review']).toarray()
route = vect.fit_transform(X['stemmed_route']).toarray()

def tfid_mean(arr):
  mean = []
  for i in range(arr.shape[0]):
    mean.append(np.where(arr[i] != 0)[0].mean())
  return mean

od = OrdinalEncoder()
df_ = pd.DataFrame()
df_['recommended'] = np.where(df_final['recommended'] == 'no', 0, 1)
df_['travel_type'] = od.fit_transform(np.array(df_final['traveller_type']).reshape(df_final.shape[0],1))
df_['seat_type'] = od.fit_transform(np.array(df_final['seat_type']).reshape(df_final.shape[0],1))
df_['title'] = tfid_mean(title)
df_['review'] = tfid_mean(review)
df_['route'] = tfid_mean(route)

X_ = df_.iloc[:,1:]
y_ = df_.iloc[:,0]

over_sample = RandomOverSampler(random_state = 1)
X_over_sample, y_over_sample = over_sample.fit_resample(X_, y_)

sc = StandardScaler()

X_over_sample = sc.fit_transform(X_over_sample)

X_train, X_test, y_train, y_test = train_test_split(X_over_sample, y_over_sample,
                                                    test_size = 0.2,
                                                    shuffle = True,
                                                    random_state = 1)
xg = xgb()
xg.fit(X_train, y_train)

y_pred = xg.predict(X_test)

print(f'f2 score: {f1_score(y_pred, y_test)}')
print(f'recall score: {recall_score(y_pred, y_test)}')
print(f'precision score: {precision_score(y_pred, y_test)}')

cross_score = cross_val_score(xg, 
                              X_train, y_train,
                              cv = 10)
print(f'cross validation score: {cross_score}')
print(f'minimum cross validation score: {cross_score.min()}')
