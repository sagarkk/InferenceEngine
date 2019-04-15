#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 23:36:43 2019

@author: sagarkk
"""
#creating test and train dataset 
reviews_train = []
for line in open('/home/sagarkk/projects/PythonFlask/movie_data/full_train.txt', 'r'):
    reviews_train.append(line.strip())
    
reviews_test = []
for line in open('/home/sagarkk/projects/PythonFlask/movie_data/full_test.txt', 'r'):
    reviews_test.append(line.strip())

#clean the data
import re

REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

def preprocess_reviews(reviews):
    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
    
    return reviews

reviews_train_clean = preprocess_reviews(reviews_train)
reviews_test_clean = preprocess_reviews(reviews_test)

from nltk.corpus import stopwords

english_stop_words = stopwords.words('english')
def remove_stop_words(corpus):
    removed_stop_words = []
    for review in corpus:
        removed_stop_words.append(
            ' '.join([word for word in review.split() 
                      if word not in english_stop_words])
        )
    return removed_stop_words

no_stop_words = remove_stop_words(reviews_train_clean)

def get_lemmatized_text(corpus):
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    return [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in corpus]

lemmatized_reviews = get_lemmatized_text(no_stop_words)



#vectorization
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


target = [1 if i < 12500 else 0 for i in range(25000)]

from sklearn.feature_extraction.text import HashingVectorizer
hv = HashingVectorizer(n_features=1802180)
X = hv.transform(lemmatized_reviews)
X_test = hv.transform(reviews_test_clean)

'''
ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2))
ngram_vectorizer.fit(lemmatized_reviews)
X = ngram_vectorizer.transform(lemmatized_reviews)
X_test = ngram_vectorizer.transform(reviews_test_clean)
'''
#train the dataset
from sklearn.svm import LinearSVC
final_model = LinearSVC(C=0.01)
final_model.fit(X, target)
#print ("Final Accuracy: %s" 
 #      % accuracy_score(target, final_model.predict(X_test)))

#y_pred = final_model.predict(X_test)

import pickle
pickle.dump(final_model,open('/home/sohit/Desktop/code/PythonFlask/model.pkl','wb'))

