import csv
import numpy as np
import pandas as pd
import torch
import tensorflow as tf
import transformers # pytorch transformers
from preprocess import process
import time
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import svm
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

train_corpora = []
valid_corpora = []
train_labels = []
valid_labels = []
corpora = []
labels = []
outputs = []
feature_weights = {}

def f1_score(predictions, labels):
  TP = 0
  FP = 0
  FN = 0
  for i in range(len(predictions)):
    if predictions[i] == 1:
      if labels[i] == 1:
        TP += 1
      else:
        FP += 1
    elif labels[i] == 1:
      FN += 1
  pre = TP/(TP+FP)
  rec = TP/(TP+FN)
  f1 = 2*((pre*rec)/(pre+rec))
  return f1

def read_data():
  train_file = open("./train.tsv")
  valid_file = open("./valid.tsv")
  for raw_line in train_file:
    line = raw_line.split("\t")
    corpora.append(line[1])
    if line[2] == "INFORMATIVE\n":
      labels.append(1)
    else:
      labels.append(0)
  for raw_line in valid_file:
    line = raw_line.split("\t")
    corpora.append(line[1])
    if line[2] == "INFORMATIVE\n":
      labels.append(1)
    else:
      labels.append(0)

def kfold(k, i):
  for index in range(len(corpora)):
    if index % k == i:
      valid_corpora.append(corpora[index])
      valid_labels.append(labels[index])
    else:
      train_corpora.append(corpora[index])
      train_labels.append(labels[index])
  
def output_kfold_avgs(k):
  avgs = []
  for kfold in range(len(outputs)):
    for i in range(len(outputs[kfold])):
      if kfold == 0:
        avgs.append(outputs[kfold][i])
      else:
        avgs[i] += outputs[kfold][i]
  for i in range(len(avgs)):
    print(i, (avgs[i])/k)

#  Main ----------------------------------------------------------------------------------------------------------------------------------
read_data()

# # greatest weights in best model--------------------------------------------------------------------------------------------------------
def partition(arr,low,high): 
    i = ( low-1 )         # index of smaller element 
    pivot = arr[high]     # pivot 
    for j in range(low , high): 
        if feature_weights[arr[j]] < feature_weights[pivot]: 
            i = i+1 
            arr[i],arr[j] = arr[j],arr[i]
    arr[i+1],arr[high] = arr[high],arr[i+1] 
    return ( i+1 ) 

def quickSort(arr,low,high): 
    if low < high: 
        pi = partition(arr,low,high)
        quickSort(arr, low, pi-1) 
        quickSort(arr, pi+1, high) 

def lr_sorted_weights(clean_train_corpora):
  vectorizer = TfidfVectorizer(ngram_range=(1,3))
  vectorizer.fit(clean_train_corpora)
  train_vec = vectorizer.transform(clean_train_corpora)
  features = vectorizer.get_feature_names()

  model = LogisticRegression(solver='liblinear')
  model.fit(train_vec, train_labels)
  weights = model.coef_[0]

  for i in range(len(weights)):
    feature_weights[features[i]] = weights[i]

  quickSort(features, 0, len(features)-1)
  
  for i in range(0, 50):
    print(feature_weights[features[i]], features[i])

  for i in range(len(features)-50, len(features)):
    print(feature_weights[features[i]], features[i])
kfold(8, 0)
clean_train_corpora = process(train_corpora, lo=False, le=False)
lr_sorted_weights(clean_train_corpora)
exit()

# kfold for feature extraction ----------------------------------------------------------------------------------------------------------------------------------
def test_feature_extractors(clean_train_corpora, clean_valid_corpora):
  curr_output = []
  
  vectorizer = CountVectorizer()
  vectorizer.fit(clean_train_corpora)
  train_vec = vectorizer.transform(clean_train_corpora)
  valid_vec = vectorizer.transform(clean_valid_corpora)
  model = LogisticRegression(solver='liblinear')
  model.fit(train_vec, train_labels)
  predictions = model.predict(valid_vec)
  f1 = f1_score(predictions, valid_labels)
  curr_output.append(f1)
  print("CV", f1)

  vectorizer = TfidfVectorizer(ngram_range=(1,3))
  vectorizer.fit(clean_train_corpora)
  train_vec = vectorizer.transform(clean_train_corpora)
  valid_vec = vectorizer.transform(clean_valid_corpora)
  model = LogisticRegression(solver='liblinear')
  model.fit(train_vec, train_labels)
  predictions = model.predict(valid_vec)
  f1 = f1_score(predictions, valid_labels)
  curr_output.append(f1)
  print("TFIDF", f1)
  outputs.append(curr_output)

k = 8
for i in range(k):
  train_corpora = []
  valid_corpora = []
  train_labels = []
  valid_labels = []
  kfold(k, i)
  clean_train_corpora = process(train_corpora)
  clean_valid_corpora = process(valid_corpora)
  test_feature_extractors(clean_train_corpora, clean_valid_corpora)
output_kfold_avgs(k)
exit()

# kfold for Preprocessing ----------------------------------------------------------------------------------------------------------------------------------
def test_preprocess(clean_train_corpora, clean_valid_corpora):
  vectorizer = TfidfVectorizer(ngram_range=(1,3))
  vectorizer.fit(clean_train_corpora)
  train_vec = vectorizer.transform(clean_train_corpora)
  valid_vec = vectorizer.transform(clean_valid_corpora)
  model = LogisticRegression(solver='liblinear')
  model.fit(train_vec, train_labels)
  predictions = model.predict(valid_vec)
  return f1_score(predictions, valid_labels)

# k = 8
for i in range(k):
  train_corpora = []
  valid_corpora = []
  train_labels = []
  valid_labels = []
  curr_output = []
  kfold(k, i)
  
  clean_train_corpora = process(train_corpora)
  clean_valid_corpora = process(valid_corpora)
  f1 = test_preprocess(clean_train_corpora, clean_valid_corpora)
  print("OP", f1)
  curr_output.append(f1)

  clean_train_corpora = process(train_corpora, sw=True)
  clean_valid_corpora = process(valid_corpora, sw=True)
  f1 = test_preprocess(clean_train_corpora, clean_valid_corpora)
  print("sw", f1)
  curr_output.append(f1)

  clean_train_corpora = process(train_corpora, an=True)
  clean_valid_corpora = process(valid_corpora, an=True)
  f1 = test_preprocess(clean_train_corpora, clean_valid_corpora)
  print("an", f1)
  curr_output.append(f1)

  clean_train_corpora = process(train_corpora, lo=True)
  clean_valid_corpora = process(valid_corpora, lo=True)
  f1 = test_preprocess(clean_train_corpora, clean_valid_corpora)
  print("lo", f1)
  curr_output.append(f1)

  clean_train_corpora = process(train_corpora, le=True)
  clean_valid_corpora = process(valid_corpora, le=True)
  f1 = test_preprocess(clean_train_corpora, clean_valid_corpora)
  print("le", f1)
  curr_output.append(f1)

  clean_train_corpora = process(train_corpora, em=True)
  clean_valid_corpora = process(valid_corpora, em=True)
  f1 = test_preprocess(clean_train_corpora, clean_valid_corpora)
  print("em", f1)
  curr_output.append(f1)

  clean_train_corpora = process(train_corpora, ur=False, us=False, ha=False, ch=False, )
  clean_valid_corpora = process(valid_corpora, ur=False, us=False, ha=False, ch=False, )
  f1 = test_preprocess(clean_train_corpora, clean_valid_corpora)
  print("twit", f1)
  curr_output.append(f1)

  outputs.append(curr_output)

output_kfold_avgs(k)
exit()

# kfold for f1-score of models ----------------------------------------------------------------------------------------------------------------------------------
def test_models(train_vec, valid_vec):
  curr_output = []
  # DummyClassifier
  model = DummyClassifier()
  model.fit(train_vec, train_labels)
  predictions = model.predict(valid_vec)
  f1 = f1_score(predictions, valid_labels)
  curr_output.append(f1)
  print("Dummy Stratified", f1)

  model = LogisticRegression(solver='liblinear')
  model.fit(train_vec, train_labels)
  predictions = model.predict(valid_vec)
  f1 = f1_score(predictions, valid_labels)
  curr_output.append(f1)
  print("logreg liblinear", f1)

  # MultinomialNB
  model = MultinomialNB()
  model.fit(train_vec, train_labels)
  predictions = model.predict(valid_vec)
  f1 = f1_score(predictions, valid_labels)
  curr_output.append(f1)
  print("MNB", f1)

  # DecisionTreeClassifier
  model = DecisionTreeClassifier()
  model.fit(train_vec, train_labels)
  predictions = model.predict(valid_vec)
  f1 = f1_score(predictions, valid_labels)
  curr_output.append(f1)
  print("DecTree", f1)

  # RandomForestClassifier
  model = RandomForestClassifier(n_estimators = 100)
  model.fit(train_vec, train_labels)
  predictions = model.predict(valid_vec)
  f1 = f1_score(predictions, valid_labels)
  curr_output.append(f1)
  print("RF", f1)

  # KNeighborsClassifier
  model = KNeighborsClassifier()
  model.fit(train_vec, train_labels)
  f1 = f1_score(predictions, valid_labels)
  curr_output.append(f1)
  predictions = model.predict(valid_vec)
  print("KNN", f1)
  outputs.append(curr_output)

k = 8
for i in range(k):
  train_corpora = []
  valid_corpora = []
  train_labels = []
  valid_labels = []
  kfold(k, i)
  clean_train_corpora = process(train_corpora)
  clean_valid_corpora = process(valid_corpora)
  
  # vectorizer = CountVectorizer()
  vectorizer = TfidfVectorizer(ngram_range=(1,3))
  vectorizer.fit(clean_train_corpora)
  train_vec = vectorizer.transform(clean_train_corpora)
  valid_vec = vectorizer.transform(clean_valid_corpora)
  test_models(train_vec, valid_vec)

output_kfold_avgs(k)
exit()
