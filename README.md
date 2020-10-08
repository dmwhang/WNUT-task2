# Dartmouth CS at WNUT-2020 Task 2: Informative COVID-19 Tweet Classification Using BERT

## Authors
Dylan Whang, Dartmouth College, Hanover, NH, dylanmwhang@gmail.com

Soroush Vosoughi, Dartmouth College, Hanover, NH, soroush@dartmouth.edu

## Abstract

We describe the systems developed for the WNUT-2020 shared task 2, identification of informative COVID-19 English Tweets. BERT is a highly performant model for Natural Language Processing tasks. We increased BERT’s performance in this classification task by fine-tuning BERT and concatenating its embeddings with Tweet-specific features and training a Support Vector Machine (SVM) for classification (henceforth called BERT+). We compared its performance to a suite of machine learning models. We used a Twitter specific data cleaning pipeline and word-level TF-IDF to extract features for the non-BERT models. BERT+ was the top performing model with an F1-score of 0.8713.

## Usage

