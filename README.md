# Dartmouth CS at WNUT-2020 Task 2: Informative COVID-19 Tweet Classification Using BERT

## Authors
Dylan Whang, Dartmouth College, Hanover, NH, dylanmwhang@gmail.com

Soroush Vosoughi, Dartmouth College, Hanover, NH, soroush@dartmouth.edu

## Abstract

We describe the systems developed for the WNUT-2020 shared task 2, identification of informative COVID-19 English Tweets. BERT is a highly performant model for Natural Language Processing tasks. We increased BERT’s performance in this classification task by fine-tuning BERT and concatenating its embeddings with Tweet-specific features and training a Support Vector Machine (SVM) for classification (henceforth called BERT+). We compared its performance to a suite of machine learning models. We used a Twitter specific data cleaning pipeline and word-level TF-IDF to extract features for the non-BERT models. BERT+ was the top performing model with an F1-score of 0.8713.

## Usage

The `preprocess.py` file consists of methods used to preprocess, clean, and generate feature from the input Tweets.

The `analyze.py` file utilizes traditional ML models to classify the tweets found in the `.tsv` files.

The `Dartmouth_CS_at_WNUT_2020_Task_2_Fine_tuning_BERT_for_Tweet_classification.ipynb` notebook utilzes Google's BERT to make a fine tuned model for prediction of tweets.
