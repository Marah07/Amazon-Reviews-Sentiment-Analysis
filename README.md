# Amazon-Reviews-Sentiment-Analysis

Overview

This project focuses on sentiment analysis of Amazon product reviews using various natural language processing (NLP) techniques and machine learning models. The goal is to develop a system that can automatically classify reviews as positive, negative, or neutral, providing valuable insights into customer sentiments.

Introduction

Amazon Reviews Sentiment Analysis is a project that explores sentiment analysis techniques applied to product reviews collected from Amazon. The project includes preprocessing steps such as language detection, tokenization, and TF-IDF transformation, followed by sentiment analysis using various methods including lexicon-based analysis, machine learning algorithms, and pre-trained transformer-based deep learning models.

Methods and Techniques

Data Preprocessing

The dataset is preprocessed using various techniques, including language detection, tokenization, stop words removal, stemming and lemmatization, TF-IDF transformation, and the creation of a Bag of Words (BoW) model.

Sentiment Analysis

NLTK Sentiment Analyzer: Utilizes lexicon-based analysis for sentiment classification.
Vader: Employs the Vader sentiment analysis tool for sentiment scoring.
TextBlob: Uses the TextBlob library for sentiment analysis.
SentiWordNet: Relies on SentiWordNet for sentiment analysis.
Machine Learning (ML): Implements logistic regression, naive Bayes, decision tree, and SVM classifiers for sentiment analysis.
Pre-trained Transformer-based Deep Learning:
Long Short-Term Memory (LSTM): Applies LSTM for sequence-based sentiment analysis.
Generative Pre-trained Transformer (GPT): Utilizes GPT for contextual understanding of reviews.

Performance Comparison

Compares the performance of various sentiment analysis algorithms using metrics such as accuracy, precision, recall, and F1 score.
