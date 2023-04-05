# Multi-Label-Classification-of-Hindi-news-articles

This project is a research project aimed at building a machine learning model to classify Hindi news articles based on their content. The dataset used for training and testing the model consists of Hindi news articles with 14 labels. The goal of the project is to predict the labels of news articles (an article can be assigned multiple labels) based on their content and calculate the accuracy of the model for each label.

## Dataset
The dataset used in the project consists of 5300 Hindi news articles with 14 labels. The dataset was divided into training and testing modules. The training module was used to train the model, and the testing module was used to evaluate the performance of the model. The accuracy results for each label have been uploaded as a PNG file.

## Libraries Used
Libraries used in the project are:

* Pandas
* Numpy
* Scikit-Learn
* NLTK

## Repository Contents
The repository contains the following files:

* feature_extraction.py: A Python script for feature extraction using the tf-idf method and storing the extracted tokens for each label in an Excel file.
* calculate_tf_idf_score.py: A Python script for calculating the tf-idf score of articles for each label using the methodology explained later.
* binary_relevance.py: A Python script containing the implementation of binary relevance using logistic regression and calculating accuracy.
* function.py: A Python script containing the function definitions for every function used in other files.
* stopword.csv: A CSV file containing list of Hindi Language stopwords.
* accuracy_result.png: A png file showing accuracy for various labels achieved after implementing the model.
* README.md: This file, containing information about the project.

## Methodology
The feature_extraction.py script was used to extract features from the dataset using the tf-idf method. The script calculates the tf-idf values of tokens for each label and stores the extracted tokens for each label according to their tf-idf scores in an Excel file. This involves loading the dataset as a dataframe object, tokenization of articles, removal of stopwords and punctuations, calculation of tf, calculation of idf, then multiplying both to get tf-idf values for tokens, and finally extracting tokens for each label based on their tf-idf values.

The calculate_tf_idf_score.py script was used to calculate the tf-idf score of articles for each label. The tf-idf score of a particular label for an article was calculated by summing the tf-idf values for each token(for that particular label) of the article and summing the value for each token.

The script includes a file binary_relevance.py to apply binary relevance on the testing dataset using tf-idf scores of articles for each label as training parameters. It uses logical regression for applying binary relevance and finally calculates accuracy for each label.


## Conclusion
This project demonstrates how machine learning techniques can be used to classify Hindi news articles based on their content. The tf-idf method and binary relevance with logistic regression were used to achieve good performance on the dataset. This research project provides the accuracy of the model for each label and can be used as a benchmark for future work in the field.
