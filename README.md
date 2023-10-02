# Comparative Analysis of Machine Learning and Deep Learning Models for Question Pair Similarity Assessment: Insights from the Quora Question Pair Challenge

This repository contains the code for implementing various Machine Learning and Deep Learning models to solve the Quora Duplicate Question Pairs Identification challenge. The goal of this project is to determine whether two questions are duplicates or convey the same intent.

## Table of Contents:

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Models](#models)
- [Preprocessing](#preprocessing)
- [Requirements](#requirements)
- [Usage](#usage)
- [References](#references)

## Introduction

The Quora Duplicate Question Pairs Identification task is a fundamental problem in Natural Language Processing (NLP). It involves assessing the similarity between pairs of questions, even when their phrasing may differ. Solving this challenge is crucial for various NLP applications, including information retrieval, chatbot development, and recommendation systems.

This project explores the use of different Machine Learning and Deep Learning models to tackle the Quora Duplicate Question Pairs Identification task. The code provided here includes implementations of models such as Logistic Regression, Support Vector Machine (SVC), Convolutional Neural Network (CNN), Recurrent Neural Network (RNN), and transformer-based models like BERT.

## Dataset

The dataset used for this project can be found on Kaggle at [Quora Question Pairs](https://www.kaggle.com/quora/question-pairs). It consists of pairs of questions from Quora, with labels indicating whether they are duplicate or not.

## Models

We have implemented and compared various models, including:

- Logistic Regression
- Support Vector Machine (SVC)
- Convolutional Neural Network (CNN)
- Recurrent Neural Network (RNN)
- BERT-based Transformers

[Add the names of the other models you've implemented]

## Preprocessing

Data preprocessing is a crucial step in NLP tasks. In this project, we have applied preprocessing techniques such as tokenization, stop-word removal, and TF-IDF feature extraction to prepare the data for model training.

## Requirements

To run the code in this repository, you'll need the following dependencies:

- `fuzzywuzzy`: For string matching and similarity computation.
- `distance`: For string similarity and distance metrics.
- `tqdm`: For adding progress bars to your code execution.
- `nltk`: For natural language processing tasks, including stopwords removal.

You can install these dependencies using `pip`:

```bash
pip install fuzzywuzzy
pip install distance
pip install tqdm
pip install nltk
```

Additionally, download the NLTK stopwords data by running the following command:

```bash
python -c "import nltk; nltk.download('stopwords')"
```

## Usage

- Install the required dependencies as mentioned in the "Requirements" section.
- Execute the code sequentially by running the Jupyter Notebook files:
- ResearchProject.ipynb: This notebook contains the implementation of various Machine Learning and Deep Learning models.
- BertBasedModel.ipynb: This notebook focuses on the implementation of a BERT-based transformer model.

Ensure you have the necessary datasets and files in the appropriate directories before running the notebooks.

## References
- S. Chouhan (Aug 31, 2021). The Quora Question Pair Similarity Problem. [Online] Available: https://towardsdatascience.com/the-quora-question-pair-similarity-problem-3598477af172 (Accessed on: 02-Aug-2023)
- S. Kumar (2022). Quora Duplicate Question Pairs Identification. [Online] Available: https://www.kaggle.com/code/sayamkumar/quora-duplicate-question-pairs-identification (Accessed on: 07-Aug-2023)
