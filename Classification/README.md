# Classification Notebook

This notebook demonstrates various classification algorithms applied to the "Social_Network_Ads" dataset. The algorithms used include Support Vector Machine (SVM), K-Nearest Neighbors (K-NN), Kernel SVM, Naive Bayes, Decision Tree, and Random Forest Classification. Each section involves importing necessary libraries, loading the dataset, splitting it into training and test sets, applying feature scaling, training the model, making predictions, evaluating the model using a confusion matrix, and visualizing the results.

## How to Run

You can open and run this notebook in Google Colab by clicking [here](https://colab.research.google.com/).

## Table of Contents

1. [Importing the Libraries](#importing-the-libraries)
2. [Importing the Dataset](#importing-the-dataset)
3. [Splitting the Dataset into the Training set and Test set](#splitting-the-dataset-into-the-training-set-and-test-set)
4. [Feature Scaling](#feature-scaling)
5. [Classification Algorithms](#classification-algorithms)
   - [Linear Support Vector Machine (SVM)](#linear-support-vector-machine-svm)
   - [K-Nearest Neighbors (K-NN)](#k-nearest-neighbors-k-nn)
   - [Kernel SVM](#kernel-svm)
   - [Naive Bayes](#naive-bayes)
   - [Decision Tree](#decision-tree)
   - [Random Forest](#random-forest)
6. [Visualizing the Results](#visualizing-the-results)

## Importing the Libraries

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.colors import ListedColorma
```

## importing the dataset

```python

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
```




