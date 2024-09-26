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

## splitting the dataset into the training set and test set

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
```

##feature scaling

```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

## classification algorithms

## linear support vector machine svm

```python
from sklearn.svm import SVC
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_train, y_train)
```

## K-Nearest Neighbors (K-NN)

```python
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_train, y_train)
```

## Kernel SVM

```python
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, y_train)
```

## Naive Bayes

```python
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
```

## Decision Tree

```python
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)
```

## Random Forest

```python
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)
```

## Visualizing the Results

## Visualizing Training Set Results

```python
X_set, y_set = sc.inverse_transform(X_train), y_train

X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.5),
                     np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=100))

Z = classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape)

plt.contourf(X1, X2, Z, alpha=0.6, cmap=ListedColormap(('salmon', 'dodgerblue')))

colors = ListedColormap(('red', 'blue'))
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                color=colors(i), label=j, edgecolor='k')

plt.title('Classifier (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
```

## Visualizing Test Set Results

```python
X_set, y_set = sc.inverse_transform(X_test), y_test

X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.5),
                     np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=100))

Z = classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape)

plt.contourf(X1, X2, Z, alpha=0.6, cmap=ListedColormap(('salmon', 'dodgerblue')))

colors = ListedColormap(('red', 'blue'))
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                color=colors(i), label=j, edgecolor='k')

plt.title('Classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
```


