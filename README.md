# Codsoft-
# Data science tasks given by codsoft 
# Task 1: Titanic Survival Prediction

# importing librarries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Data collection/loading and processing.

titanic_data =pd.read_csv('/content/Titanic-Dataset.csv')
titanic_data.head()

titanic_data.head(10)
titanic_data.describe()
titanic_data.tail(10)
titanic_data.shape
titanic_data.info()
titanic_data.isnull().sum()
# remove missing/null values
titanic_data = titanic_data.drop(columns='Cabin' , axis=1)
# replacing missing values with mean number
titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)
titanic_data.info()
titanic_data.isnull().sum()
# lets fix embarked
print(titanic_data['Embarked'].mode())
# replace the mode value with the missing value
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)
titanic_data.isnull().sum()
titanic_data.describe()
# how many survived?
titanic_data['Survived'].value_counts()
# visualizing data
sns.set()
sns.countplot(titanic_data['Survived'])
titanic_data['Sex'].value_counts()
# count plot for "Sex" column
sns.countplot(titanic_data['Sex'])
#  Analysing Gender wise survivors
sns.countplot(x='Sex', hue='Survived', data=titanic_data)
# count plot for "Pclass" column
sns.countplot(x='Pclass', data=titanic_data)
sns.countplot(x='Pclass', hue='Survived', data=titanic_data)
titanic_data['Sex'].value_counts()
titanic_data['Embarked'].value_counts()
titanic_data.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)
X = titanic_data.drop(columns = ['PassengerId','Name','Ticket','Survived'],axis=1)
Y = titanic_data['Survived']
print(X)
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=2)
print(X.shape, X_train.shape,X_test.shape)
model = LogisticRegression()
# use the train data on logisticregression model
model.fit(X_train, Y_train)
X_train_prediction = model.predict(X_train)
print(X_train_prediction)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of training data : ', training_data_accuracy)
# check accuracy of test data
X_test_prediction = model.predict(X_test)
print(X_test_prediction)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score of test data:', test_data_accuracy)









