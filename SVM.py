# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 00:23:41 2019

@author: Abhishek
"""

# importing necessary libraries 
from sklearn import datasets 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
import pandas as pd 
from os import listdir
from sklearn.utils import shuffle
import sklearn.utils
from sklearn.naive_bayes import GaussianNB 
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn import datasets
from sklearn import svm

def find_csv_filenames( path_to_dir, suffix=".csv" ):
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]


data2 = pd.DataFrame()    

filenames = find_csv_filenames(r"./CSV")
for name in filenames:
    data = pd.read_csv(r"./CSV/"+name) 
    data.to_csv(r"./CSV/"+name)

filenames = find_csv_filenames(r"./CSV")
for name in filenames:
    data1 = pd.read_csv(r"./CSV/"+name) 
    data2=data2.append(data1)

#shuffle data
data2 = shuffle(data2)
data2 = data2.sample(frac=1).reset_index(drop=True)
data2 = sklearn.utils.shuffle(data2)
data2 = data2.reset_index(drop=True)

X = data2

y = pd.DataFrame(data=data2, columns=['y'])

X.drop(X.iloc[:, 0:1], inplace=True, axis=1)

del X['y']

sign = {0 : "No seizure", 1: "Has seizures"}
#sign = {"No chance": 0, "Low Chance": 1,"Moderate chance" : 2, "High Chance" : 3 ,"Critical": 4}
y.y = [sign[item] for item in y.y]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0) 
#print(OneVsRestClassifier(LinearSVC(random_state=0)).fit(X_train, y_train).predict(X_test))
model=svm.SVC()
model.fit(X_train, y_train)
pred=model.predict(X_test)
print("\nSVM ALGORITHM")
accuracy = accuracy_score(y_test, pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# # creating a confusion matrix 
cm = confusion_matrix(y_test, pred) 

print("Confusion matrix: \n",cm)

print("Classification Report: ")
print(classification_report(y_test, pred))
