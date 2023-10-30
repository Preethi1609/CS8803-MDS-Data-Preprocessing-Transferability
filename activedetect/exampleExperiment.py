#!/usr/bin/env python
from activedetect.loaders.csv_loader import CSVLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from activedetect.experiments.Experiment import Experiment
import pandas as pd
"""
Example Experiment Script
"""

#Loads the first 100 lines of the dataset
#loaded data is a list of lists [ [r1], [r2],...,[r100]]
# c = CSVLoader()
# loadedData = c.loadFile('datasets/adult.data')[:1000]

loadedData = pd.read_csv('datasets/adult.data')[:1000]
features = loadedData.iloc[:, :-1].values.astype(str) # get features, ditch labels, convert df to numpy array
#print(loadedData.iloc[:, -1])
labels = loadedData.iloc[:, -1].map(lambda x: (x == ' <=50K') * 1.0).values
#print(features)

#all but the last column are features
#features = [l[0:-1] for l in loadedData]

#last column is a label, turn into a float
#labels = [1.0*(l[-1]==' <=50K') for l in loadedData]

#print(labels)

#run the experiment, results are stored in uscensus.log
#features, label, sklearn model, name
e = Experiment(features, labels, RandomForestClassifier(), "uscensus")
e.runAllAccuracy() #here1


