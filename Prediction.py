# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 10:18:10 2019

@author: Shriyash Shende

"""

import pandas as pd

s = pd.read_csv('Data.csv',  encoding = "ISO-8859-1")


#Splitting training and test data
train = s[s['Date'] < '20150101']
test = s[s['Date'] > '20141231']

#Data Processing
# Removing punctuations
txt_data=train.iloc[:,2:27]
txt_data.replace("[^a-zA-Z]"," ",regex=True, inplace=True)

#Renaming The column
list1= [i for i in range(25)]
new_Index=[str(i) for i in list1]
txt_data.columns= new_Index

#Converting into lower case
for index in new_Index:
    txt_data[index]=txt_data[index].str.lower()

#Combining for nlp
headlines = []
for row in range(0,len(txt_data.index)):
    headlines.append(' '.join(str(x) for x in txt_data.iloc[row,0:25]))




from sklearn.feature_extraction.text import CountVectorizer

countvector=CountVectorizer(ngram_range=(2,2))
traindataset=countvector.fit_transform(headlines)


from sklearn.ensemble import RandomForestClassifier
# implement RandomForest Classifier
randomclassifier=RandomForestClassifier(n_estimators=200,criterion='entropy')
randomclassifier.fit(traindataset,train['Label'])
test_transform= []
for row in range(0,len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_dataset = countvector.transform(test_transform)
predictions = randomclassifier.predict(test_dataset)

# check accuracy
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score



matrix=confusion_matrix(test['Label'],predictions)
print(matrix)
score=accuracy_score(test['Label'],predictions)
print(score)
report=classification_report(test['Label'],predictions)
print(report)

