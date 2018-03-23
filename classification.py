# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 12:10:28 2018

@author: Karthikeyan
"""

#load Data
import pandas as pd
import numpy as np
import os

os.getcwd()
CODELOC = "F:\\Chat_bot\\NLPBot\\"
sentence = pd.read_csv('sentences.csv')

sentence.head(10)
sentence.shape

##feature engineering
#Extracting some parts of POS sequence
import nltk
from nltk import word_tokenize
list_of_triple_string = []
sentence = "Can a dog see in colour?"
sentenceparse = word_tokenize(sentence)
pos_tag = nltk.pos_tag(sentenceparse)
pos = [i[1] for i in pos_tag]
print("word  mapped to part of speech tag: ", pos_tag)
print("pos_tag: ", pos)

n = len(pos)
for i in range(0, n-3):
    t = ".".join(pos[i:i+3])
    list_of_triple_string.append(t)
    
print("Sequence of triples", list_of_triple_string)

#Extracting features
import sys
sys.path.append(CODELOC)
import features

sentence = "Can a dog see in colour?"
sentence = features.strip_sentence(sentence)
print(sentence)
pos = features.get_pos(sentence)
triples = features.get_triples(pos)

print(triples)

#Dictionary of features
sentence = ["Sorry, I don't know about the weather.",
            "That is a tricky question to answer.",
            "What does OCM stand for",
            "MAX is a Mobile Application Accelerator",
            "Can a dog see in colour?",
            "how are you"
            ]
id = 1
for s in sentence:
    features_dict = features.features_dict(str(id), s)
    features_string,header = features.get_string(str(1), s)
    print(features_dict)
    
    id += 1

#Building a machine learning model
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("featuresDump.csv")
print(str(len(df)), "rows loaded")

#Strip any leading space from col names
df.columns = df.columns[:].str.strip() 
df['class'] = df['class'].map(lambda x:x.strip())
width = df.shape[1]

#Test-Train split
np.random.seed(seed = 12)
df['is_train']= np.random.uniform(0, 1, len(df)) <= .75
train, test = df[df['is_train'] == True], df[df['is_train'] == False]
print(str(len(train)), "rows split into train", str(len(test)), "rows split into test")
features = df.columns[1:width-1]
print("FEATURES = {}".format(features))

##Fit a model with training data
#fit on rf model 

clf = RandomForestClassifier(n_jobs = 2, n_estimators= 100)
clf.fit(train[features], train['class'])

#Predict on test values

preds = clf.predict(test[features])
predout = pd.DataFrame({'id': test['id'], 'Prediction' : preds, 'actual' : test['class']})

##Basic validation
#Cross_check accuracy
print(pd.crosstab(test['class'], preds, rownames = ['actual'], colnames = ['preds']))
print("\n", pd.crosstab(test['class'], preds, rownames = ['actual'], 
                        colnames = ['preds']).apply(lambda r: round(r/r.sum()*100,2),axis = 1))

from sklearn.metrics import accuracy_score
print("\n\n Accuracy_score: ", round(accuracy_score(test['class'],preds),3))

#Load sentence data & generate features
FNAME = "F:\\Chat_bot\\NLPBot\\analysis\\pythonFAQ.csv"
import csv
import hashlib
import features

fin = open(FNAME, 'rt')
reader = csv.reader(fin)

keys = ["id",
        "wordCount",
        "stemmedCount",
        "stemmedEndNN",
        "CD",
        "NN",
        "NNP",
        "NNPS",
        "NNS",
        "PRP",
        "VBG",
        "VBZ",
        "startTuple0",
        "endTuple0",
        "endTuple1",
        "endTuple2",
        "verbBeforeNoun",
        "qMark",
        "qVerbCombo",
        "qTripleScore",
        "sTripleScore",
        "class"]
rows =[]
next(reader)
for line in reader:
    sentence = line[0]
    c = line[1]
    id = hashlib.md5(str(sentence).encode('utf-8')).hexdigest()[:16]
    
    f = features.features_dict(id, sentence, c)
    row = []
    
    for key in keys:
        value = f[key]
        row.append(value)
    row.append(row)
faq = pd.DataFrame(rows, columns = keys)
fin.close()

#predict agaist FAQ test model
featuresNames = faq.columns[1:width-1]
faqPreds = clf.predict(faq[featuresNames])


#Adhoc testing
testout = {'Q': 'QUESTIONS', 'C': 'CHAT', 'S': 'STATEMENT'}
mysentence1 = 'what is your name?'
mysentence2 = 'This is house'
mysentence3 = 'Is the cat Dead'

myFeatures = features.features_dict('1', mysentence2, 'x')
values = []
for key in keys:
    values.append(myFeatures[key])
    
s = pd.Series(values)
width = len(s)
myFeatures = s[1:width-1]
predict = clf.predict([myFeatures])    
print("\n\n Prediction is: ", testout[predict[0].strip()])












































