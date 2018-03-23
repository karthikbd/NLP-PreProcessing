# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 12:52:00 2018

@author: Karthikeyan
"""

##Sample nltk CFG
import nltk
gren_grammar = nltk.CFG.fromstring("""
                                   S -> NP VP
                                   PP -> P NP
                                   NP -> Det N | Det N PP | 'I'
                                   VP -> V NP | VP PP
                                   Det -> 'an' | 'my'
                                   N -> 'elephant' | 'pajamas'
                                   V -> 'shot'
                                   P -> 'in'
                                   """)
sentence = ['I', 'shot', 'an', 'elephant', 'in', 'my', 'pajamas']
parser = nltk.ChartParser(gren_grammar)
i = 0
for tree in parser.parse(sentence):
    i =i+1
    print('tree', i)
    print(tree)


##Stanford NlP setup
import os
from nltk.parse.stanford import StanfordDependencyParser
path = "F:\\Chat_bot\\stanford-corenlp-full-2016-10-31\\stanford-corenlp-full-2016-10-31\\"
path_to_jar = path + 'stanford-corenlp-3.7.0.jar'
path_to_models_jar = path + 'stanford-corenlp-3.7.0-models.jar'
dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)
os.environ['JAVA_HOME'] = 'C:\\Program Files\\Java\\jdk1.8.0_144\\bin'

##Example 1
result = dependency_parser.raw_parse("I shot an elephant on my pajamas")
dep = next(result)
for t in dep.triples():
    print(t)

#Find the root word or head word from the phrase
dep.root['word']

#Extract the depenency tree
list(dep.tree())
dep.tree().draw()

##Example 2
sentence2 = "Embedded System Developer Openings in Chennai DLK Technologies Pvt.Ltd - Chennai, Tamil Nadu Today, DLK is a key player in the field of Big Data, Embedded Analytics, Mobile, Sales Automation and Sales Performance.... Apply securely with Indeed Resume 20-Feb"
result = dependency_parser.raw_parse(sentence2)
dep = next(result)
dep.root['word']
list(dep.tree())
dep.tree().draw()

##Example 3
sentence3 = "I shot an elephant on my pajamas"
result = dependency_parser.raw_parse(sentence3)
dep = next(result)
print("Head word: ", dep.root['word'])
dep.tree().draw()

##Simple algorithm for reconstructing the sentence
import re
regexpSub = re.compile(r'subj')
regexpObj = re.compile(r'obj')
regxNoun = re.compile('^N.* | ^PR.*')
root = dep.root['word']

#A random selection of sentence with different styles domains etc
sentence4 = ["Exp: 2-5 years;(1+in analytics)",
"Expert understanding and demonstrated skills of using "R". ",
"Exposure to SAS, Python, SPSS, Julia, etc also an advantage.",
"Excellent ability to assimilate multi-disciplinary problems across industries, create hypotheses and craft solutions using data science skills and techniques. Bring together different technologies to solve a problem. Strong understanding of databases, file systems (big data stores, especially) and database/SQL languages. Very strong articulation skills.",
"Articulation skills are both oratory and written. Responsibilities The individual will be a designer of solutions that address specific business outcomes.",
"These will be across industries and functions and must be designed generically to handle reuse. Sometimes the questions will be unknown, which the individual must creatively discover and solve. Solutions will be complete packages of BI and advanced data science related models in "R". "]

def get_compund(triples, word):
    compound = []
    for t in triples:
        if t[0][0]== word:
            if regxNoun.search(t[2][1]):
                compound.append(t[2][0])
    return compound

for sentence in sentence4:
    result = dependency_parser.raw_parse(sentence)
    dep = next(result)
    root = [dep.root['word']]
    root.append(get_compund(dep.tree(), root))
    subj = []
    obj = []
    
    for t in dep.triples():
        if regexpSub.search(t[1]):
            subj.append(t[2][0])
            subj.append(get_compund(dep.triples(), t[2][0]))
        if regexpObj.search(t[1]):
            obj.append(t[2][0])
            obj.append(get_compund(dep.triples(), t[2][0])) 
    print("\n",sentence)
    print("\nSubject",subj,"\nTopic", root, "\nObject", obj)
    


















