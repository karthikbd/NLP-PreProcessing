# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 12:46:29 2018

@author: Karthikeyan
"""

########NLTK corpus
#Downloading and exploring the corpous

import nltk
nltk.download("reuters")

#Download additional Corpora from NLTK
nltk.download("punkt") #plunkt tokenize model
nltk.download("averaged_perceptron_tagger") #part-of-speech tokenizer
nltk.download("stopwords") #stop word

#Categories, files, words
from nltk.corpus import reuters
categories = reuters.categories()
print("No. of categories:", len(categories))
print(categories[0:9],categories[-10:])

#extracting words
words = reuters.words()
print("number of words", len(words))
print("First 11 words", words[0:10])

##filtering text by category
#Extracting specific category

tradewords = reuters.words(categories = "trade")
len(tradewords)

#remove stop words ad puntuations

from nltk.corpus import stopwords
import string
print(stopwords.words("english"))

###
tradewords = [w for w in tradewords if w.lower() not in stopwords.words('english')]

tradewords = [w for w in tradewords if w not in string.punctuation]
punctcombo = [c+"\"" for c in string.punctuation] + ["\""+c for c in string.punctuation]
tradewords = [w for w in tradewords if w not in punctcombo]
len(tradewords)

##word frequency distribution
fdist = nltk.FreqDist(tradewords)
fdist.plot(20, cumulative = False)

for word, frequency in fdist.most_common(10):
    print(word, frequency)

#Bi-grams
bitradewords = nltk.bigrams(tradewords) 
bifdist = nltk.FreqDist(bitradewords)
bifdist.plot(20, cumulative = False)

##Exploring the penn tree bank
nltk.download('treebank')
from nltk.corpus import treebank
words = treebank.words()
tagged = treebank.tagged_words()
print(type(tagged))
print("word count", len(words))
print("tagged word samples", tagged[0:9])

parsed = treebank.parsed_sents()[0]
print(parsed)
type(parsed)

import IPython
IPython.core.display.display(parsed)


