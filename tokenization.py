# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 14:43:30 2018

@author: Karthikeyan
"""
##Tokenizing and tagging text
#Sentence tokenizing
import nltk
passage = " Senior Analyst/Fellow - Data Science - Data Analytics McKinsey & Company  328 reviews  - Chennai, Tamil Nadu Our Data Analytics team provides analytics insights to consulting teams and clients across the globe. The team is predominantly composed of data scientists and... 19-Feb Project Manager Big Data Analytics Wipro LTD  9,249 reviews  - Chennai, Tamil Nadu Mandatory Skills: Infra SI Project Management Job Description: Key skills required for the job are: Infra SI Project Management-L2 (Mandatory) As a Project... 20-Feb Reporting Analyst HP  9,140 reviews  - Chennai, Tamil Nadu Basic knowledge of data science & analysis. Help team members learn new analytics tools and techniques. Analysis of backlog Gathers demand data from country and... 19-Feb Big Data Developer Capgemini: RO - Chennai, Tamil Nadu Implement IT Data Governance (Update Data Dictionary and Meta-model, enrich data lineage, develop data quality rules provided by Data Management officers).... 19-Feb"
doc = nltk.sent_tokenize(passage)
for s in doc:
    print('>', s)


#Word tokenizer
from nltk import word_tokenize

sentence = "Senior Analyst/Fellow - Data Science - Data Analytics McKinsey & Company  328 reviews  - Chennai, Tamil Nadu Our Data Analytics team provides analytics insights to consulting teams and clients across the globe. The team is predominantly composed of data scientists and... 19-Feb"

#default tokenizer
tree_tokens = word_tokenize(sentence)

#punct_tokens
punct_tokenizer = nltk.tokenize.WordPunctTokenizer()
punct_tokens = punct_tokenizer.tokenize(sentence)

space_tokenizer = nltk.tokenize.SpaceTokenizer()
space_tokens = space_tokenizer.tokenize(sentence)

print('Default : ', tree_tokens)
print('Punct : ', punct_tokens)
print('Space : ', space_tokens)

#Part of speech tagging
pos_tree = nltk.pos_tag(tree_tokens)
print(pos_tree)
pos_space = nltk.pos_tag(space_tokens)
print(pos_space)

#printing the noun
import re
regex = re.compile("^N.*")
noun = []
for l in pos_tree:
    if regex.match(l[1]):
        noun.append(l[0])
print("Nouns : ", noun)


##Stemming aand lematization
#Stemming
porter = nltk.PorterStemmer()
lancaster = nltk.LancasterStemmer()
snow_ball = nltk.stem.snowball.SnowballStemmer('english')

print([porter.stem(t) for t in tree_tokens])
print([lancaster.stem(t) for t in tree_tokens])
print([snow_ball.stem(t) for t in tree_tokens])

#Sentence 2
sentence2 = "Embedded System Developer Openings in Chennai DLK Technologies Pvt.Ltd - Chennai, Tamil Nadu Today, DLK is a key player in the field of Big Data, Embedded Analytics, Mobile, Sales Automation and Sales Performance.... Apply securely with Indeed Resume 20-Feb"
tokens2 = word_tokenize(sentence2)
print("\n",sentence2)
for stemmer in [porter, lancaster, snow_ball]:
    print([stemmer.stem(t) for t in tokens2])

#lematizing
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

#word_net_lemmatization
wnl = nltk.WordNetLemmatizer()
tokens2_pos = nltk.pos_tag(tokens2)
print([wnl.lemmatize(t) for t in tree_tokens])
print([wnl.lemmatize(t) for t in tokens2])







