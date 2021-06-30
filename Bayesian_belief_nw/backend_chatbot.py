import pandas as pd
import pyttsx3
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier,_tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import csv
# Program to measure the similarity between 
# two sentences using cosine similarity.
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

df = pd.read_csv('Training.csv')
description = pd.read_csv('symptom_Description.csv', header = None)
precaution = pd.read_csv('symptom_precaution.csv', header = None)
severity = pd.read_csv('Symptom_severity.csv', header = None)
notes = pd.read_csv('disease_notes.csv',encoding='latin1')

severity[0] = severity[0].str.replace('[_]', ' ')
df.columns = df.columns.str.replace('[_]', ' ')

def calc_condition(exp,days):
    sum=0
    days = int(days)
    for item in exp:
        sum=sum+list(severity[1][severity[0]==item])[0]
        print("item:"+str(item)+" sum:"+str(sum))
    if((sum*days)/(len(exp)+1)>13):
        print("You should take the consultation from doctor. ")
    else:
        print("It might not be that bad but you should take precautions.")


#updated
print("Symptom?")
sym_in = input()
X =sym_in
X = X.lower()
cols = list(df.columns)
cosine_scores = {}
sym_exp = []
l=[]
# sw contains the list of stopwords
sw = stopwords.words('english') 
# tokenization
X_list = word_tokenize(X)
for i in cols:
    Y=i
    Y_list = word_tokenize(Y)
    l1 =[];l2 =[]
  
    # remove stop words from the string
    X_set = {w for w in X_list if not w in sw} 
    Y_set = {w for w in Y_list if not w in sw}

    # form a set containing keywords of both strings 
    rvector = X_set.union(Y_set) 
    for w in rvector:
        if w in X_set: l1.append(1) # create a vector
        else: l1.append(0)
        if w in Y_set: l2.append(1)
        else: l2.append(0)
    c = 0

    # cosine formula 
    for i in range(len(rvector)):
            c+= l1[i]*l2[i]
    cosine = c / float((sum(l1)*sum(l2))**0.5)
    cosine_scores[Y] = cosine
    l.append(cosine)
    
cosine_scores
sorted_dict = {}
sorted_keys = sorted(cosine_scores, key=cosine_scores.get, reverse=True)  # [1, 3, 2]

for w in sorted_keys:
    sorted_dict[w] = cosine_scores[w]

df1 = df.copy()

sym=''
reply1=''

#print(sorted_dict)
for j in range(len(sorted_dict)):
    print(list(sorted_dict.keys())[j]+'?')
    reply1 = input()
    if(reply1 == 'yes'):
        sym = list(sorted_dict.keys())[j]
        sym_exp.append(sym)
        break
print('For how many days?')
days = input()

df1 = df1[df1[sym]==1]
df1.drop(sym, axis=1, inplace=True)
df1 = df1.loc[:, (df1 != 0).any(axis=0)]
questions = list(df1.sum(axis = 0, skipna = True)[:-1].sort_values().index)
diag=list(df1['prognosis'].unique())
dg = []
while(1):
    print('Do you have '+questions[-1]+'?')
    response = input()
    if(response == 'yes'):
        df1 = df1[df1[questions[-1]]==1]
        sym_exp.append(questions[-1])
    df1.drop(questions[-1], axis=1, inplace=True)
    df1 = df1.loc[:, (df1 != 0).any(axis=0)] 
    questions = list(df1.sum(axis = 0, skipna = True)[:-1].sort_values().index)
    try:
        dg = list(df1['prognosis'].unique())
        diag = diag+dg
    except:
        print("Done")
    
    if(len(questions)<4 or len(dg)<2):
        break
print("\nDiagnosis:")
frequency = {}

# iterating over the list
for item in diag:
    # checking the element in dictionary
    if item in frequency:
        # incrementing the counr
        frequency[item] += 1
    else:
        # initializing the count
        frequency[item] = 1
        
sorted_freq = {}
sorted_key = sorted(frequency, key=frequency.get, reverse=True)  # [1, 3, 2]
for w in sorted_key:
    sorted_freq[w] = frequency[w]
    
for k in sorted_freq.keys():
    print(k)

dis_diag = sorted_freq.keys()
#lead = dis_diag[0]
print("\nTriage Level:")
print(calc_condition(sym_exp, days))

print('\nPrescription:')
print(list(notes['Medication'][notes['prognosis']==list(sorted_freq.keys())[0]])[0])

print('\nPrecautions:')
for p in range(1, 5):
    print(list(precaution[p][precaution[0]==list(sorted_freq.keys())[0]])[0])

print('\nTests you need:')
print(list(notes['Diagnosis'][notes['prognosis']==list(sorted_freq.keys())[0]])[0])

