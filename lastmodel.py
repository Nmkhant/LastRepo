# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 11:03:13 2021

@author: ASUS
"""

import nltk
import pandas as pd
import numpy as np
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import altair as alt
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer=WordNetLemmatizer()
stemming = PorterStemmer()
stops = set(stopwords.words("english"))

train_data = pd.read_csv("train.csv")

train_original = train_data.copy()

train_original.tweet = train_original.tweet.astype(str)

def remove_pattern(text,pattern):
    
    # re.findall() finds the pattern i.e @user and puts it in a list for further task
    r = re.findall(pattern,text)
    
    # re.sub() removes @user from the sentences in the dataset
    for i in r:
        text = re.sub(i,"",text)
    
    return text

train_original['tweet'] = np.vectorize(remove_pattern)(train_original['tweet'], "@[\w]*") 

def apply_cleaning_function_to_list(X):
    cleaned_X = []
    for element in X:
        cleaned_X.append(clean_text(element))
    return cleaned_X

def clean_text(raw_text):
     # Convert to lower case
    text = raw_text.lower()
    
    # Tokenize
    tokens = nltk.word_tokenize(text)
    
    # Keep only words (removes punctuation + numbers)
    # use .isalnum to keep also numbers
    token_words = [w for w in tokens if w.isalpha()]
    
    # Stemming
    #stemmed_words = [stemming.stem(w) for w in token_words]
    # lemmatizing
    lemmatized_words=[lemmatizer.lemmatize(word) for word in token_words]
    
    # Remove stop words
    meaningful_words = [w for w in lemmatized_words if not w in stops]
    # Rejoin meaningful stemmed words
    joined_words = ( " ".join(meaningful_words))
    
    # Return cleaned data
    return joined_words

text_to_clean =list(train_original['tweet'])
cleaned_text = apply_cleaning_function_to_list(text_to_clean)

all_words_positive = ' '.join(text for text in train_original['tweet'][train_original['label']==0])

all_words_negative = ' '.join(text for text in train_original['tweet'][train_original['label']==1])


def Hashtags_Extract(x):
    hashtags=[]
    
    # Loop over the words in the tweet
    for i in x:
        ht = re.findall(r'#(\w+)',i)
        hashtags.append(ht)
    
    return hashtags

ht_positive = Hashtags_Extract(train_original['tweet'][train_original['label']==0])

ht_positive_unnest = sum(ht_positive,[])

ht_negative = Hashtags_Extract(train_original['tweet'][train_original['label']==1])

ht_negative_unnest = sum(ht_negative,[])

word_freq_positive = nltk.FreqDist(ht_positive_unnest)
word_freq_negative = nltk.FreqDist(ht_negative_unnest)

df_positive = pd.DataFrame({'Hashtags':list(word_freq_positive.keys()),'Count':list(word_freq_positive.values())})
df_negative = pd.DataFrame({'Hashtags':list(word_freq_negative.keys()),'Count':list(word_freq_negative.values())})

df_negative_plot = df_negative.nlargest(20,columns='Count') 
df_positive_plot = df_positive.nlargest(20, columns = 'Count')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=CountVectorizer()
vectorizer.fit(cleaned_text)

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test =train_test_split(cleaned_text,train_original['label'],train_size=0.75,test_size=0.25,random_state=42,shuffle=True) 

Tfidf_vect=TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(cleaned_text)
Train_X_Tfidf=Tfidf_vect.transform(X_train)
Test_X_Tfidf=Tfidf_vect.transform(X_test)

from sklearn.naive_bayes import MultinomialNB
nb=MultinomialNB()
nb.fit(Train_X_Tfidf,y_train)
nb_pred = nb.predict(Test_X_Tfidf)

#from xgboost import XGBClassifier
#xgb = XGBClassifier(random_state=29,learning_rate=0.7)
#xgb.fit(Train_X_Tfidf, y_train)
#xgb_pred = xgb.predict(Test_X_Tfidf)

from sklearn.tree import DecisionTreeClassifier
dct = DecisionTreeClassifier(criterion='entropy', random_state=1)
dct.fit(Train_X_Tfidf, y_train)
dct_pred = dct.predict(Test_X_Tfidf)

from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

nb_accuracy = accuracy_score(y_test, nb_pred)*100
nb_Precision_Macro_Score = precision_score(y_test, nb_pred,average = 'macro')*100
nb_Recall_Score = recall_score(y_test, nb_pred, average = 'macro')*100
nb_F1_score =f1_score(y_test, nb_pred)*100

#xgb_accuracy = accuracy_score(y_test, xgb_pred)*100
#xgb_Precision_Macro_Score = precision_score(y_test, xgb_pred,average = 'macro')*100
#xgb_Recall_Score = recall_score(y_test, xgb_pred, average = 'macro')*100
#xgb_F1_score =f1_score(y_test, xgb_pred)*100
    
dct_accuracy = accuracy_score(y_test, dct_pred)*100
dct_Precision_Macro_Score = precision_score(y_test, dct_pred,average = 'macro')*100
dct_Recall_Score = recall_score(y_test, dct_pred, average = 'macro')*100
dct_F1_score =f1_score(y_test, dct_pred)*100
    
import pickle
modelsave = 'final_model.h5'
pickle.dump(nb, open(modelsave, 'wb')) 

def testing(text_input):
    result=(nb.predict(Tfidf_vect.transform([text_input]))[0])
    if result==1:
        return result
    elif result==0:
        return result
    
df_hashtag_positive = pd.DataFrame({
    'Positive-Hashtags': df_positive_plot['Hashtags'],
    'Count': df_positive_plot['Count']
     })
a = alt.Chart(df_hashtag_positive).mark_bar(opacity=1).encode(
y='Positive-Hashtags', x='Count')

df_hashtag_negative = pd.DataFrame({
    'Negative-Hashtags': df_negative_plot['Hashtags'],
    'Count': df_negative_plot['Count']
     })
b = alt.Chart(df_hashtag_negative).mark_bar(opacity=1).encode(
y='Negative-Hashtags', x='Count')

df_f1score = pd.DataFrame({
    'Model': ['Naive Bayes', 'Decision Tree', 'XGBClassifier'],
    'F1score': [nb_F1_score , dct_F1_score, 54.3123543123543 ] 
    })

df_accuracy = pd.DataFrame({
    'Model': ['Naive Bayes', 'Decision Tree', 'XGBClassifier'],
    'Accuracy': [nb_accuracy, dct_accuracy ,95.09448129145288]
    })

df_precision = pd.DataFrame({
    'Model': ['Naive Bayes', 'Decision Tree', 'XGBClassifier' ],
    'Precision Macro Score': [nb_Precision_Macro_Score, dct_Precision_Macro_Score ,86.84412593895061]
    })

df_recall  = pd.DataFrame({
    'Model': ['Naive Bayes', 'Decision Tree', 'XGBClassifier' ],
    'Recall Score': [nb_Recall_Score, dct_Recall_Score, 70.39676128562652]
    })

f1 = alt.Chart(df_f1score).mark_bar(opacity=1).encode(
y='F1score', x='Model').properties(height=400,width=250)

acc = alt.Chart(df_accuracy).mark_bar(opacity=1).encode(
y='Accuracy', x='Model').properties(height=400,width=250)

pre = alt.Chart(df_precision).mark_bar(opacity=1).encode(
y='Precision Macro Score', x='Model').properties(height=400,width=250)

recall = alt.Chart(df_recall).mark_bar(opacity=1).encode(
y='Recall Score', x='Model').properties(height=400,width=250)
