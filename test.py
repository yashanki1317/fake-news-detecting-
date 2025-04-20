# Importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string
data_fake = pd.read_csv('Fake.csv')
data_true = pd.read_csv('True.csv')
data_fake.head()
data_true.head()
data_fake["class"] = 0
data_true['class'] = 1
print (data_fake.shape, data_true.shape)
data_fake_manual_testing = data_fake.tail(10)
data_fake = data_fake.iloc[:-10]

data_true_manual_testing = data_true.tail(10)
data_true = data_true.iloc[:-10]

print(data_fake.shape, data_true.shape)
data_fake_manual_testing["class"] = 0
data_true_manual_testing['class'] = 1
print (data_true_manual_testing.head(10))
print (data_fake_manual_testing.head(10))
data_merge = pd.concat([data_fake, data_true], axis=0)
print(data_merge.shape)
print(data_merge.head())
print(data_merge.columns)
data = data_merge.drop(['title','subject','date'] , axis = 1)
print(data.isnull().sum())
data_merge = data_merge.sample(frac=1).reset_index(drop=True)
print(data.head())
print(data_merge['class'].value_counts())
print(data_merge.sample(10))
data.reset_index(inplace = True)
data.drop(['index'], axis=1, inplace = True)
print(data.columns)
print(data.head())
def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', ' ', text)  
    text = re.sub(r"\W", " ", text)  
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)  
    text = re.sub(r'‹.*?›', ' ', text)  
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)  
    text = re.sub(r'\n', ' ', text)  
    text = re.sub(r' \w*\d\w*', ' ', text)  
    return text
data['text']= data['text'].apply(wordopt)
x=data['text']
y=data['class']
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.25, random_state=42)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(X_train)
xv_test= vectorization.transform (X_test)
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit (xv_train, Y_train)
pred_lr = LR.predict (xv_test)
print (LR.score(xv_test, Y_test))
print(classification_report(Y_test, pred_lr))
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier()
print (DT.fit(xv_train, Y_train))
pred_dt = DT.predict(xv_test)
print(DT.score(xv_test, Y_test))
print(classification_report(Y_test, pred_lr))
from sklearn.ensemble import GradientBoostingClassifier
GB = GradientBoostingClassifier(random_state = 0)
GB. fit(xv_train, Y_train)
pred_gb = GB.predict(xv_test)
GB.score(xv_test, Y_test)
print(classification_report(Y_test, pred_gb))
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(random_state = 0)
RF.fit(xv_train, Y_train)
pred_rf = RF.predict(xv_test)
RF.score(xv_test, Y_test)
print(classification_report(Y_test, pred_rf))
def output_label(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"

def manual_testing(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)

    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)

    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GB = GB.predict(new_xv_test)
    pred_RF = RF.predict(new_xv_test)
    print("\n\nLR prediction: {} \nDT prediction: {} \nGB prediction: {} \nRF prediction: {}".format(
    output_label(pred_LR[0]),
    output_label(pred_DT[0]),
    output_label(pred_GB[0]),
    output_label(pred_RF[0])
))
news="NASA Confirms Earth Will Go Completely Dark for 6 Days in November 2025 Due to Solar Storm"
manual_testing(news)









 








