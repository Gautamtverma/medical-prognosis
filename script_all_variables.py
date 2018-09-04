# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 01:48:38 2018

@author: Gautam
"""
#importing the relevant packages
from scipy.stats import pearsonr
import numpy as np
from scipy.stats import chi2
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import chi2_contingency
import gc

#importing the dataset into spyder environment
df1 = pd.read_csv('C:\Gautam\Dell\Gtm\Personal Documents\data\Case Study 1\data.csv')

df1.shape
# checking the datatypes of the features
df1.dtypes

# 1. Checking for Missing values in the dataset-
df1.isnull().sum()

#calculate percentage of missing values in the column- alcohol_consumption_per_day
df1['alcohol_consumption_per_day'].isnull().sum()/df1.shape[0]

#calculate percentage of missing values in the column- Pregnancy
df1['Pregnancy'].isnull().sum()/df1.shape[0]

#calculate percentage of missing values in the column- Genetic_Pedigree_Coefficient
df1['Genetic_Pedigree_Coefficient'].isnull().sum()/df1.shape[0]

# removing the column "Pregnancy" as the missing value percentage is about 78%
df1=df1.drop(['Pregnancy'],axis=1)


#imputing the mean of the respective columns for these continuous variables
df1["alcohol_consumption_per_day"].fillna(df1["alcohol_consumption_per_day"].mean(), inplace=True)
df1["Genetic_Pedigree_Coefficient"].fillna(df1["Genetic_Pedigree_Coefficient"].mean(), inplace=True)

#checking the completion of the missing value treatment 
df1.isnull().sum()

#Univariate Analysis on the individual features
df1.describe()

# Evaluating class imbalance in the class label percentage
df1['Blood_Pressure_Abnormality'].value_counts()


#Bivariate Analysis on the features
#check correlation in the independent variables
corr = df1.corr()
sns.heatmap(df1.corr(),xticklabels=corr.columns.values,yticklabels=corr.columns.values)

df1.boxplot(column=['Level_of_Hemoglobin'],by=['Blood_Pressure_Abnormality'])


#Goodness of fit statistic
def chisq_of_df_cols(df1, c1, c2):
    groupsizes = df1.groupby([c1, c2]).size()
    ctsum = groupsizes.unstack(c1)
    # fillna(0) is necessary to remove any NAs which will cause exceptions
    return(chi2_contingency(ctsum.fillna(0)))

#evaluate correlation between different categorical variables including the dependent and independent variables
chi2, p, dof, expected=chisq_of_df_cols(df1, 'Smoking', 'Level_of_Stress')


#convert the dataframe into array
data1 = df1.values
data1.dtype

#creating the independent feature array
A=data1[:,2:15]
A_names=['Level_of_Hemoglobin', 'Genetic_Pedigree_Coefficient', 'Age', 'BMI', 'Sex', 'Smoking', 'Physical_activity', 'salt_content_in_the_diet', 'alcohol_consumption_per_day', 'Level_of_Stress', 'Chronic_kidney_disease', 'Adrenal_and_thyroid_disorders']

#creating the dependent feature array
B=data[:,1:2]
B_name=['Blood_Pressure_Abnormality']

#evaluating multicollinearity in the dataset
pearsonr(A[:,10],A[:,11])


#Feature Selection technique to shortlist most important features
#using f_classif algo because of categorical target variable (ANOVA) 
skb1 = SelectKBest(score_func=f_classif, k=10)
skb1.fit(A,B)
print("scores_:",skb1.scores_)
print("pvalues_:",skb1.pvalues_)
print("selected index:",skb1.get_support(True))
print("after transform:",skb1.transform(A)) 

selected_indexes = skb1.get_support(True)

# Create new dataframe with only shortlisted columns
A_select = A[:,selected_indexes]

#assigning names to the shortlisted array
A_select_names=df1.dtypes.index[skb1.get_support(True)]

print("The shortlisted features",A_select_names)

#validating the correct creation of the shortlisted independent variables
A_select.shape

#creating training and test data for A and B Variables using 70 : 30 random split
A_train, A_test, B_train, B_test = train_test_split(A_select, B, test_size=0.30)

#initializing the ensemble model- Random Forest
model_new = RandomForestClassifier()

#fitting the model on the training data 
model_new.fit(A_train,B_train)

#evaluating the model performance for underfitting and overfitting
#checking the model performance on train data first
B_pred1=model_new.predict(A_train)
cm=confusion_matrix(B_train, B_pred1)
print(cm)
print(classification_report(B_train, B_pred1))
model_new.score(A_train, B_train)


#checking the model performance on test data next to form an opinion on bias and variance
B_pred=model_new.predict(A_test)
cm=confusion_matrix(B_test, B_pred)
print(cm)
print(classification_report(B_test, B_pred))
model_new.score(A_test, B_test)


#Cross validation of the model
c,r=B.shape
B=B.reshape(c,)
scores = cross_val_score(model_new, A, B, cv=6)
print (scores)

gc.collect()

test_data= pd.read_csv('C:\Gautam\Dell\Gtm\Personal Documents\data\Case Study 1\sample.csv')

test_data.head()

test_data["alcohol_consumption_per_day"].fillna(test_data["alcohol_consumption_per_day"].mean(), inplace=True)
test_data["Genetic_Pedigree_Coefficient"].fillna(test_data["Genetic_Pedigree_Coefficient"].mean(), inplace=True)
test_data=test_data.drop(['Pregnancy'],axis=1)

test = test_data.values
test.shape

A_test=test[:,2:15]
A_test_names=['Level_of_Hemoglobin', 'Genetic_Pedigree_Coefficient', 'Age', 'BMI', 'Sex', 'Smoking', 'Physical_activity', 'salt_content_in_the_diet', 'alcohol_consumption_per_day', 'Level_of_Stress', 'Chronic_kidney_disease', 'Adrenal_and_thyroid_disorders']

B_test=test[:,1:2]
B_test_name=['Blood_Pressure_Abnormality']

# Create new dataframe with only shortlisted columns
A_test_select = A_test[:,selected_indexes]

#assigning names to the shortlisted array
A_test_select_names=test_data.dtypes.index[skb1.get_support(True)]

B_pred=model_new.predict(A_test_select)
conf=confusion_matrix(B_test, B_pred)
print(conf)
print(classification_report(B_test, B_pred))
model_new.score(A_test_select, B_test)

