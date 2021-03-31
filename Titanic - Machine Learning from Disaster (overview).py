import numpy as np
import pandas as pd
import os
os.chdir('Titanic (Kaggle)/Data')
import matplotlib.pyplot as plt

train = pd.read_csv('train.csv').fillna(value = {'Embarked':'S'}) # since there are only 3 missing values for the 'Embarked' variable, I just filled them with 'S', which is
test = pd.read_csv('test.csv').fillna(value = {'Embarked':'S'})   # by far the most common value

alldata = pd.concat([train.drop('Survived', axis=1), test]).reset_index(drop=True)
#alldata.shape  (1309, 11)

## Splitting the Data

from sklearn.model_selection import train_test_split # sklearn.cross_validation is depreciated
my_train_data, my_CV_data = train_test_split(train, test_size=0.25, random_state=0)

## Overview of the Data

fig, ((ax1, ax2, ax3, ax4),(ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, sharey=False, figsize=(20,10))

a = alldata.groupby('Sex').size()
ax1.bar(a.index, a, width=0.5, edgecolor='black')
ax1.set_title('Sex')

a = alldata.groupby('Embarked').size()
ax2.bar(a.index, a, width=0.5, edgecolor='black')
_=ax2.set_title('Embarked')

a = alldata.groupby('SibSp').size()
ax3.bar(a.index, a, width=0.5, edgecolor='black')
_=ax3.set_title('Siblings/Spouses')

a = alldata.groupby('Parch').size()
ax4.bar(a.index, a, width=0.5, edgecolor='black')
_=ax4.set_title('Parents/children')

ax5.hist(alldata['Age'], bins=20, edgecolor='black')
_=ax5.set_title('Age')

## The 'Age' variable has a lot of missing data, the following code shows that Age varies accross Sex and Port of embarkment, 
## thus dealing with the problem with mean imputation for instance will skew the results

sex_bool = alldata['Sex'].map({'male':1, 'female':0}).astype(bool)

embarked_C_bool = alldata['Embarked'].map({'C':1, 'Q':0, 'S':0}).astype(bool)
embarked_Q_bool = alldata['Embarked'].map({'C':0, 'Q':1, 'S':0}).astype(bool)

from scipy.stats import f_oneway
#alldata[sex_bool]['Age'].isnull().sum() 185 males
#alldata[~sex_bool]['Age'].isnull().sum() 78 females

mask = alldata['Age'].isnull()
print(f_oneway(alldata[~mask & sex_bool]['Age'], alldata[~mask & ~sex_bool]['Age']))

print(f_oneway(alldata[~mask & embarked_C_bool]['Age'], alldata[~mask & embarked_Q_bool]['Age'], 
         alldata[~mask & ~embarked_C_bool & ~embarked_Q_bool]['Age']))

# pvalue<0.05 => we reject H0 : Age's distribution varies across sex and port of embarkation

