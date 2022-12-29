#Mike Ogrysko
#ML CS 737
#Classification pipeline to predict if a passenger from Titanic survived or not
#from Kaggle exercise https://www.kaggle.com/c/titanic/overview

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns; sns.set(style="ticks", color_codes=True)
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC, SVC

# Locate and load the data files
df_train_org = pd.read_csv('./assignment_titanic_train.csv')
df_test_org = pd.read_csv('./assignment_titanic_test.csv')

# Sanity check
print(f'train #rows={len(df_train_org)}, #columns={len(df_train_org.columns)}')
print(f'test #rows={len(df_test_org)}, #columns={len(df_test_org.columns)}')

#look at test data columns and non-null values
df_test_org.info()

#look at train data columns and non-null values
df_train_org.info()

#look at the totals of survivors and non-survivors by sex and passenger class
sns.catplot(x="Sex", hue="Pclass", col="Survived", data=df_train_org, kind="count",height=4, aspect=.7);

#look at the totals of survivors and non-survivors by sex and the number of siblings/spouses
sns.catplot(x="SibSp", hue="Sex", col="Survived", data=df_train_org, kind="count",height=4, aspect=.7);

#look at the totals of survivors and non-survivors by sex and the number of parents/children
sns.catplot(x="Parch", hue="Sex", col="Survived", data=df_train_org, kind="count",height=4, aspect=.7);

#Number of null values in the train data set
df_train_org.isnull().sum()

#Number of null values in the test data set
df_test_org.isnull().sum()

#we know that we can drop the cabin number and ticket number
df_train_org = df_train_org.drop(['Ticket','Cabin'], axis=1)
df_test_org = df_test_org.drop(['Ticket','Cabin'], axis=1)

#look at the empty Embarked records in trainin data set
df_train_org[df_train_org['Embarked'].isnull()]

#no sibling/spouse or parents/children, so use mode?
df_train_org['Embarked'].mode()

#filling Embarked with mode S
df_train_org['Embarked'].fillna('S', inplace=True)
df_train_org.isnull().sum()

#look at empty Fare value in test dataset
df_test_org[df_test_org['Fare'].isnull()]

#use median fare value for passengers with the same Pclass and Embarked values?
df_test_org[(df_test_org['Pclass'] == 3) & (df_test_org['Embarked'] == 'S')].median()

#filling Fare with median 8.05
df_test_org['Fare'].fillna(8.05, inplace=True)
df_test_org.isnull().sum()

#use median age for missing age values
male_median_age = df_train_org[df_train_org['Sex'] == 'male']['Age'].median()
female_median_age = df_train_org[df_train_org['Sex'] == 'female']['Age'].median()
print("Train")
print('Male median age: ', male_median_age)
print("Female median age: ", female_median_age)
te_male_median_age = df_test_org[df_test_org['Sex'] == 'male']['Age'].median()
te_female_median_age = df_test_org[df_test_org['Sex'] == 'female']['Age'].median()
print("Test")
print('Male median age: ', te_male_median_age)
print("Female median age: ", te_female_median_age)

#fill in missing age values
df_train_org.loc[(df_train_org.Age.isnull()) & (df_train_org['Sex'] == 'male'),'Age'] = male_median_age
df_train_org.loc[(df_train_org.Age.isnull()) & (df_train_org['Sex'] == 'female'),'Age'] = female_median_age
df_test_org.loc[(df_test_org.Age.isnull()) & (df_test_org['Sex'] == 'male'),'Age'] = te_male_median_age
df_test_org.loc[(df_test_org.Age.isnull()) & (df_test_org['Sex'] == 'female'),'Age'] = te_female_median_age

#look at train dataset nulls
df_train_org.isnull().sum()

#look at test dataset nulls
df_test_org.isnull().sum()

#is the passenger traveling alone? 0 if no (sibling/spouse or parent/child > 0), 1 if yes
df_train_org['TravelAlone'] = 1
df_train_org.loc[((df_train_org['SibSp'] > 0) | (df_train_org['Parch'] > 0)), 'TravelAlone'] = 0
df_train_org.head()

#add TravelAlone to the test set
df_test_org['TravelAlone'] = 1
df_test_org.loc[((df_test_org['SibSp'] > 0) | (df_test_org['Parch'] > 0)), 'TravelAlone'] = 0
df_test_org.head()

#a few resources online recommend stripping title from name
df_train_org['Title'] = [Name.split(',')[1].split('.')[0].strip() for Name in df_train_org['Name']]
df_train_org.head()

#evaluate the number of titles
df_train_org['Title'].value_counts()

#combine titles where possible
df_train_org['Title'] = df_train_org['Title'].replace(['Mlle','Ms'],'Miss')
df_train_org['Title'] = df_train_org['Title'].replace(['Mme'],'Mrs')
df_train_org['Title'] = df_train_org['Title'].replace(['the Countess','Capt','Don','Jonkheer','Sir','Lady','Major','Col','Rev','Dr'],'Other')
df_train_org['Title'].value_counts()

#add title to test data set
df_test_org['Title'] = [Name.split(',')[1].split('.')[0].strip() for Name in df_test_org['Name']]
df_test_org['Title'] = df_test_org['Title'].replace(['Mlle','Ms'],'Miss')
df_test_org['Title'] = df_test_org['Title'].replace(['Mme'],'Mrs')
df_test_org['Title'] = df_test_org['Title'].replace(['the Countess','Capt','Don','Dona','Jonkheer','Sir','Lady','Major','Col','Rev','Dr'],'Other')
df_test_org['Title'].value_counts()


#drop name since we have the passenger id and title
df_train_org = df_train_org.drop(['Name'], axis=1)
df_test_org = df_test_org.drop(['Name'], axis=1)

#drop SibSp and Parch with addition of TravelAlone
df_train_org = df_train_org.drop(['SibSp','Parch'], axis=1)
df_test_org = df_test_org.drop(['SibSp','Parch'], axis=1)

#check categorical features in train dataset
for col in df_train_org.columns:
    if df_train_org[col].dtype == object:
        print(col, df_train_org[col].unique())

#convert categorical features to numeric
#female = 0, male = 1
df_train_org.Sex = df_train_org.Sex.map({'female':0, 'male':1})
#S = 0, C = 1, Q = 2
df_train_org.Embarked=df_train_org.Embarked.map({'S':0, 'C':1, 'Q':2})
#Mr = 0, Mrs = 1, Miss = 2, Master = 3, Other = 4
df_train_org.Title=df_train_org.Title.map({'Mr':0, 'Mrs':1, 'Miss':2,'Master':3,'Other':4})

#check categorical features in test dataset
for col in df_test_org.columns:
    if df_test_org[col].dtype == object:
        print(col, df_test_org[col].unique())
#convert categorical features to numeric in test dataset
#female = 0, male = 1
df_test_org.Sex = df_test_org.Sex.map({'female':0, 'male':1})
#S = 0, C = 1, Q = 2
df_test_org.Embarked=df_test_org.Embarked.map({'S':0, 'C':1, 'Q':2})
#Mr = 0, Mrs = 1, Miss = 2, Master = 3, Other = 4
df_test_org.Title=df_test_org.Title.map({'Mr':0, 'Mrs':1, 'Miss':2,'Master':3,'Other':4})

#replace age with ordinals
df_train_org['AgeCats'] = pd.cut(df_train_org['Age'],5)
df_train_org[['AgeCats','Survived']].groupby('AgeCats', as_index = False).mean().sort_values(by = 'AgeCats')

#replace ages
df_train_org.loc[(df_train_org['Age'] <= 16.336), 'Age'] = 0
df_train_org.loc[((df_train_org['Age'] > 16.336) & (df_train_org['Age'] <= 32.252)), 'Age'] = 1
df_train_org.loc[((df_train_org['Age'] > 32.252) & (df_train_org['Age'] <= 48.168)), 'Age'] = 2
df_train_org.loc[((df_train_org['Age'] > 48.168) & (df_train_org['Age'] <= 64.084)), 'Age'] = 3
df_train_org.loc[(df_train_org['Age'] > 64.084), 'Age'] = 4
#drop AgeCats
df_train_org = df_train_org.drop(['AgeCats'], axis=1)

#convert age to int
df_train_org['Age'] = df_train_org['Age'].apply(np.int64)
df_train_org.head()

#convert age in test dataset
df_test_org['AgeCats'] = pd.cut(df_test_org['Age'],5)
df_test_org.loc[(df_test_org['Age'] <= 16.336), 'Age'] = 0
df_test_org.loc[((df_test_org['Age'] > 16.336) & (df_test_org['Age'] <= 32.252)), 'Age'] = 1
df_test_org.loc[((df_test_org['Age'] > 32.252) & (df_test_org['Age'] <= 48.168)), 'Age'] = 2
df_test_org.loc[((df_test_org['Age'] > 48.168) & (df_test_org['Age'] <= 64.084)), 'Age'] = 3
df_test_org.loc[(df_test_org['Age'] > 64.084), 'Age'] = 4
df_test_org = df_test_org.drop(['AgeCats'], axis=1)
df_test_org['Age'] = df_test_org['Age'].apply(np.int64)
df_test_org.head()

#replace fare with ordinals
df_train_org['FareCats'] = pd.cut(df_train_org['Fare'],5)
df_train_org[['FareCats','Survived']].groupby('FareCats', as_index = False).mean().sort_values(by = 'FareCats')

#replace fares
df_train_org.loc[(df_train_org['Fare'] <= 102.466), 'Fare'] = 0
df_train_org.loc[((df_train_org['Fare'] > 102.466) & (df_train_org['Fare'] <= 204.932)), 'Age'] = 1
df_train_org.loc[((df_train_org['Fare'] > 204.932) & (df_train_org['Fare'] <= 307.398)), 'Age'] = 2
df_train_org.loc[((df_train_org['Fare'] > 307.398) & (df_train_org['Fare'] <= 409.863)), 'Age'] = 3
df_train_org.loc[(df_train_org['Fare'] > 409.863), 'Fare'] = 4
#drop AgeCats
df_train_org = df_train_org.drop(['FareCats'], axis=1)

#convert fare to int
df_train_org['Fare'] = df_train_org['Fare'].apply(np.int64)
df_train_org.head()

#convert fare in test dataset
df_test_org['FareCats'] = pd.cut(df_test_org['Fare'],5)
df_test_org.loc[(df_test_org['Fare'] <= 102.466), 'Fare'] = 0
df_test_org.loc[((df_test_org['Fare'] > 102.466) & (df_test_org['Fare'] <= 204.932)), 'Age'] = 1
df_test_org.loc[((df_test_org['Fare'] > 204.932) & (df_test_org['Fare'] <= 307.398)), 'Age'] = 2
df_test_org.loc[((df_test_org['Fare'] > 307.398) & (df_test_org['Fare'] <= 409.863)), 'Age'] = 3
df_test_org.loc[(df_test_org['Fare'] > 409.863), 'Fare'] = 4
df_test_org = df_test_org.drop(['FareCats'], axis=1)
df_test_org['Fare'] = df_test_org['Fare'].apply(np.int64)
df_test_org.head()

#correlation
X = df_train_org.values
N, M = X.shape
#Average of features
Xavg = np.zeros(M)
for i in range(M):
    Xavg[i] = df_train_org.iloc[:,i].mean()

Xcor = np.zeros((M,M))
for x in range(M):
    for y in range(M):
        Xcor[x,y] = np.sum((X[:,x] - Xavg[x]) * (X[:,y] - Xavg[y])) / ((np.sqrt(np.sum((X[:,x] - Xavg[x])**2))) * (np.sqrt(np.sum((X[:,y] - Xavg[y])**2))))

cols = df_train_org.columns
sns.heatmap(Xcor, annot=True, xticklabels=cols, yticklabels=cols)
sns.set(rc = {'figure.figsize':(30,15)})
plt.show()

X_train = df_train_org.drop(['Survived'], axis=1)
y_train = df_train_org['Survived']
X_test = df_test_org

print("X_train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)
print("X_test shape: ", X_test.shape)

#svm
svc = SVC()
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, y_train) * 100, 2)
acc_svc

#random forest
rf = RandomForestClassifier(n_estimators = 100)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
acc_rf = round(rf.score(X_train, y_train) * 100, 2)
acc_rf

#logistic regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
acc_lr = round(lr.score(X_train, y_train) * 100, 2)
acc_lr

def save_preds(_fn, _y_pred, _df):
    import csv
    with open(_fn, 'w') as fout:
        writer = csv.writer(fout, delimiter=',', lineterminator='\n')
        writer.writerow(['Survived', 'PassengerId'])
        for y, passengerId in zip(_y_pred, _df['PassengerId']):
            writer.writerow([y, passengerId])

save_preds('predictions_ogrysko.csv', y_pred, df_test_org)

save_preds('predictions_ogrysko_rf.csv', y_pred_rf, df_test_org)

save_preds('predictions_ogrysko_lr.csv', y_pred_lr, df_test_org)

#Final Accuracies - From Kaggle
#SVM - Score: 0.63157
#RF - Score: 0.75598
#LR - Score: 0.76794

