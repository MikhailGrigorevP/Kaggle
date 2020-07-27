import numpy as np
import sklearn
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import pyplot as plt
from wordcloud import WordCloud
from sklearn.metrics import accuracy_score, log_loss
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import LabelEncoder

data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')
data = pd.concat([data_train, data_test], ignore_index=True, sort=False)

data['Age_Range'] = pd.cut(data.Age, [0, 10, 20, 30, 40, 50, 60, 70, 80])
data['Family'] = data.Parch + data.SibSp
data['Is_Alone'] = data.Family == 0

data['Fare_Category'] = pd.cut(data_train['Fare'], bins=[0, 7.90, 14.45, 31.28, 120], labels=['Low', 'Mid',
                                                                                              'High_Mid', 'High'])

# Missing

data['Salutation'] = data.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())
data.Salutation.nunique()

grp = data.groupby(['Sex', 'Pclass'])
data.Age = grp.Age.apply(lambda x_: x_.fillna(x_.median()))
data.Age.fillna(data.Age.median, inplace=True)

data.Embarked.fillna(data.Embarked.mode()[0], inplace=True)
data.Cabin = data.Cabin.fillna('NA')

data = pd.concat([data, pd.get_dummies(data.Cabin, prefix="Cabin"),
                  pd.get_dummies(data.Age_Range, prefix="Age_Range"),
                  pd.get_dummies(data.Embarked, prefix="Emb", drop_first=True),
                  pd.get_dummies(data.Salutation, prefix="Title", drop_first=True),
                  pd.get_dummies(data.Fare_Category, prefix="Fare", drop_first=True),
                  pd.get_dummies(data.Pclass, prefix="Class", drop_first=True)], axis=1)

data['Is_Alone'] = LabelEncoder().fit_transform(data['Is_Alone'])
data['Sex'] = LabelEncoder().fit_transform(data['Sex'])

data.drop(['Pclass', 'Fare', 'Cabin', 'Fare_Category', 'Name', 'Salutation', 'Ticket', 'Embarked', 'Age_Range', 'SibSp',
           'Parch', 'Age'], axis=1, inplace=True)

# scaler

scaler = sklearn.preprocessing.StandardScaler()

# Prediction

X_pred = data[data.Survived.isnull()].drop(['Survived'], axis=1)

# Training data

train_data = data.dropna()
X = train_data.drop(['Survived'], axis=1)
y = train_data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# Random forest
# entropy
# gini

alg_frst_model = RandomForestClassifier(random_state=1)
alg_frst_params = [{
    "criterion": ['entropy', 'gini'],
    "n_estimators": [350, 400, 450, 500, 550, 600, 650, 700],
    "min_samples_split": [6, 8, 10],
    "min_samples_leaf": [1, 2, 4],
    "n_jobs": [-1, 1, 2]
}]

cv = sklearn.model_selection.StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
#
# alg_frst_grid = sklearn.model_selection.GridSearchCV(alg_frst_model, alg_frst_params, cv=cv, refit=True, verbose=1,
#                                                      n_jobs=-1)
# alg_frst_grid.fit(X_train, np.ravel(y_train))
# alg_frst_best = alg_frst_grid.best_estimator_
# print("Accuracy (random forest auto): {} with params {}"
#       .format(alg_frst_grid.best_score_, alg_frst_grid.best_params_))

# Accuracy (random forest auto): 0.8218211941038028 with params {'criterion': 'entropy', 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 350, 'n_jobs': -1}

clf = RandomForestClassifier(criterion='entropy',
                             n_estimators=350,
                             min_samples_split=10,
                             min_samples_leaf=1,
                             max_features='auto',
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1)

clf.fit(X_train, y_train)
print("RF Accuracy: " + repr(round(clf.score(X_test, y_test) * 100, 2)) + "%")

result_rf = cross_val_score(clf, X_train, y_train, cv=10, scoring='accuracy')
print('The cross validated score for Random forest is:', round(result_rf.mean() * 100, 2))
y_pred = cross_val_predict(clf, X_train, y_train, cv=10)


result = clf.predict(X_pred)
submission = pd.DataFrame({'PassengerId': X_pred.PassengerId, 'Survived': result})
submission.Survived = submission.Survived.astype(int)
print(submission.shape)
filename = 'TitanicPredictions.csv'
submission.to_csv(filename, index=False)
print('Saved file: ' + filename)
