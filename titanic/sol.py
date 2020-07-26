import numpy as np
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

# data
print(data)

# data shape
print(data.shape)
# (1309, 12)

# data info
print(data.info())

# Data columns (total 12 columns):
#  #   Column       Non-Null Count  Dtype
# ---  ------       --------------  -----
#  0   PassengerId  1309 non-null   int64
#  1   Survived     891 non-null    float64
#  2   Pclass       1309 non-null   int64
#  3   Name         1309 non-null   object
#  4   Sex          1309 non-null   object
#  5   Age          1046 non-null   float64
#  6   SibSp        1309 non-null   int64
#  7   Parch        1309 non-null   int64
#  8   Ticket       1309 non-null   object
#  9   Fare         1308 non-null   float64
#  10  Cabin        295 non-null    object
#  11  Embarked     1307 non-null   object

# Missing values
plt.ylabel("Missing values:")
plt.plot(pd.DataFrame(data.isnull().sum()))
plt.show()
print(data.isnull().sum())

sns.heatmap(data.isnull(), cbar=False).set_title("Missing values heatmap")
plt.show()

# PassengerId       0
# Survived        418   <--
# Pclass            0
# Name              0
# Sex               0
# Age             263   <--
# SibSp             0
# Parch             0
# Ticket            0
# Fare              1
# Cabin          1014   <--
# Embarked          2   <--

# PassengerId
# > Doesn't affect survived

# Survived
# > Validation

# Pclass

pClass_1 = round(
    (data_train[data_train.Pclass == 1].Survived == 1).value_counts()[1] /
    len(data_train[data_train.Pclass == 1]) * 100, 2)
pClass_2 = round(
    (data_train[data_train.Pclass == 2].Survived == 1).value_counts()[1] /
    len(data_train[data_train.Pclass == 2]) * 100, 2)
pClass_3 = round(
    (data_train[data_train.Pclass == 3].Survived == 1).value_counts()[1] /
    len(data_train[data_train.Pclass == 3]) * 100, 2)

pClassDf = pd.DataFrame(
    {"Survived": {"Class 1": pClass_1,
                  "Class 2": pClass_2,
                  "Class 3": pClass_3},
     "Not survived": {"Class 1": 100 - pClass_1,
                      "Class 2": 100 - pClass_2,
                      "Class 3": 100 - pClass_3}})
pClassDf.plot.bar().set_title("Survived ~ Slass")
plt.show()

# Name
# > Doesn't affect survived

# Sex

print(data.Sex)

sex_1 = round(
    (data_train[data_train.Sex == 'male'].Survived == 1).value_counts()[1] /
    len(data_train[data_train.Sex == 'male']) * 100, 2)
sex_2 = round(
    (data_train[data_train.Sex == 'female'].Survived == 1).value_counts()[1] /
    len(data_train[data_train.Sex == 'female']) * 100, 2)

pClassDf = pd.DataFrame(
    {"Survived": {"Male": sex_1,
                  "Female": sex_2},
     "Not survived": {"Male": 100 - sex_1,
                      "Female": 100 - sex_2}})
pClassDf.plot.bar().set_title("Survived ~ Sex")
plt.show()

# Age

print(data.Age)

data['Age_Range'] = pd.cut(data.Age, [0, 10, 20, 30, 40, 50, 60, 70, 80])
sns.countplot(x="Age_Range", hue="Survived", data=data, palette=["C1", "C0"]).legend(
    labels=["Not survived", "Survived"])

# SibSp

print(data.SibSp)

ss = pd.DataFrame()
ss['survived'] = data_train.Survived
ss['sibling_spouse'] = pd.cut(data_train.SibSp, [0, 1, 2, 3, 4, 5, 6, 7, 8], include_lowest=True)

x = sns.countplot(x="sibling_spouse", hue="survived", data=ss, palette=["C1", "C0"]).legend(
    labels=["Not survived", "Survived"])
x.set_title("Survival ~ Number of siblings or spouses")
plt.show()

# Parch

print(data.Parch)

pc = pd.DataFrame()
pc['survived'] = data_train.Survived
pc['parents_children'] = pd.cut(data_train.Parch, [0, 1, 2, 3, 4, 5, 6], include_lowest=True)
x = sns.countplot(x="parents_children", hue="survived", data=pc, palette=["C1", "C0"]).legend(
    labels=["Not survived", "Survived"])

x.set_title("Survival ~ Parents/Children")
plt.show()

# Ticket
# > Doesn't affect survived

# Fare

print(data.Fare)

data_train['Fare_Category'] = pd.cut(data_train['Fare'], bins=[0, 7.90, 14.45, 31.28, 120], labels=['Low', 'Mid',
                                                                                                    'High_Mid', 'High'])
x = sns.countplot(x="Fare_Category", hue="Survived", data=data_train, palette=["C1", "C0"]).legend(
    labels=["Not survived", "Survived"])
x.set_title("Survival ~ Fare")

# Cabin
data.Cabin = data.Cabin.fillna('NA')

# Embarked

print(data.Embarked)

p = sns.countplot(x="Embarked", hue="Survived", data=data_train, palette=["C1", "C0"])
p.set_xticklabels(["Southampton", "Cherbourg", "Queenstown"])
p.legend(labels=["Not survived", "Survived"])
p.set_title("Survival ~ Embarking.")

# Missing values

data_train['Salutation'] = data_train.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())
data_train.Salutation.nunique()
wc = WordCloud(width=1000, height=450, background_color='white').generate(str(data_train.Salutation.values))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()

data_train.Salutation.value_counts()

grp = data_train.groupby(['Sex', 'Pclass'])
data_train.Age = grp.Age.apply(lambda x_: x_.fillna(x_.median()))
data_train.Age.fillna(data_train.Age.median, inplace=True)

# Prediction

X_pred = data[data.Survived.isnull()].drop(['Survived'], axis=1)

# Training data
train_data = data.dropna()
feature_train = train_data['Survived']
label_train = train_data.drop(['Survived'], axis=1)
