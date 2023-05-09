## data preprocessing

import numpy as np
import pandas as pd

df = pd.read_csv('/content/drive/MyDrive/Data mining & ML/data.csv', na_values=['#NAME?'])

df.head(50)

df['race'].unique()

df.columns

df['income']

df['income'] = [0 if x == '<=50K' else 1 for x in df['income']]

df['income']

df

from google.colab import drive
drive.mount('/content/drive')

X = df.drop('income', 1)
y = df.income

X.head()

y

X.isnull().sum()

X['education_num'].head(10)

X['education'].unique()

len(X['education'].unique())

pd.get_dummies(X['education']).head(50)

for col_name in X.columns:
    if X[col_name].dtypes == 'object':
        unique_cat = len(X[col_name].unique())
        print(f"Feature '{col_name}' has {unique_cat} unique categories")

X['native_country'].value_counts().sort_values(ascending=False)

X['native_country'] = ['United-States ' if x == 'United-States' else 'Other' for x in X['native_country']]

X['native_country'].value_counts().sort_values(ascending=False)

todummy_list = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']

# Function to dummy all the categorical variables used for modeling
def dummy_df(df, todummy_list):
    for x in todummy_list:
        dummies = pd.get_dummies(df[x], prefix=x)
        df = df.drop(x, 1)
        df = pd.concat([df, dummies], axis=1)
    return df

X = dummy_df(X, todummy_list)

X.shape

len(X.columns)

X.columns

X.isnull().sum().sort_values(ascending=False).head()

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='median')

imputer.fit(X)
X = pd.DataFrame(data=imputer.transform(X) , columns=X.columns)


X.isnull().sum().sort_values(ascending=False).head()



## Logistic Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline


from google.colab import drive
drive.mount('/content/drive')

df = pd.read_csv('/content/drive/MyDrive/Lab/titanic_train.csv')

df.head()

df.shape

df.isnull().sum().sort_values(ascending=False).head()

df=df.drop(['Cabin'], axis = 1)

df['Age'].fillna(df['Age'].median(), inplace=True)

df.isnull().sum().sort_values(ascending=False).head()

new_df = df.dropna()

new_df

new_df.isnull().sum().sort_values(ascending=False).head()

new_df.columns

new_df=new_df.drop(['PassengerId','Name','Ticket'], axis = 1)

new_df

x = new_df.drop(['Survived'], axis = 1)

x

y = new_df['Survived']

y

from sklearn.preprocessing import LabelEncoder
le_x= LabelEncoder()
x['Gender'] = le_x.fit_transform(x.Gender)
x.Embarked = le_x.fit_transform(x.Embarked)

x

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest= train_test_split(x,y,test_size=.3,random_state=1)

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()
logmodel.fit(xtrain,ytrain)

logmodel.score(xtest,ytest)

predictions = logmodel.predict(xtest)

predictions

from sklearn.metrics import classification_report

ytest.shape

print(classification_report(ytest,predictions))

from sklearn.metrics import accuracy_score
from sklearn.metrics import  confusion_matrix

print(confusion_matrix(ytest, predictions))

logmodel.score(xtest,ytest)
