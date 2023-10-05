import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"D:\Data Science\CodSoft Tasks\Task 1 Titanic\tested.csv")
df.head(10)

df.shape
df.describe()
df['Survived'].value_counts()

sns.countplot(x=df['Survived'], hue=df['Pclass'])

df["Sex"]
sns.countplot(x=df['Sex'], hue=df['Survived'])
df.groupby('Sex')[['Survived']].mean()

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

df['Sex'] = labelencoder.fit_transform(df['Sex'])
df.head()

sns.countplot(x=df['Sex'], hue=df['Survived'])
df.isna().sum() 

df=df.drop(['Age'], axis=1)
df_final = df
df_final.head(10)

X=df[['Pclass', 'Sex']]
Y=df['Survived']

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

from sklearn.linear_model  import LogisticRegression

log = LogisticRegression(random_state = 0)
log.fit(X_train, Y_train)

pred = print(log.predict(X_test))
print(Y_test)

import warnings
warnings.filterwarnings("ignore")
res=log.predict([[2,0]])

if(res==0):
    print("Not Survived")
else:
    print("Survived")