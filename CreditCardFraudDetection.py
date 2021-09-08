import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# importing dataset
# API command to download dataset -> kaggle datasets download -d jacklizhi/creditcard
# Link to download dataset -> https://www.kaggle.com/jacklizhi/creditcard

dataset = pd.read_csv('creditcard.csv')

dataset.info()

# checking null values

dataset.isnull().sum()

# distribution of legit and fraudulent transaction

dataset['Class'].value_counts()
# 0 -> Legit Transaction
# 1 -> Fraudulent Transaction

# separating the data for analysis

legit = dataset[dataset['Class'] == 0]
fraud = dataset[dataset['Class']  == 1]

# statistical measures of the data

legit['Amount'].describe()

fraud['Amount'].describe()

# compare the values for both transaction

compare = dataset.groupby('Class').mean()

# Under-Sampling
# build a sample dataset containing similar distribution of normal transaction and fraudulent transaction

legit_sample = legit.sample(n=492)

# concatanating two datasets

new_dataset = pd.concat([legit_sample,fraud],axis=0)

compare = new_dataset.groupby('Class').mean()

# splitting

x = new_dataset.iloc[:,:30]
y = new_dataset.iloc[:,30]

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.2,
                                                    random_state=2,
                                                    shuffle=True,
                                                    stratify=y)

# model

LR = LogisticRegression(max_iter=3000)

LR.fit(x_train,y_train)

y_pred = LR.predict(x_test)

accuracy_score(y_test, y_pred)



