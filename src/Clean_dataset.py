import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data\data.csv')
print(df['status'].value_counts())

df_cleaned = df[~df['status'].isin(['Suicidal', 'Anxiety' , 'Bipolar' ,'Stress' ,'Personality disorder'])]

print(df_cleaned['status'].value_counts())

x_train , x_test = train_test_split(
    df_cleaned,
    test_size = 0.1,
    stratify =  df_cleaned['status'],
    random_state= 42
)

print(x_test['status'].value_counts())
print(x_train['status'].value_counts())

x_test.to_csv('testing_dataset.csv' , index=False)
x_train.to_csv('training_dataset.csv' , index=False)

