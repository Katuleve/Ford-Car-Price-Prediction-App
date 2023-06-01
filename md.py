import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
dataset = pd.read_csv('ford.csv')

dataset.replace({'transmission': {'Automatic':0, 'Manual':1, 'Semi-Auto':2}}, inplace=True)
dataset.replace({'fuelType': {'Petrol':0, 'Diesel':1, 'Hybrid':2, 'Electric':3, 'Other':4}}, inplace=True)

X = dataset.drop(['model', 'price'], axis = 1).values
y = dataset['price']

scaler = StandardScaler()
scaler.fit(X)

standardized_x = scaler.transform(X)

x = standardized_x

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=.1)

xgb_model = XGBRegressor()
xgb_model.fit(x_train, y_train)

input_data = (2017, 0, 15944, 0, 150, 57.7, 1.0)

input_changed = np.array(input_data).reshape(1,-1)

std_input = scaler.transform(input_changed)

prediction = xgb_model.predict(std_input)

import joblib

joblib.dump(xgb_model, 'xgb_model.pkl')

joblib.dump(scaler, 'scaler.pkl')