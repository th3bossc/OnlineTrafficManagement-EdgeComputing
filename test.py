import pandas as pd
from models.xgboost import XgBoost
from utils.data import load_data 
from sklearn.model_selection import train_test_split
model = XgBoost()

X, y = load_data('datasets/vehicle_data.csv')
X_train = X[:1000]
y_train = y[:1000]


model.train(X_train, y_train)
model.save('models/saved/xgboost-test.json')