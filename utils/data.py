import pandas as pd
import numpy as np
def load_data(path):
    data = pd.read_csv(path)
    data = data.drop(columns=['time'])
    
    return preprocess_data(data)


def preprocess_data(data):
    for i in range(1, 13):
        data['load_t-'+str(i)] = data['vehicles'].shift(i)
    data.dropna(inplace=True)

    X = data[['load_t-'+str(i) for i in range(1, 13)]].to_numpy()
    y = data['vehicles'].to_numpy()
    return X, y


def get_new_data(start):
    if start > 6000:
        start = 100
    X, y = load_data('datasets/vehicle_data.csv')
    X, y = X[start:start+100], y[start:start+100]
    return X, y