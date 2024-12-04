from models.xgboost import XgBoost
from utils.data import load_data


def retraining_trigger(window_index, retrain_flags, my_model_f1s_proxy, static_model_f1s_proxy):
    
    # naive 
    # return True
    
    # timing based
    # if (window_index - 1) % 20 == 0:
    #     return True
    # else:
    #     return False
    
    # accuracy based
    if len(retrain_flags) == 0:
        return True
    
    # scenario 1: we have not retrained for last window, and there is drift
    if retrain_flags[-1] == False:
        if my_model_f1s_proxy[-1] - static_model_f1s_proxy[-1] < 0.1:
            return True
        else: 
            return False
        
    # scenario 2: we have retraiend for last window, and we need/do not need more retrain
    else:
        if abs(my_model_f1s_proxy[-1] - static_model_f1s_proxy[-1]) < 0.1:
            return False
        else:
            return True 
        
        
def retrain_model(model, X_train, y_train):
    model.train(X_train, y_train) 
    return model 


def get_new_data():
    pass


# main streaming function
def continous_retrain():

    # load data
    X_train, y_train, X_test, y_test = load_data('../datasets/vehicle_data.csv')
    
    # initialize model
    model = XgBoost()
    
    # initialize variables
    window_index = 0
    retrain_flags = []
    my_model_f1s_proxy = []
    static_model_f1s_proxy = []



