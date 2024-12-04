from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
import time
import joblib
from .general_model import GeneralModel

class RandomForest(GeneralModel):
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
        
    def predict(self, input):
        return self.model.predict(input)
    
    def test(self, X_test, y_test):
        start_time = time.time()
        preds = self.model.predict(X_test)
        test_time = time.time() - start_time
        mse = root_mean_squared_error(y_test, preds)
        
        return mse, test_time
        
    def save(self, path):
        joblib.dump(self.model, path)
        
    def load(self, path):
        self.model = joblib.load(path)