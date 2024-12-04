from models.xgboost import XgBoost
from utils.data import get_new_data
import copy
from queue import Queue
from collections import deque
import threading
import numpy as np
from time import sleep
from utils.metrics import compute_acc
import matplotlib.pyplot as plt
import signal
import sys



################## global vars ########################

buffer_size = 2000
update_threshold = 1500
evaluation_window = 2000

performance_threshold = 4e-5


data_buffer = deque(maxlen=buffer_size)
label_buffer = deque(maxlen=buffer_size)

prediction_errors = deque(maxlen=evaluation_window)
baseline_error = None

update_lock = threading.Lock()
model_switch_lock = threading.Lock()
is_training = False
training_thread = None 

error_over_time = []

with update_lock, model_switch_lock:
    primary_model = XgBoost()
    primary_model.load('models/saved/xgboost-test.json')

    retrain_model = None



    
update_queue = Queue()
#######################################################


def signal_handler(sig, frame):
    update_queue.put(None)
    
    error_list, is_training_list = [], []
    for error in error_over_time:
        error_list.append(error['error'])
        is_training_list.append(error['is_training'])
    
    
    plt.plot(error_list)
    plt.title('Error over time')
    plt.xlabel('Time')
    plt.ylabel('Error')
    
    for i in range(len(is_training_list)):
        if is_training_list[i]:
            plt.axvline(x=i, color='r', linestyle='--')

    print('saving')
    plt.savefig('plot_image.png', dpi=300, bbox_inches='tight')
    sys.exit(0)
    
signal.signal(signal.SIGINT, signal_handler)
    

def _start_update_worker():
    def update_worker():
        while True:
            update_task = update_queue.get()
            if update_task is None:
                break
            _process_update(update_task)
            update_queue.task_done()
            
    training_thread = threading.Thread(target=update_worker, daemon=True)
    training_thread.start()
    
def _process_update(data):
    X_update, y_update = data
    global primary_model, retrain_model, baseline_error, is_training
    try:
        with update_lock:
            is_training = True
            retrain_model = copy.deepcopy(primary_model)
            print('updating model')
            retrain_model.train(X_update, y_update)
            
        with model_switch_lock:
            primary_model = retrain_model
            retrain_model = None 
            
            
        if len(prediction_errors) > 0:
            baseline_error = np.mean(list(prediction_errors))
            
    finally:
        is_training = False


def add_sample(X, y):
    data_buffer.append(X)
    label_buffer.append(y)
    
    with model_switch_lock:
        pred = primary_model.predict(X.reshape(1, -1))[0]
        
    error = compute_acc(pred, y)
    prediction_errors.append(error)
    
    _check_and_update()
    

def _check_and_update():
    global baseline_error, is_training
    if is_training or update_queue.qsize() > 0:
        return 
    
    should_update = False 
    
    if len(data_buffer) >= update_threshold:
        should_update = True 
        
    if len(prediction_errors) > evaluation_window:
        current_error = np.mean(prediction_errors)
        
        if baseline_error is None:
            baseline_error = current_error
            
        if current_error > baseline_error * (performance_threshold):
            print('Performance threshold reached')
            should_update = True

    if should_update:
        X_update = np.array(list(data_buffer))
        y_update = np.array(list(label_buffer))
        update_queue.put((X_update, y_update))
        
        data_buffer.clear()
        label_buffer.clear()

        

def predict(input):
    with model_switch_lock:
        primary_model.predict(input.reshape(1, -1))[0]
        

def train(X_train, y_train):
    global baseline_error
    with model_switch_lock:
        primary_model.train(X_train, y_train)
        baseline_error = None 
        prediction_errors.clear()
        

def test(X_test, y_test):
    with model_switch_lock:
        mse, test_time = primary_model.test(X_test, y_test)
        
        print(f'Time taken to test: {test_time}')
        print(f'Average error: {mse}')
    
    
def get_performance_metrics():
    return {
        'buffer_size': len(data_buffer),
        'prediction_error': len(prediction_errors),
        'current_error': np.mean(list(prediction_errors)) if prediction_errors else None,
        'baseline_error': baseline_error,
        'is_training': is_training,
        'pending_updates': update_queue.qsize()
    }


_start_update_worker()
start = 1000
i = 0
while True:
    if start == 6000:
        start = 1000
    X, y = get_new_data(start)
    start += 100
    
    
    for sample, label in zip(X, y):
        add_sample(sample, label)    
    
        predict(sample)
        
        
    metrics = get_performance_metrics()
    i += 1
    print(metrics)
    
    error_over_time.append({'error': metrics['current_error'], 'is_training': metrics['is_training']})
    
    
