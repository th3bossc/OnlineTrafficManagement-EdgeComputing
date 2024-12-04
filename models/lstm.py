import torch
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time

from .general_model import GeneralModel 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class TimeSeriesDataset(Dataset):
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, idx):
        sequence, label = self.X_train[idx], self.y_train[idx]
        # Convert to torch tensors
        sequence = torch.tensor(sequence, dtype=torch.float32).unsqueeze(-1)  # Shape: (sequence_length, 1)
        label = torch.tensor(label, dtype=torch.float32)
        return sequence, label


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(LSTM, self).__init__()
        self.model = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
        
    def forward(self, x):
        out, _ = self.model(x)
        out = self.fc(out[:, -1, :])
        return out
    

class TrafficPredictionLSTM(GeneralModel):
    def __init__(self):
        self.model = LSTM()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.model.to(device)
    
    
    def train(self, X_train, y_train, num_epochs=100):
        self.model.train()
        dataset = TimeSeriesDataset(X_train, y_train)
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            for features, targets in train_loader:
                features, targets = features.to(device), targets.to(device)
                
                outputs = self.model(features)
                loss = self.criterion(outputs, targets.view(-1, 1))
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
                
            if epoch % 10 == 0:
                print(f'Epoch {epoch} Loss: {torch.sqrt(torch.tensor(epoch_loss / len(train_loader)))}')
                
                
    def predict(self, input):
        self.model.eval()
        data = torch.tensor(input, dtype=torch.float32).unsqueeze(-1).to(device)
        
        with torch.no_grad():
            prediction = self.model(data)
            
        return prediction.cpu().numpy().squeeze(-1)
                
                
    def test(self, X_test, y_test):
        dataset = TimeSeriesDataset(X_test, y_test)
        test_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        self.model.eval()
        total_loss = 0
        total_time = 0
        
        with torch.no_grad():
            for features, targets in test_loader:
                features, targets = features.to(device), targets.to(device)
                start_time = time.time()
                outputs = self.model(features)
                total_time += time.time() - start_time
                loss = self.criterion(outputs, targets.view(-1, 1))
                total_loss += loss.item()
                
        average_loss = total_loss / len(test_loader)
        rmse_loss = torch.sqrt(torch.tensor(average_loss))
        
        return rmse_loss, total_time / len(test_loader)
        
        
        
    def save(self, path):
        torch.save(self.model.state_dict(), path)
        
        
    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        
    