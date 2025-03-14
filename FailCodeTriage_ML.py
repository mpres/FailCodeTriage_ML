#Mpresley 3/10/25 test file
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

# Screening Dataset class for handling
#Mpresley 3/12/25
class ScreeningDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Create Neural Network Model that will make the failcode Prediction 3/13/25,
class FailCodeClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(FailCodeClassifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.layers(x)

def prep_data(csv_path):
    # Load and preprocess data
    data = pd.read_csv(csv_path)
    
    # Assuming the last column is the target variable
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    
