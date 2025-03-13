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

# Create Neural Network Model that will make the failcode Prediction
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
