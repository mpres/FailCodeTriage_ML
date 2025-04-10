#Mpresley 3/10/25 test file
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import sys
import os

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
    #Mpresley 4/2/25, take away scaling (using catgorical data)
    #scaler = StandardScaler()
    #X = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    #Mpresley 4/2/25 Encode data
    X_test =  label_encode_and_convert(X_test)
    y_test =  label_encode_and_convert(y_test)

    X_train =  label_encode_and_convert(X_train)
    y_train =  label_encode_and_convert(y_train)


    # Create data loaders
    train_dataset = ScreeningDataset(X_train, y_train)
    test_dataset = ScreeningDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader, X_train.shape[1]

def train_model(model, train_loader, num_epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total = 0
        correct = 0

        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            total += labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        accuracy = 100 * correct / total
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}, Train Accuracy: {accuracy:.2f}')

def evaluate_model(model, test_loader):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    correct = 0
    total = 0
    total_loss = 0

    with torch.no_grad():
        for features, labels in test_loader:
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            total_loss += loss.item()


    accuracy = 100 * correct / total
    test_loss = total_loss/len(test_loader)
    print(f'Test Accuracy: {accuracy:.2f}%, Test Loss: {test_loss:.4f}%')

#MPresley added 03/31/25,
def label_encode_and_convert(array):
    assert(array.ndim < 3)
    encoder = LabelEncoder()
    result = np.zeros(array.shape, dtype=np.float32)

    if array.ndim == 1:
      result = encoder.fit_transform(array)
    else:
      for col in range(array.shape[1]):
          result[:, col] = encoder.fit_transform(array[:, col]).astype(np.float32)

    return result

def predict_probabilities(model, features):
    model.eval()
    with torch.no_grad():
        features = torch.FloatTensor(features)
        outputs = model(features)
        return outputs.numpy()

def train_and_eval_model(model, train_loader, test_loader, num_epochs=10):
  num_e = 1
  for i in range(num_epochs):
    train_model(model, train_loader,num_e)
    evaluate_model(model, test_loader)
  return None

def main():
    # Example usage
    csv_path = "your_data.csv"  # Replace with your CSV file path

    # Prepare data
    train_loader, test_loader, input_size = prep_data(csv_path)

    # Initialize and train model
    model = FailCodeClassifier(input_size)
    train_model(model, train_loader)

    # Evaluate model
    evaluate_model(model, test_loader)

    # Example of getting probabilities for new data
    new_data = torch.randn(1, input_size)  # Replace with your actual new data
    probabilities = predict_probabilities(model, new_data)
    print("Predicted probabilities for each class:", probabilities)

if __name__ == "__main__":
    main()
