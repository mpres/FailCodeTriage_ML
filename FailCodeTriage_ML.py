#Mpresley 3/10/25 test file
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import sys
import os
import re

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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

    if "col7" in data.columns:
      split_data = data['col7'].str.split('~', expand=True)
      data = pd.concat([data,split_data],axis=1)
      data = data.drop('col7',axis=1)
    else:
      print("no col7")

    #process the time data
    #Mpresley 7/18/25 look for mixed formats
    data['time_col'] = pd.to_datetime(data['col13'], format='mixed').astype('int64') // 10**9
    scaler = MinMaxScaler()
    data['normalized_sklearn_minmax_time'] = scaler.fit_transform(data[['time_col']])
    data = data.drop(["col13", "time_col"], axis=1)

    #Process the wom data
    data['col8_num'] = data['col8'].apply(lambda x: convert_text_to_int(x, text_replace={"-":""}) )
    data['col8_num_scaled'] = scaler.fit_transform(data[['col8_num']])
    data = data.drop(["col8","col8_num"], axis=1)

    #Process the Model data
    data['col2_num'] = data['col2'].apply(lambda x: convert_text_to_int(x) )
    data = data.drop("col2", axis=1)

    #Drop the product field
    data = data.drop("col3", axis=1)

    #Dummy Data
    #Mpresley 5/17/25, don't create dummy data of col15
    dummy_cols = ['col1','col4','col6','col10','col11','col14','col5','col11','col12']
    data = pd.get_dummies(data, columns=dummy_cols)
    #Mpresley 5/17/25, move col15 to the end to make it y data
    data['col15'] =  data.pop('col15')


    # Assuming the last column is the target variable
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    #Mpresley 7/9/25
    y_alpha = y
    y_dict = {}

    #Mpresley 5/17/25 we can try single encoding the y data before splitting it, to make different categories
    y = label_encode_and_convert(y)

    # Scale features
    #scaler = StandardScaler()
    #X = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    #Mpresley 4/2/25 Encode data
    X_test =  label_encode_and_convert(X_test)
    #Mpresley 5/17/25 y data already encoded
    #y_test =  label_encode_and_convert(y_test)

    #Mpresley look for ints in colum one and clear them out
    int_indices = np.where([isinstance(x, (int, np.integer)) for x in X_train[:,0]])

    X_train = np.delete(X_train,int_indices,axis=0)
    #Mpresley 7/9/25 probably will need to delete the same index/indexes from the y_train

    #Mpresley 7/10/25 we need to delete the y label indexs as well
    y_train = np.delete(y_train,int_indices,axis=0)

    X_train =  label_encode_and_convert(X_train)
    #Mpresley 5/17/25 y data already encoded
    #y_train =  label_encode_and_convert(y_train)


    # Create data loaders
    train_dataset = ScreeningDataset(X_train, y_train)
    test_dataset = ScreeningDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    #scaler =

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
        print(f'Train Loss: {total_loss/len(train_loader):.4f}, Train Accuracy: {accuracy:.2f}')

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
    print(f'Test Loss: {test_loss:.4f}%, Test Accuracy: {accuracy:.2f}%')

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
    print(f'Epoch: {i}/{num_epochs}')
    train_model(model, train_loader,num_e)
    evaluate_model(model, test_loader)
  return None

def convert_text_to_int(text, map_replace = {},text_replace = {}, filter_nums=1):
  '''
  Author: Mpresley
  Date: 4/17/25
  Purpose: this is used to convert a column in pandas taht is text into a numeric form,
           Generally this is only useful when the int value is embedded in the text data or
           can be mapped into an int (using a dictionary)
  Parameters:
              1. text (this is the raw text value, needs to be a text value)
              2. map_repalce (this should be a dictionary, if the text value is a key in the map
              it will be replaced with the value)
              3. text_place (is also a dictionary, if a pattern is found in the text, it will be replace with
              the value of this dictionary)
              4. filter nums is a boolean flag that will filter the numeric values if left on
  Outline:
            1. Validate data types of the parameters using asserts
            2. return map_relace value if found
            3. do a a text_place .replace function on the text
            4. filter nums only
            5. if possible convert remaining value to int
  Requirements: import re
  '''

  # Step 1. validate input tpe
  try:
    assert type(text) == str
  except:
    print(f"the text parameter {text} has to be a string")
    return 0

  try:
    assert type(map_replace) == dict
  except:
    print (f"the map_replace parameter {map_replace} has to be a dictionary")
    return 0

  try:
    assert type(text_replace) == dict
  except:
    print(f"the text_replace parameter {text_replace} has to be a dictionary")
    return 0


  #Step 2 look for map replace

  if text in map_replace:
    return map_replace[text]

  #Step 3 look for patterns to substitute

  for item in text_replace:
    text = text.replace(item,text_replace[item])

  # Step 4 filter out numeric values

  if filter_nums == 1:
    text = re.sub(r"[a-zA-Z]", "", text)
    text = text.replace("+","")

  # Step 5 try to convert to int
  try:
    int_val = int(text)
  except:
    print(f"The value {text} can not convert into an int")
    return 0

  return int_val

def main():
    # Example usage
    #csv_path = "your_data.csv"  # Replace with your CSV file path
    #MPresley 5/16/25 use command line for csv path
    csv_path = sys.argv[1]
    if len(sys.argv) > 2:
      num_epochs = int(sys.argv[2])
    else:
      num_epochs = 10



    # Prepare data
    train_loader, test_loader, input_size = prep_data(csv_path)

    # Initialize and train model
    model = FailCodeClassifier(input_size)
    #train_model(model, train_loader)


    # Evaluate model
    #evaluate_model(model, test_loader)

    #Mpresley 5/16/25, using the train_and_eval function
    train_and_eval_model(model, train_loader, test_loader, num_epochs=num_epochs )


    # Example of getting probabilities for new data
    new_data = torch.randn(1, input_size)  # Replace with your actual new data
    probabilities = predict_probabilities(model, new_data)
    print("Predicted probabilities for each class:", probabilities)

if __name__ == "__main__":
    main()
