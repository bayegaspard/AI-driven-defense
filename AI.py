import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device used: {DEVICE}')

# Model definition
class Net(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            probabilities = torch.softmax(outputs, dim=1)
        return probabilities

# Dataset loading and preprocessing
print('Loading dataset...')
csv_path = '/home/offsec/llm/USMA/adv_rob/CICIDS2017_preprocessed.csv'  # Adjusted CSV path
data = pd.read_csv(csv_path, header=0, low_memory=False)

print('Cleaning data...')
data.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
data.dropna(inplace=True)

# Dropping categorical columns (IP addresses, ports, protocol, and timestamp)
print('Dropping categorical columns...')
categorical_cols = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Protocol', 'Timestamp']
data.drop(columns=categorical_cols, inplace=True, errors='ignore')

print('Extracting features and labels...')
X = data.drop(columns=['Label']).values
y = data['Label'].values

print('Encoding labels...')
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print('Splitting data...')
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

print('Scaling features...')
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print('Converting data to PyTorch tensors...')
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(DEVICE)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(DEVICE)

print('Creating data loaders...')
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

print('Initializing model...')
input_size = X_train.shape[1]
num_classes = len(label_encoder.classes_)
model = Net(input_size, num_classes).to(DEVICE)

print('Setting up training...')
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print('Starting training loop...')
for epoch in range(10):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/10], Loss: {avg_loss:.4f}')

print('Evaluating model...')
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Test Accuracy: {accuracy * 100:.2f}%')
