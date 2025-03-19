import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
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
csv_path = '/home/offsec/llm/USMA/adv_rob/CICIDS2017_preprocessed.csv'
data = pd.read_csv(csv_path, header=0, low_memory=False)

print('Cleaning data...')
data.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
data.dropna(inplace=True)

# Dropping categorical columns
print('Dropping categorical columns...')
categorical_cols = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Protocol', 'Timestamp']
data.drop(columns=categorical_cols, inplace=True, errors='ignore')

# Cap classes to specific sample size
samples_per_class = 5000  # Adjust this number as needed
print(f'Capping each class to {samples_per_class} samples...')
data_capped = data.groupby('Label').apply(lambda x: x.sample(n=min(len(x), samples_per_class), random_state=42)).reset_index(drop=True)

print('Extracting features and labels...')
X = data_capped.drop(columns=['Label']).values
y = data_capped['Label'].values

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

train_losses = []
train_accuracies = []

print('Starting training loop...')
for epoch in range(10):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    train_losses.append(avg_loss)
    train_accuracies.append(accuracy)

    print(f'Epoch [{epoch+1}/10], Loss: {avg_loss:.4f}, Accuracy: {accuracy*100:.2f}%')

print('Evaluating model...')
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

accuracy = (torch.tensor(y_pred) == torch.tensor(y_true)).float().mean().item()
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Plotting loss and accuracy
plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, '-o', label='Loss')
plt.plot(train_accuracies, '-o', label='Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.title('Training Loss and Accuracy')
plt.legend()

# Confusion matrix
plt.subplot(1, 2, 2)
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.tight_layout()
plt.savefig('training_results.png')

# Classification report
print('Classification Report:\n', classification_report(y_true, y_pred, target_names=label_encoder.classes_))
