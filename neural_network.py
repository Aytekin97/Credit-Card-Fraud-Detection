import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Check if GPU is available and assign device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------
# Data Preparation & Resampling
# -------------------------------

# Load the dataset
data = pd.read_csv("PS_20174392719_1491204439457_log.csv")

# Drop unnecessary columns
data_cleaned = data.drop(columns=["isFlaggedFraud", "nameOrig", "nameDest"])

# Encode the 'type' column using LabelEncoder
label_encoder = LabelEncoder()
data_cleaned["type"] = label_encoder.fit_transform(data_cleaned["type"])

# Define features (X) and target (y)
X = data_cleaned.drop(columns=["isFraud"])
y = data_cleaned["isFraud"]

# Split the data into training and testing sets (resampling will be applied on training only)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    stratify=y,
                                                    random_state=42)

print("Original training class distribution:")
print(y_train.value_counts())

# STEP 1: Undersample the majority class
rus = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
X_train_under, y_train_under = rus.fit_resample(X_train, y_train)
print("\nAfter undersampling, training class distribution:")
print(pd.Series(y_train_under).value_counts())

# STEP 2: Use SMOTE to oversample the minority class so that both classes become balanced
smote = SMOTE(sampling_strategy=1.0, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_under, y_train_under)
print("\nAfter SMOTE, resampled training class distribution:")
print(pd.Series(y_train_resampled).value_counts())

# Standardize the features (important for neural networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Convert arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_resampled.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Create a dataset and DataLoader for training
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

# We'll create a validation split from the training dataset (20% for validation)
val_fraction = 0.2
num_train = len(train_dataset)
indices = list(range(num_train))
split = int(np.floor(val_fraction * num_train))
np.random.seed(42)
np.random.shuffle(indices)
train_idx, val_idx = indices[split:], indices[:split]
train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)

train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler)
val_loader = DataLoader(train_dataset, batch_size=32, sampler=val_sampler)

# -------------------------------
# Define the Neural Network Model
# -------------------------------

class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x

input_dim = X_train_tensor.shape[1]
model = SimpleNN(input_dim).to(device)
print(model)

# Loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------------------------------
# Training Loop with Early Stopping
# -------------------------------

num_epochs = 100
patience = 10
best_val_loss = np.inf
patience_counter = 0

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_X.size(0)
    
    epoch_loss = running_loss / len(train_idx)
    train_losses.append(epoch_loss)
    
    # Validation phase
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            running_val_loss += loss.item() * batch_X.size(0)
    epoch_val_loss = running_val_loss / len(val_idx)
    val_losses.append(epoch_val_loss)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
    
    # Early stopping check
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        patience_counter = 0
        best_model_state = model.state_dict()  # Save best model
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

# Load the best model state
model.load_state_dict(best_model_state)

# -------------------------------
# Evaluate on the Test Set
# -------------------------------

model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor.to(device))
    test_loss = criterion(test_outputs, y_test_tensor.to(device)).item()
    y_pred_prob = test_outputs.cpu().numpy()
    y_pred = (y_pred_prob > 0.5).astype(int)

print("\nTest Loss:", test_loss)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -------------------------------
# Plot Training History
# -------------------------------
plt.figure(figsize=(12, 5))

plt.subplot(1,2,1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
plt.legend()

plt.subplot(1,2,2)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over Epochs (Zoomed)")
plt.ylim(min(val_losses)-0.01, max(val_losses)+0.01)
plt.legend()

plt.tight_layout()
plt.show()
