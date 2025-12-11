import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def load_data():
    print("Loading data...")
    with open('X_train.pkl', 'rb') as f:
        X_train = pickle.load(f)
    with open('X_test.pkl', 'rb') as f:
        X_test = pickle.load(f)
    with open('y_train.pkl', 'rb') as f:
        y_train = pickle.load(f)
    with open('y_test.pkl', 'rb') as f:
        y_test = pickle.load(f)
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def preprocess_data(X_train, X_test):
    print("Preprocessing data...")
    
    # Handle 'Visual Condition' column if it exists and is categorical
    if 'Visual Condition' in X_train.columns:
        le = LabelEncoder()
        # Fit on train, transform both
        # Ensure we handle all possible categories if known, otherwise fit on combined or just train
        # Assuming Train has all categories
        X_train['Visual Condition'] = le.fit_transform(X_train['Visual Condition'].astype(str))
        # Handle unseen labels in test if any (simple approach: map to most frequent or error)
        # For this dataset, we assume 'Poor', 'Good', 'Medium' are present
        X_test['Visual Condition'] = X_test['Visual Condition'].astype(str).map(lambda s: le.transform([s])[0] if s in le.classes_ else 0)
        
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled

from cable_model import CableFailureModel

def train_model(model, train_loader, criterion, optimizer, num_epochs=60):
    print("Starting training...")
    train_losses = []
    train_accs = []
    
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
        
    return train_losses, train_accs

def evaluate_model(model, test_loader, y_test_original):
    print("\nEvaluating model...")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            
    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='weighted')
    rec = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)
    
    print("===========================================================")
    print("Model Evaluation Metrics")
    print("===========================================================")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    
    return cm, acc, prec, rec, f1

def plot_metrics(train_losses, train_accs, cm):
    # Plot Loss and Accuracy
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.title('Training Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy', color='green')
    plt.title('Training Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    print("Saved training_curves.png")
    
    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    print("Saved confusion_matrix.png")

def main():
    # 1. Load Data
    X_train, X_test, y_train, y_test = load_data()
    
    # 2. Preprocess
    # Ensure y is integer
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    
    X_train_scaled, X_test_scaled = preprocess_data(X_train, X_test)
    
    # Convert to Tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)
    
    # DataLoaders
    batch_size = 32
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 3. Build Model
    model = CableFailureModel(input_size=6, num_classes=3)
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 4. Train
    train_losses, train_accs = train_model(model, train_loader, criterion, optimizer, num_epochs=60)
    
    # 5. Evaluate
    cm, acc, prec, rec, f1 = evaluate_model(model, test_loader, y_test)
    
    # 6. Save Model
    torch.save(model.state_dict(), 'model_cable_failure.pt')
    print("\nModel saved as model_cable_failure.pt")
    
    # 7. Plot
    plot_metrics(train_losses, train_accs, cm)

if __name__ == "__main__":
    main()
