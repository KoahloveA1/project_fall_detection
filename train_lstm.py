import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Hyperparameters
INPUT_SIZE = 38
HIDDEN_SIZE = 64
NUM_LAYERS = 2
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001
SEQ_LEN = 30

DATA_DIR = "/Users/ledangkhoa/do_an/dataset_tensors"
MODEL_PATH = "/Users/ledangkhoa/do_an/fall_lstm_best_v2.pt"

# Detect device (MPS for Apple Silicon, otherwise CPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Define Model
class FallDetectionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(FallDetectionLSTM, self).__init__()
        # Thêm Dropout 0.5 giữa các lớp LSTM để chống Overfit
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
        
        # Thêm các lớp Linear và Dropout
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        # Lấy output từ time step cuối cùng
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

def train_and_evaluate():
    # 2. Load Data
    print("Loading data...")
    X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
    y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
    X_val = np.load(os.path.join(DATA_DIR, 'X_val.npy'))
    y_val = np.load(os.path.join(DATA_DIR, 'y_val.npy'))
    
    # Convert to PyTorch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    
    # DataLoaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Calculate pos_weight for BCEWithLogitsLoss due to class imbalance
    # Cân bằng lại pos_weight = 2.0 để mô hình bớt nhạy cảm (ít báo động giả)
    pos_weight = torch.tensor([2.0], dtype=torch.float32).to(device)
    
    # 3. Init Model, Loss, Optimizer
    model = FallDetectionLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Thêm L2 Regularization (weight_decay) để chống Overfit
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # 4. Training Loop
    best_val_loss = float('inf')
    
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)
                
                # Sigmoid to get probability, then threshold at 0.5
                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.5).float()
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
                
        val_loss /= len(val_loader.dataset)
        
        # Calculate metrics
        acc = accuracy_score(all_labels, all_preds)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {acc:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            
    print(f"\nTraining complete. Best model saved to {MODEL_PATH}")
    
    # 5. Final Evaluation on best model
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    final_preds = []
    final_labels = []
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float().cpu().numpy()
            final_preds.extend(preds)
            final_labels.extend(batch_y.numpy())
            
    # Metrics
    accuracy = accuracy_score(final_labels, final_preds)
    precision = precision_score(final_labels, final_preds)
    recall = recall_score(final_labels, final_preds)
    f1 = f1_score(final_labels, final_preds)
    cm = confusion_matrix(final_labels, final_preds)
    
    print("\n--- FINAL EVALUATION METRICS ---")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
if __name__ == "__main__":
    train_and_evaluate()
