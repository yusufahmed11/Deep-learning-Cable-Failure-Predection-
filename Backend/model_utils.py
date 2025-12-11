import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Try to import torch, but handle failure gracefully
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
except OSError:
    TORCH_AVAILABLE = False

# Define Model Architecture (Must match training)
if TORCH_AVAILABLE:
    class CableFailureModel(nn.Module):
        def __init__(self, input_size=6, num_classes=3):
            super(CableFailureModel, self).__init__()
            
            # Layer 1: Linear(6 -> 64) -> BN -> ReLU -> Dropout(0.3)
            self.layer1 = nn.Sequential(
                nn.Linear(input_size, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
            
            # Layer 2: Linear(64 -> 128) -> BN -> ReLU -> Dropout(0.3)
            self.layer2 = nn.Sequential(
                nn.Linear(64, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
            
            # Layer 3: Linear(128 -> 64) -> BN -> ReLU -> Dropout(0.3)
            self.layer3 = nn.Sequential(
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
            
            # Layer 4: Linear(64 -> 32) -> ReLU
            self.layer4 = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU()
            )
            
            # Layer 5: Linear(32 -> 16) -> ReLU
            self.layer5 = nn.Sequential(
                nn.Linear(32, 16),
                nn.ReLU()
            )
            
            # Output Layer: Linear(16 -> 3)
            self.output_layer = nn.Linear(16, num_classes)
            
        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.layer5(x)
            x = self.output_layer(x)
            return x
else:
    CableFailureModel = None

# Global variables
model = None
scaler = None
columns = None

def load_resources():
    global model, scaler, columns
    
    print("Loading resources...")
    
    # 1. Load Training Data to fit Scaler
    try:
        with open('../X_train.pkl', 'rb') as f:
            X_train = pickle.load(f)
        
        columns = X_train.columns.tolist()
        print(f"Training columns: {columns}")
        
        # Fit Scaler
        scaler = StandardScaler()
        scaler.fit(X_train)
        print("Scaler fitted.")
        
    except Exception as e:
        print(f"Error loading X_train.pkl: {e}")
        # Fallback scaler if file missing
        scaler = StandardScaler()
        # Mock fit with some dummy data to avoid errors
        scaler.fit(np.array([[10, 0.5, 0.5, 400, 0, 0], [30, 0.8, 0.8, 600, 1, 0]]))

    # 2. Load Model
    if TORCH_AVAILABLE:
        try:
            model = CableFailureModel(input_size=6, num_classes=3)
            model.load_state_dict(torch.load('../model_cable_failure.pt'))
            model.eval()
            print("Model loaded.")
        except Exception as e:
            print(f"Error loading model: {e}")
            model = None
    else:
        print("Torch not available. Using fallback logic.")
        model = None

def preprocess_input(data: pd.DataFrame):
    # Data must have columns: Age, Partial Discharge, Neutral Corrosion, Loading, Visual Condition
    # We need to transform 'Visual Condition' to OHE: Visual Condition_Medium, Visual Condition_Poor
    
    # Create empty OHE columns
    data['Visual Condition_Medium'] = 0
    data['Visual Condition_Poor'] = 0
    
    # Map Visual Condition
    if 'Visual Condition' in data.columns:
        mask_medium = data['Visual Condition'].astype(str).str.lower() == 'medium'
        mask_poor = data['Visual Condition'].astype(str).str.lower() == 'poor'
        
        data.loc[mask_medium, 'Visual Condition_Medium'] = 1
        data.loc[mask_poor, 'Visual Condition_Poor'] = 1
        
        # Drop original
        data = data.drop(columns=['Visual Condition'])
    
    # Ensure all required columns exist and are in correct order
    required_cols = ['Age', 'Partial Discharge', 'Neutral Corrosion', 'Loading', 'Visual Condition_Medium', 'Visual Condition_Poor']
    
    # Fill missing with 0
    for col in required_cols:
        if col not in data.columns:
            data[col] = 0
            
    # Reorder
    data = data[required_cols]
    
    # Scale
    data_scaled = scaler.transform(data)
    
    if TORCH_AVAILABLE:
        return torch.tensor(data_scaled, dtype=torch.float32)
    else:
        return data_scaled

def get_explanation(row):
    reasons = []
    if row.get('Partial Discharge', 0) > 0.6:
        reasons.append("High Partial Discharge levels detected.")
    if row.get('Neutral Corrosion', 0) > 0.8:
        reasons.append("Severe Neutral Corrosion indicates degradation.")
    if row.get('Loading', 0) > 550:
        reasons.append("Cable is operating under high load.")
    if str(row.get('Visual Condition', '')).lower() == 'poor':
        reasons.append("Visual condition is rated as Poor.")
    if row.get('Age', 0) > 25:
        reasons.append("Cable exceeds recommended operational age.")
        
    if not reasons:
        reasons.append("Parameters are within normal operating limits.")
        
    return reasons

def predict_row(row_dict):
    # Row dict has raw values
    df = pd.DataFrame([row_dict])
    
    # Preprocess
    input_data = preprocess_input(df.copy())
    
    if model and TORCH_AVAILABLE:
        # Predict with Model
        with torch.no_grad():
            outputs = model(input_data)
            _, predicted = torch.max(outputs.data, 1)
            cls = predicted.item()
            
            if cls == 0:
                risk = "Low"
                hi = 5
            elif cls == 1:
                risk = "Medium"
                hi = 3
            else:
                risk = "High"
                hi = 1
    else:
        # Fallback Logic (Deterministic)
        # Based on the same rules as frontend
        score = 5
        
        pd_val = float(row_dict.get('Partial Discharge', 0))
        corr_val = float(row_dict.get('Neutral Corrosion', 0))
        load_val = float(row_dict.get('Loading', 0))
        age_val = float(row_dict.get('Age', 0))
        vis_val = str(row_dict.get('Visual Condition', '')).lower()
        
        if pd_val > 0.5: score -= 1
        if pd_val > 0.8: score -= 1
        
        if corr_val > 0.6: score -= 1
        if corr_val > 0.9: score -= 1
        
        if vis_val == 'medium': score -= 1
        if vis_val == 'poor': score -= 2
        
        if load_val > 500: score -= 1
        if load_val > 650: score -= 1
        
        if age_val > 30: score -= 1
        
        hi = max(1, min(5, score))
        
        if hi >= 4: risk = "Low"
        elif hi == 3: risk = "Medium"
        else: risk = "High"
            
    explanation = get_explanation(row_dict)
    
    return {
        "health_index": hi,
        "risk_level": risk,
        "explanation": explanation
    }
