import torch
import torch.nn as nn

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
