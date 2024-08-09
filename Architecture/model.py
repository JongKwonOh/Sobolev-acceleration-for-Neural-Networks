import torch
import torch.nn as nn
import numpy as np

class FourierFeatureModel(nn.Module): # Fourier Feature
    def __init__(self, num_features=64):
        super(FourierFeatureModel, self).__init__()
        self.fc1 = nn.Linear(1, num_features)
        nn.init.normal_(self.fc1.weight, mean=0.0, std=1.0)
        self.fc2 = nn.Linear(num_features, num_features) 
        self.fc3 = nn.Linear(num_features, num_features) 
        self.fc4 = nn.Linear(num_features, 1) 
        self.act1 = nn.Tanh()
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
        
    def forward(self, x):
        x = torch.sin(2*np.pi*self.fc1(x))
        x = self.act1(self.fc2(x))
        x = self.act1(self.fc3(x))
        x = self.fc4(x)
        return x

class StandardModel(nn.Module): # Standard
    def __init__(self, num_features=64):
        super(StandardModel, self).__init__()
        self.fc1 = nn.Linear(1, num_features)
        self.fc2 = nn.Linear(num_features, num_features) 
        self.fc3 = nn.Linear(num_features, num_features) 
        self.fc4 = nn.Linear(num_features, 1) 
        self.act1 = nn.Tanh()
        
        nn.init.kaiming_uniform_(self.fc1.weight)
        nn.init.kaiming_uniform_(self.fc2.weight)
        nn.init.kaiming_uniform_(self.fc3.weight)
        nn.init.kaiming_uniform_(self.fc4.weight)
        
    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act1(self.fc2(x))
        x = self.act1(self.fc3(x))
        x = self.fc4(x)
        return x

class SIRENModel(nn.Module): # SIREN
    def __init__(self, num_features=64):
        super(SIRENModel, self).__init__()
        self.fc1 = nn.Linear(1, num_features)
        self.fc2 = nn.Linear(num_features, num_features) 
        self.fc3 = nn.Linear(num_features, num_features) 
        self.fc4 = nn.Linear(num_features, 1) 
        self.act1 = torch.sin
        
        nn.init.kaiming_uniform_(self.fc1.weight)
        nn.init.kaiming_uniform_(self.fc2.weight)
        nn.init.kaiming_uniform_(self.fc3.weight)
        nn.init.kaiming_uniform_(self.fc4.weight)
        
    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act1(self.fc2(x))
        x = self.act1(self.fc3(x))
        x = self.fc4(x)
        return x

def get_model(model_type, num_features):
    if model_type == "Standard":
        return StandardModel(num_features)
    elif model_type == "FourierFeature":
        return FourierFeatureModel(num_features)
    elif model_type == "SIREN":
        return SIRENModel(num_features)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

