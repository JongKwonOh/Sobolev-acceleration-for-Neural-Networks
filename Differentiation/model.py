import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, num_features=64):
        super(Model, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, num_features),
            nn.Tanh(),
            nn.Linear(num_features, num_features),
            nn.Tanh(),
            nn.Linear(num_features, num_features),
            nn.Tanh(),
            nn.Linear(num_features, num_features),
            nn.Tanh(),
            nn.Linear(num_features, 1)
        )

    def forward(self, x, y):
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        xy = torch.cat((x, y), dim=1)
        
        return self.fc(xy)
    
def get_model(num_features):
    return Model(num_features)