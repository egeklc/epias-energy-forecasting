import torch
import numpy as np
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, horizon, lin_features=32, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size,
                            num_layers=num_layers, 
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0
                            )
        
        self.linear = nn.Linear(in_features=hidden_size, out_features=lin_features)
        self.activation = nn.LeakyReLU()
        self.fc = nn.Linear(in_features=lin_features, out_features=horizon)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.linear(out)
        out = self.activation(out)
        out = self.fc(out)
        return out
