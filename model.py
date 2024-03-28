import torch as T
import torch.nn as nn
import torch.optim as optim

STATE_SPACE = 36
ACTION_SPACE = 3

ACTION_LOW = -1
ACTION_HIGH = 1

class DuellingDQN(nn.Module):
    
    def __init__(self, input_dim=STATE_SPACE, output_dim=ACTION_SPACE):
        super(DuellingDQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=128)
        self.layer_norm1 = nn.LayerNorm(128)
        
        
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64)
        self.layer_norm2 = nn.LayerNorm(64)
        
        
        self.lstm3 = nn.LSTM(input_size=64, hidden_size=32)
        self.layer_norm3 = nn.LayerNorm(32)
        
        
        self.V = nn.Linear(32, 1)
        self.A = nn.Linear(32, self.output_dim)
       
        
        self.relu = nn.ReLU()
        
    def forward(self, state):
        
        lstm1_output, _ = self.lstm1(state)
        x = self.layer_norm1(self.relu(lstm1_output))

        lstm2_output, _ = self.lstm2(x)
        x = self.layer_norm2(self.relu(lstm2_output))

        lstm3_output, _ = self.lstm3(x)
        x = self.layer_norm3(self.relu(lstm3_output))
        
        V = self.relu(self.V(x))
        A = self.relu(self.A(x))
        
        x = V + A - A.mean()
        
        return x