import torch
import torch.nn as nn

class SDFS(nn.Module):
    def __init__(self, static_input_size, dynamic_input_size, output_size):
        super(SDFS, self).__init__()
        self.fc1 = nn.Linear(static_input_size + dynamic_input_size, 128)
        self.fc2 = nn.Linear(128, output_size)
    
    def forward(self, static_input, dynamic_features):
        x = torch.cat((static_input, dynamic_features), dim=1)
        x = torch.relu(self.fc1(x))
        output = self.fc2(x)
        return output