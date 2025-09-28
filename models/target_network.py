import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class TargetNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(TargetNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x, weights):
        x = F.conv2d(x, weights['conv1.weight'], weights['conv1.bias'], padding=1)
        x = F.relu(x)
        x = self.pool(x)
        
        x = F.conv2d(x, weights['conv2.weight'], weights['conv2.bias'], padding=1)
        x = F.relu(x)
        x = self.pool(x)
        
        x = x.view(x.size(0), -1) 
        
        x = F.linear(x, weights['fc1.weight'], weights['fc1.bias'])
        x = F.relu(x)
        
        x = F.linear(x, weights['fc2.weight'], weights['fc2.bias'])
        return x

    def get_param_dict(self):
        params = OrderedDict()
        for name, param in self.named_parameters():
            params[name] = param.shape
        return params
