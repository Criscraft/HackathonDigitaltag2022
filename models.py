import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class Network(nn.Module):
    
    def __init__(self):
        super().__init__()
        #define network layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64, 3)
        
        #initialize layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight)
                init.constant_(m.bias, 0)
        

    def forward(self, x):
        #implementation of the forward pass
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x