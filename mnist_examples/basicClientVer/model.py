import torch
from torch import nn
import torch.nn.functional as f

class MyNet(torch.nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.layer = 1000
        self.fc1 = torch.nn.Linear(28*28, self.layer)
        self.fc2 = torch.nn.Linear(self.layer, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)

        return f.log_softmax(x, dim=1)
