from torch import nn

class MinimalModel(nn.Module):
    """
    Minimal pytorch model, for use as example
    """

    def __init__(self):
        super(MinimalModel, self).__init__()
        self.fc1 = nn.Linear(3, 5)
        self.fc2 = nn.Linear(5, 4)

    def forward(self, x):
        x = self.fc2(self.fc1(x))
        return x