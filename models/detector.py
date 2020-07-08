import torch.nn as nn

class Detector(nn.Module):
    def __init__(self, in_features, out_features=2):
        super(Detector, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        scores = self.layers(x)
        return scores


