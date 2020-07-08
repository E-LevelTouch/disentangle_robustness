import torch.nn as nn
import torch
import pdb

class Combiner(nn.Module):
    def __init__(self, in_features, og_features, num_classes=10):
        super(Combiner, self).__init__()
        self.combine = nn.Linear(in_features, og_features)
        self.activation = nn.ReLU()
        self.fc = nn.Linear(og_features, num_classes)

    def forward(self, robust_rep, nr_rep) :
        combined_rep = self.activation(self.combine(torch.cat([robust_rep, nr_rep],dim=-1)))
        scores = self.fc(combined_rep)
        return scores