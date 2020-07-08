import torch.nn as nn
from utils import batch_accuracy


def combine(robust_features, nr_features):
    pass


# def get_combiner_loss(robust_features, nr_features, labels, combiner):
#     combine_logits = combiner(robust_features, nr_features)
#     criterion = nn.CrossEntropyLoss()
#     loss = criterion(combine_logits, labels)
#     acc = batch_accuracy(combine_logits, labels)
#     return loss, acc

def get_combiner_loss(logits, labels, combiner):
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, labels)
    acc = batch_accuracy(logits, labels)
    return loss, acc