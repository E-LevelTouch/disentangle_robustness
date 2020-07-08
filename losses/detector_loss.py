import torch.nn as nn
import torch
import pdb
from utils import batch_accuracy

def permute(normal_rep, adv_rep):
    stacked_batch = torch.cat([normal_rep, adv_rep], dim=0)
    ids_shuffled = torch.randperm(len(normal_rep) + len(adv_rep))
    permuted_images = stacked_batch[ids_shuffled, :]
    labels = torch.tensor([int(i >= len(normal_rep)) for i in ids_shuffled])
    return permuted_images, labels


def get_detector_loss(normal_rep, adv_rep, detector):
    permuted_images, labels = permute(normal_rep, adv_rep)
    labels = labels.to(permuted_images.device)
    criterion = nn.CrossEntropyLoss()
    logits = detector(permuted_images)
    loss = criterion(logits, labels)
    acc = batch_accuracy(logits, labels)
    return loss, acc
