import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

def get_brittle_loss(normal_rep, adv_rep, label, margin, normal_logits,
                     similarity, alpha, acc_first, global_pos_sim=False):
    # Xentropy : accurate for normal examples
    criterion = nn.CrossEntropyLoss()
    useful_loss = criterion(normal_logits, label)
    # normal_rep := non-robust features of gt label
    # adv_rep := non-robust features of attack's target class
    if global_pos_sim:
        pass
    else:
        pos_sim = 1
    inconsistent_loss = (margin - pos_sim + similarity(normal_rep, adv_rep)).clamp(min = 0).mean()
    return useful_loss + acc_first * alpha * inconsistent_loss, useful_loss, inconsistent_loss


def get_robust_loss(normal_rep, label, fc):
    robust_loss = F.cross_entropy(fc(normal_rep), label)
    return robust_loss



