"""
Utilities. Partially based on code from
https://github.com/modestyachts/CIFAR-10.1
"""
import io
import json
import os
import pickle

import numpy as np
import pathlib

from models.wideresnet import WideResNet
from models.shake_shake import ShakeNet
from models.cifar_resnet import ResNet

import torch
from torch.nn import Sequential, Module
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import pdb


cifar10_label_names = ['airplane', 'automobile', 'bird',
                       'cat', 'deer', 'dog', 'frog', 'horse',
                       'ship', 'truck']


def get_model(name, num_classes=10, normalize_input=False):
    name_parts = name.split('-')
    if name_parts[0] == 'wrn':
        depth = int(name_parts[1])
        widen = int(name_parts[2])
        model = WideResNet(
            depth=depth, num_classes=num_classes, widen_factor=widen)
        
    elif name_parts[0] == 'ss':
        model = ShakeNet(dict(depth=int(name_parts[1]),
                              base_channels=int(name_parts[2]),
                              shake_forward=True, shake_backward=True,
                              shake_image=True, input_shape=(1, 3, 32, 32),
                              n_classes=num_classes,
                              ))
    elif name_parts[0] == 'resnet':
        model = ResNet(num_classes=num_classes, depth=int(name_parts[1]))
    else:
        raise ValueError('Could not parse model name %s' % name)

    if normalize_input:
        model = Sequential(NormalizeInput(), model)

    return model


class NormalizeInput(Module):
    def __init__(self, mean=(0.4914, 0.4822, 0.4465),
                 std=(0.2023, 0.1994, 0.2010)):
        super().__init__()

        self.register_buffer('mean', torch.Tensor(mean).reshape(1, -1, 1, 1))
        self.register_buffer('std', torch.Tensor(std).reshape(1, -1, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std

# TODO: decide whether to remove all code below this line
# Should we add some description of how these files
# were obtained?
def load_tinyimage_subset(other_data_path,
                          version_string='v7'):
    image_data_filename = 'tinyimage_subset_data'
    if version_string != '':
        image_data_filename += '_' + version_string
    image_data_filename += '.pickle'
    image_data_filepath = os.path.abspath(os.path.join(other_data_path, image_data_filename))
    indices_filename = 'tinyimage_subset_indices'
    if version_string != '':
        indices_filename += '_' + version_string
    indices_filename += '.json'
    indices_filepath = os.path.abspath(os.path.join(other_data_path, indices_filename))
    print('Loading indices from file {}'.format(indices_filepath))
    assert pathlib.Path(indices_filepath).is_file()
    print('Loading image data from file {}'.format(image_data_filepath))
    assert pathlib.Path(image_data_filepath).is_file()
    with open(indices_filepath, 'r') as f:
        indices = json.load(f)
    with open(image_data_filepath, 'rb') as f:
        image_data = pickle.load(f)
    num_entries = 0
    for kw, kw_indices in indices.items():
        for entry in kw_indices:
            assert entry['tinyimage_index'] in image_data
            num_entries += 1
    assert num_entries == len(image_data)
    return indices, image_data


def load_cifar10_by_keyword(unique_keywords=True, version_string='v7'):
    cifar10_keywords = load_cifar10_keywords(unique_keywords=unique_keywords,
                                             lists_for_unique=True,
                                             version_string=version_string)
    cifar10_by_keyword = {}
    for ii, keyword_entries in enumerate(cifar10_keywords):
        for entry in keyword_entries:
            cur_keyword = entry['nn_keyword']
            if not cur_keyword in cifar10_by_keyword:
                cifar10_by_keyword[cur_keyword] = []
            cifar10_by_keyword[cur_keyword].append(ii)
    return cifar10_by_keyword


def load_cifar10_keywords(other_data_path, 
                          unique_keywords=True,
                          lists_for_unique=False,
                          version_string='v7'):
    filename = 'cifar10_keywords'
    if unique_keywords:
        filename += '_unique'
    if version_string != '':
        filename += '_' + version_string
    filename += '.json'
    keywords_filepath = os.path.abspath(os.path.join(other_data_path, filename))
    print('Loading keywords from file {}'.format(keywords_filepath))
    assert pathlib.Path(keywords_filepath).is_file()
    with open(keywords_filepath, 'r') as f:
        cifar10_keywords = json.load(f)
    if unique_keywords and lists_for_unique:
        result = []
        for entry in cifar10_keywords:
            result.append([entry])
    else:
        result = cifar10_keywords
    assert len(result) == 60000
    return result


def load_distances_to_cifar10(version_string='v7'):
    data_path = os.path.join(os.path.dirname(__file__), '../data/')
    filename = 'tinyimage_cifar10_distances'
    if version_string != '':
        filename += '_' + version_string
    filename += '.json'
    filepath = os.path.abspath(os.path.join(data_path, filename))
    print('Loading distances from file {}'.format(filepath))
    assert pathlib.Path(filepath).is_file()
    with open(filepath, 'r') as f:
        tmp = json.load(f)
    if version_string == 'v4':
        assert len(tmp) == 372131
    elif version_string == 'v6':
        assert len(tmp) == 1646248
    elif version_string == 'v7':
        assert len(tmp) == 589711
    result = {}
    for k, v in tmp.items():
        result[int(k)] = v
    return result


def load_new_test_data_indices(version_string='v7'):
    data_path = os.path.join(os.path.dirname(__file__), '../data/')
    filename = 'cifar10.1'
    if version_string == '':
        version_string = 'v7'
    if version_string in ['v4', 'v6', 'v7']:
        filename += '_' + version_string
    else:
        raise ValueError('Unknown dataset version "{}".'.format(version_string))
    ti_indices_data_path = os.path.join(os.path.dirname(__file__), '../data/')
    ti_indices_filename = 'cifar10.1_' + version_string + '_ti_indices.json'
    ti_indices_filepath = os.path.abspath(os.path.join(ti_indices_data_path, ti_indices_filename))
    print('Loading Tiny Image indices from file {}'.format(ti_indices_filepath))
    assert pathlib.Path(ti_indices_filepath).is_file()
    with open(ti_indices_filepath, 'r') as f:
        tinyimage_indices = json.load(f)
    assert type(tinyimage_indices) is list
    if version_string == 'v6' or version_string == 'v7':
        assert len(tinyimage_indices) == 2000
    elif version_string == 'v4':
        assert len(tinyimage_indices) == 2021
    return tinyimage_indices


def adjust_learning_rate(optimizer, epoch, lr_params):
    """decrease the learning rate"""
    lr = lr_params['lr']
    schedule = lr_params['lr_schedule']
    epochs = lr_params['epochs']
    # schedule from TRADES repo (different from paper due to bug there)
    if schedule == 'trades':
        if epoch >= 0.75 * epochs:
            lr = lr * 0.1
    # schedule as in TRADES paper
    elif schedule == 'trades_fixed':
        if epoch >= 0.75 * epochs:
            lr = lr * 0.1
        if epoch >= 0.9 * epochs:
            lr = lr * 0.01
        if epoch >= epochs:
            lr = lr * 0.001
    # cosine schedule
    elif schedule == 'cosine':
        lr = lr * 0.5 * (1 + np.cos((epoch - 1) / epochs * np.pi))
    # schedule as in WRN paper
    elif schedule == 'wrn':
        if epoch >= 0.3 * epochs:
            lr = lr * 0.2
        if epoch >= 0.6 * epochs:
            lr = lr * 0.2 * 0.2
        if epoch >= 0.8 * epochs:
            lr = lr * 0.2 * 0.2 * 0.2
    else:
        raise ValueError('Unkown LR schedule %s' % schedule)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def strip_data_parallel(s):
    if s.startswith('module'):
        return s[len('module.'):]
    else:
        return s


def prepare_model(model_path, arch, use_cuda=False, parallel=False):
    checkpoint = torch.load(model_path)
    state_dict = checkpoint.get('state_dict', checkpoint)
    num_classes = checkpoint.get('num_classes', 10)
    normalize_input = checkpoint.get('normalize_input', False)
    model = get_model(arch, num_classes=num_classes,
                      normalize_input=normalize_input)
    state_dict = {strip_data_parallel(k): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    if parallel:
        model = torch.nn.DataParallel(model)
    if use_cuda:
        model = model.cuda()
        cudnn.benchmark = True
    return model

def load_model(model, ckpt_path, use_cuda=True, parallel=False):
    ckpt = torch.load(ckpt_path)
    state_dict = {strip_data_parallel(k): v for k, v in ckpt.items()}
    model.load_state_dict(state_dict)
    if parallel:
        model = torch.nn.DataParallel(model)
    if use_cuda:
        model = model.cuda()
        cudnn.benchmark = True
    return model


def batch_accuracy(scores, labels):
    _, predicted = scores.max(1)
    accuracy = predicted.eq(labels).sum().item() / labels.size(0)
    return accuracy

class TotalAccuracy():
    def __init__(self):
        self.total = 0
        self.correct = 0

    def update(self, scores, labels):
        _, predicted = scores.max(1)
        correct = predicted.eq(labels).sum().item()
        self.correct += correct
        self.total += len(scores)

    def get_acc(self):
        return  self.correct / self.total



def combine_pgd(disen_model, X, y, epsilon=8/255, num_steps=30, step_size=0.01, random_start=True):
    perturbation = torch.zeros_like(X, requires_grad=True)

    if random_start:
        perturbation = torch.rand_like(X, requires_grad=True)
        perturbation.data = perturbation.data * 2 * epsilon - epsilon

    opt = optim.SGD([perturbation], lr=1e-3)
    for _ in range(num_steps):
        opt.zero_grad()
        #pdb.set_trace()
        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(disen_model.com_score(X + perturbation), y)

        loss.backward()

        perturbation.data = (
            perturbation + step_size * perturbation.grad.detach().sign()).clamp(
            -epsilon, epsilon)
        perturbation.data = torch.min(torch.max(perturbation.detach(), -X),
                                      1 - X)  # clip X+delta to [0,1]
        X_pgd = Variable(torch.clamp(X.data + perturbation.data, 0, 1.0),
                         requires_grad=False)

    return X_pgd


