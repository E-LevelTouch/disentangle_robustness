from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torch.backends.cudnn as cudnn
import argparse
from utils import get_model
from tqdm import tqdm
import pickle
import os
import torch.nn as nn


def get_train_loader(dataset_params, loader_params):
    dataset = CIFAR10(root=dataset_params['root'], train=False, transform=dataset_params['transform'])
    loader = DataLoader(dataset, batch_size=loader_params['batch_size'], shuffle=loader_params['shuffle'],
                        num_workers=loader_params['workers'])
    return loader

class Dummy(nn.Module):
    def __init__(self, model):
        super(Dummy, self).__init__()
        self.model = model

    def forward(self, x, return_prelogit):
        return self.model(x, return_prelogit)


# def prepare_model(model_path, use_cuda, arch):
#     checkpoint = torch.load(model_path)
#     state_dict = checkpoint.get('state_dict', checkpoint)
#     num_classes = checkpoint.get('num_classes', 10)
#     normalize_input = checkpoint.get('normalize_input', False)
#     model = get_model(arch, num_classes=num_classes,
#                       normalize_input=normalize_input)
#     #model = Dummy(model)
#     if use_cuda:
#         model = torch.nn.DataParallel(model).cuda()
#         cudnn.benchmark = True
#         if not all([k.startswith('module') for k in state_dict]):
#             state_dict = {'module.' + k: v for k, v in state_dict.items()}
#     else:
#         def strip_data_parallel(s):
#             if s.startswith('module'):
#                 return s[len('module.'):]
#             else:
#                 return s
#
#         state_dict = {strip_data_parallel(k): v for k, v in state_dict.items()}
#
#     model.load_state_dict(state_dict)
#     return model

def prepare_model(model_path, arch, use_cuda=True, parallel=True):
    def strip_data_parallel(s):
        if s.startswith('module'):
            return s[len('module.'):]
        else:
            return s
    checkpoint = torch.load(model_path)
    state_dict = checkpoint.get('state_dict', checkpoint)
    num_classes = checkpoint.get('num_classes', 10)
    normalize_input = checkpoint.get('normalize_input', False)
    model = get_model(arch, num_classes=num_classes,
                      normalize_input=normalize_input)
    state_dict = {strip_data_parallel(k): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = Dummy(model)
    if parallel:
        model = torch.nn.DataParallel(model)
    if use_cuda:
        model = model.cuda()
        cudnn.benchmark = True
    return model


def extract_feature(loader, model, device, rep_path):
    reps = []
    for data, target in tqdm(loader):
        data, target = data.to(device), target.to(device)
        logits, data_rep = model(data, return_prelogit=True)
        # detach here to prevent cuda out of memory
        reps.append(data_rep.detach().cpu())
    pickle.dump(torch.cat(reps, dim=0), open(rep_path, 'wb'), pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data_loader
    parser.add_argument('--root',default='./data', type=str)
    parser.add_argument('--batch_size', default=1000, type=int)
    parser.add_argument('--shuffle', default=True, type=bool)
    parser.add_argument('--workers', default=10, type=int)

    # model
    parser.add_argument('--model_path', default='./rst_adv/checkpoint-epoch200.pt')
    parser.add_argument('--model', '-m', default='wrn-28-10', type=str, help='Name of the model')
    parser.add_argument('--output_suffix', default='', type=str, help='String to add to log filename')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training')

    args = parser.parse_args()
    #cuda_settings
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    dl_kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    # data settings
    transform = transforms.Compose([transforms.ToTensor()])
    dataset_params = {'root':args.root, 'transform':transform}
    loader_params = {'batch_size': args.batch_size, 'shuffle':args.shuffle, 'workers': 1}
    train_loader = get_train_loader(dataset_params=dataset_params, loader_params=loader_params)
    # model
    model = prepare_model(model_path=args.model_path, arch=args.model)
    extract_feature(loader=train_loader, model=model, device=device, rep_path=os.path.join(args.root, 'robust_features_test.pkl'))

