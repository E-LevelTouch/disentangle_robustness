from disentangle_training import eval_normal, get_loader, DisentangleModel, eval_parallel
from disentangle_training import get_model
from models.combiner import Combiner
from models.detector import Detector
import torch
from torchvision import transforms
import torch.nn as nn


r_encoder = get_model('wrn-28-10')
nr_encoder = get_model('wrn-28-10')
detector = Detector(in_features=640 + 640)
combiner = Combiner(in_features=640 + 640, og_features=640)
model = DisentangleModel(robust_encoder=r_encoder, nr_encoder=nr_encoder, detector=detector, combiner=combiner)
checkpoint = torch.load('ckpt_epoch_42.pt')
model.load_state_dict(checkpoint)
model = nn.DataParallel(model).cuda()


transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
def collate_fn(data):
    images, robust_features, labels = list(zip(*data))
    images = torch.stack(images, dim=0)
    robust_features = torch.stack(robust_features, dim=0)
    labels = torch.tensor(labels)
    return images, robust_features, labels
dataset_params = {'root': './data', 'transform': transform, 'robust_path':'./data/robust_features_test.pkl'}
loader_params = {'batch_size': 128, 'shuffle': True,
                 'workers': 4, 'collate_fn':collate_fn}

loader = get_loader(dataset_params=dataset_params,loader_params=loader_params, train=False)
#eval_normal(model, loader)
eval_parallel(model, loader)
