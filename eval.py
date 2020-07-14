from disentangle_training import eval_normal, get_loader, DisentangleModel, eval_adv,eval_two_stage, eval_robust
from disentangle_training import get_model
from models.combiner import Combiner
from models.detector import Detector
import torch
from torchvision import transforms
import torch.nn as nn
from utils import load_model


similarity = nn.CosineSimilarity()
loss_params = {'margin': 0.9, 'similarity': similarity, 'alpha': 1}
attack_params = {'eps': 0.031, 'step_size': 0.01, 'num_steps': 30}

r_encoder = get_model('wrn-28-10')
nr_encoder = get_model('wrn-28-10')
detector = Detector(in_features=640 + 640)
combiner = Combiner(in_features=640 + 640, og_features=640)
model = DisentangleModel(robust_encoder=r_encoder, nr_encoder=nr_encoder, detector=detector,
                         combiner=combiner, loss_params=loss_params, attack_params=attack_params)
#model = load_model(model, ckpt_path='output/proto/ckpts/ckpt_epoch_50.pt')
model = load_model(model, ckpt_path='ckpt_epoch_200.pt')
model.eval()
transform = transforms.Compose([
        transforms.ToTensor(),
    ])

def collate_fn(data):
    images, robust_features, labels = list(zip(*data))
    images = torch.stack(images, dim=0)
    robust_features = torch.stack(robust_features, dim=0)
    labels = torch.tensor(labels)
    return images, robust_features, labels
dataset_params = {'root': './data', 'transform': transform, 'robust_path':'./data/robust_features_test.pkl'}
loader_params = {'batch_size': 64, 'shuffle': True,
                 'workers': 4, 'collate_fn':collate_fn}
attack_params = {'eps': 0.031, 'step_size': 0.01, 'num_steps': 40}
loader = get_loader(dataset_params=dataset_params,loader_params=loader_params, train=False)
#eval_normal(model, loader)
#eval_adv(model, loader, attack_params=attack_params)
eval_robust(model, loader, attack_params=attack_params)

