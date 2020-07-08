from utils import get_model
from disentangle_training import get_loader
from torchvision import transforms
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import pdb


model = get_model('wrn-28-10')
transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
dataset_params = {'root': './data', 'transform': transform, 'robust_path':'./data/robust_features_train.pkl'}
# TODO: FIX COLLOATE FN
def collate_fn(data):
    images,  labels = list(zip(*data))
    images = torch.stack(images, dim=0)
    #robust_features = torch.stack(robust_features, dim=0)
    labels = torch.tensor(labels)
    return images, labels
loader_params = {'batch_size': 128, 'shuffle': True,
                 'workers': 0, 'collate_fn':collate_fn}
loader = get_loader(dataset_params, loader_params)
device = torch.device("cuda")
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001,
                          momentum=0.9,
                          weight_decay=5e-4,
                          nesterov=False)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

dataset = CIFAR10(root='./data', train=True, transform=transform)
test_loader = DataLoader(dataset, batch_size=128, shuffle=True)

model.to(device)
for epoch in range(10):
    progress = tqdm(test_loader)
    # for images, normal_robust_rep, labels in progress:
    #     images, normal_robust_rep, labels = images.to(device), normal_robust_rep.to(device), labels.to(device)
    #     normal_nr_logits, normal_nr_rep = model(images, return_prelogit=True)
    #     xent = criterion(normal_nr_logits, labels)
    #     xent.backward()
    #     optimizer.step()
    #     progress.set_description('epoch %d ,xent:%.2f'%(epoch, xent.detach()))

    for images, labels in progress:
        #pdb.set_trace()
        images, labels = images.to(device), labels.to(device)
        normal_nr_logits, normal_nr_rep = model(images, return_prelogit=True)
        xent = criterion(normal_nr_logits, labels)

        optimizer.zero_grad()
        xent.backward()
        optimizer.step()

        _, predicted = torch.max(normal_nr_logits.data, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()

        progress.set_description('epoch %d ,xent:%.2f, acc %.3f'%(epoch, xent.item(), correct/total))

