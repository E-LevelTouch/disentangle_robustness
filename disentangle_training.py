from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pickle
import argparse
from losses.encoder_loss import get_brittle_loss
from losses.detector_loss import get_detector_loss
from losses.combiner_loss import get_combiner_loss
from advertorch.attacks import LinfPGDAttack
import torch
from torchvision import transforms
from models.detector import Detector
from models.combiner import Combiner
from utils import get_model
import torch.optim as optim
import os
import logging
from tensorboardX import SummaryWriter
from tqdm import tqdm
from utils import adjust_learning_rate
import pdb
from utils import prepare_model, batch_accuracy, TotalAccuracy
from models.cifar_resnet import ResNet
import torch.backends.cudnn as cudnn



class DisentangleDataset(Dataset):
    def __init__(self, raw_dataset, robust_path):
        self.raw_dataset = raw_dataset
        self.robust_features = pickle.load(open(robust_path, 'rb'))

    def __getitem__(self, index):
        image, label = self.raw_dataset[index]
        robust_features = self.robust_features[index]
        return image, robust_features, label

    def __len__(self):
        return len(self.raw_dataset)


def get_loader(dataset_params, loader_params, train=True):
    raw_dataset = CIFAR10(root=dataset_params['root'], train=train,
                          transform=dataset_params['transform'])
    dataset = DisentangleDataset(raw_dataset, dataset_params['robust_path'])
    loader = DataLoader(dataset=dataset, batch_size=loader_params['batch_size'],shuffle=True,
                        num_workers=loader_params['workers'], collate_fn=loader_params['collate_fn'])
    return loader


def get_adv_images(model, images, attack_params):
    attack = LinfPGDAttack(model, loss_fn=nn.CrossEntropyLoss(), eps=attack_params['eps'],
                          nb_iter=attack_params['num_steps'], eps_iter=attack_params['step_size'],
                          rand_init=True)
    adv_images = attack.perturb(images)
    return adv_images

def log_loss(writer:SummaryWriter, loss_metric, iter):
    for key, value in loss_metric.items():
        writer.add_scalar(key, value, iter)


class DisentangleModel(nn.Module):
    def __init__(self, robust_encoder, nr_encoder, detector, combiner, attack_params, loss_params):
        super(DisentangleModel, self).__init__()
        self.robust_encoder = robust_encoder
        self.nr_encoder = nr_encoder
        self.detector = detector
        self.combiner = combiner
        self.criterion = nn.CrossEntropyLoss()
        self.attack_params = attack_params
        self.loss_params = loss_params

    def get_nr_loss(self, normal_rep, adv_rep, label, normal_logits):
        useful_loss = self.criterion(normal_logits, label)
        # todo: fix global_pos_sim
        pos_sim = 1
        inconsistent_loss = (self.loss_params['margin'] - pos_sim + self.loss_params['similarity'](normal_rep, adv_rep)).clamp(min=0).mean()
        return useful_loss + loss_params['alpha'] * inconsistent_loss, useful_loss, inconsistent_loss

    def get_adv_images(self, images):
        attack = LinfPGDAttack(self.nr_encoder, loss_fn=nn.CrossEntropyLoss(),eps=self.attack_params['eps'],
                               nb_iter=self.attack_params['num_steps'], eps_iter=self.attack_params['step_size'],
                               rand_init=True)
        adv_images = attack.perturb(images)
        return adv_images

    def permute(self, normal_rep, adv_rep):
        stacked_batch = torch.cat([normal_rep, adv_rep], dim=0)
        ids_shuffled = torch.randperm(len(normal_rep) + len(adv_rep))
        permuted_images = stacked_batch[ids_shuffled, :]
        labels = torch.tensor([int(i >= len(normal_rep)) for i in ids_shuffled])
        return permuted_images, labels

    def get_detector_loss(self, normal_rep, adv_rep):
        permuted_images, labels = self.permute(normal_rep, adv_rep)
        labels = labels.to(permuted_images.device)
        logits = self.detector(permuted_images)
        loss = self.criterion(logits, labels)
        acc = batch_accuracy(logits, labels)
        return loss, torch.tensor(acc).cuda()

    def robust_forword(self, x, return_prelogit=False):
        return self.robust_encoder(x, return_prelogit)

    def nr_forword(self, x, return_prelogit=False):
        return self.nr_encoder(x, return_prelogit)

    def combine_predict(self, r, nr):
        return self.combiner(r, nr)

    def combiner_data(self, score, label):
        loss = self.criterion(score, label)
        acc = torch.tensor(batch_accuracy(score, label)).cuda()
        return loss, acc

    def forward(self, x, labels, return_prelogit=True):
        normal_r_logits, normal_r_rep = self.robust_forword(x, return_prelogit)
        normal_nr_logits, normal_nr_rep = self.nr_forword(x, return_prelogit)

        adv_x = self.get_adv_images(x)
        adv_r_logits, adv_r_rep = self.robust_forword(adv_x, return_prelogit)
        adv_nr_logits, adv_nr_rep = self.nr_forword(adv_x, return_prelogit)

        normal_rep = torch.cat([normal_r_rep, normal_nr_rep], dim=-1)
        adv_rep = torch.cat([adv_r_rep, adv_nr_rep], dim=-1)
        dec_loss, dec_acc = self.get_detector_loss(normal_rep, adv_rep)

        combine_score = self.combine_predict(normal_r_rep, normal_nr_rep)
        com_loss, com_acc = self.combiner_data(combine_score, labels)

        nr_loss, nr_xent, nr_brittle = self.get_nr_loss(normal_nr_rep, adv_nr_rep, labels, normal_nr_logits)

        out = {'n_r_log':normal_r_logits, 'n_r_rep':normal_r_rep, 'n_nr_log':normal_nr_logits, 'n_nr_rep':normal_nr_rep,
               'a_r_log':adv_r_logits, 'a_r_rep':adv_r_rep, 'a_nr_log':adv_nr_logits, 'a_nr_rep':adv_nr_rep,
               'com_loss':com_loss, 'com_acc':torch.tensor(com_acc), 'dec_loss':dec_loss, 'dec_acc':torch.tensor(dec_acc)}

        return normal_r_logits, normal_r_rep, normal_nr_logits, normal_nr_rep, \
               adv_r_logits, adv_r_rep, adv_nr_logits, adv_nr_rep,\
               dec_loss, torch.tensor(dec_acc), com_loss, torch.tensor(com_acc), \
               nr_loss, nr_xent, nr_brittle


def train(model, train_loader, device, loss_params, optimizer, attack_params, epoch, writer:SummaryWriter):
    iter = len(train_loader) * (epoch - 1) + 1
    progress = tqdm(train_loader)
    for images, normal_robust_rep, labels in progress:
        try:
            #images, normal_robust_rep, labels = images.to(device), normal_robust_rep.to(device),labels.to(device)
            images, normal_robust_rep, labels = images.cuda(), normal_robust_rep.cuda(), labels.cuda()
            # todo : seperate non_robust func
            #adv_images = get_adv_images(model.module.nr_encoder, images, attack_params).cuda()

            #normal_r_logits, normal_robust_rep = model.robust_encoder(images, return_prelogit=True)
            #normal_nr_logits, normal_nr_rep = model.nr_encoder(images, return_prelogit=True)
            # normal_nr_logits, normal_nr_rep = model.module.nr_encoder(images, return_prelogit=True)
            # adv_nr_logits, adv_nr_rep  = model.module.nr_encoder(adv_images, return_prelogit=True)
            # adv_r_logits, adv_robust_rep = model.module.robust_encoder(adv_images, return_prelogit=True)

            normal_r_logits, normal_r_rep, normal_nr_logits, normal_nr_rep, adv_r_logits, adv_r_rep, adv_nr_logits, adv_nr_rep, detector_loss, det_acc, combine_loss, combine_acc, nr_loss, nr_xent, nr_brittle = model(images, labels)
            # normal_nr_logits, normal_nr_rep = out['nr_log'], out['nr_rep']
            # adv_r_logits, adv_robust_rep, adv_nr_logits, adv_nr_rep = out['r_log'], out['r_rep'], \
            #                                                           out['nr_log'], out['nr_rep']

            # nr_loss, nr_xent, nr_brittle = get_brittle_loss(normal_rep=normal_nr_rep, adv_rep=adv_nr_rep, label=labels,
            #                                                 margin=loss_params['margin'],
            #                                                 normal_logits=normal_nr_logits,
            #                                                 similarity=loss_params['similarity'],
            #                                                 alpha=loss_params['alpha'],
            #                                                 acc_first=int(lr_params['acc_first'] < epoch))

            nr_acc = batch_accuracy(normal_nr_logits, labels)
            nr_rob = batch_accuracy(adv_nr_logits, labels)

            # detector_loss, det_acc = out['dec_loss'], out['dec_acc']
            # combine_loss, com_acc = out['com_loss'], out['com_acc']
            #print(nr_loss.shape, detector_loss.shape, combine_loss.shape)
            if epoch > lr_params['nr_first']:
                final_loss = nr_loss.mean() + torch.mean(detector_loss) + torch.mean(combine_loss)
            else:
                final_loss = nr_loss.mean()
            #pdb.set_trace()
            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()
            # TODO: LOG accuracy of detector, combiner, nr
            loss_metric = {'final_loss':final_loss.detach(), 'detector_loss':detector_loss.mean(),
                           'combine_loss':combine_loss.mean(), 'nr_loss': nr_loss.mean(),
                           'nr_acc':nr_acc, 'nr_rob':nr_rob,
                           'det_acc':det_acc.mean(), 'com_acc':combine_acc.mean(),
                           'nr_xent':nr_xent.mean(), 'nr_brittle':nr_brittle.mean()}

            log_loss(writer, loss_metric, iter)
            progress.set_description('%d_ep:final_loss:%.2f,nr_loss:%.2f,dec_loss: %.2f,com_loss: %.2f,acc:%.2f,com:%.2f,det:%.2f'
                                     %(epoch, final_loss.item(), nr_loss.mean().item(), detector_loss.mean().item(),
                                       combine_loss.mean().item(), nr_acc, combine_acc.mean(), det_acc.mean()))
            iter += 1
        except Exception as e:
            print(e)


def eval_normal(model, loader):
    counter = TotalAccuracy()
    for images, normal_robust_rep, labels in tqdm(loader):
        try:
            images, normal_robust_rep, labels = images.cuda(), normal_robust_rep.cuda(),labels.cuda()
            normal_nr_logits, normal_nr_rep = model.nr_encoder(images, return_prelogit=True)
            scores = model.combiner(normal_robust_rep, normal_nr_rep)
            counter.update(scores, labels)
        except Exception as e:
            print(e)
    print(counter.get_acc())

def eval_parallel(model, loader):
    counter = TotalAccuracy()
    for images, normal_robust_rep, labels in tqdm(loader):
        try:
            images, normal_robust_rep, labels = images.cuda(), normal_robust_rep.cuda(),labels.cuda()
            scores = model(images)
            counter.update(scores, labels)
        except Exception as e:
            print(e)
    print(counter.get_acc())

def eval_adv(model, loader, loss_params, attack_params, epoch, writer):
    pass


def create_trial_out_dir(root):
    if not os.path.exists(root):
        os.makedirs(root)
    ckpt_dir = os.path.join(root, 'ckpts')
    summary_dir = os.path.join(root, 'summary')
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)
    if not os.path.exists(summary_dir):
        os.mkdir(summary_dir)
    return ckpt_dir, summary_dir


def collate_fn(data):
    images, robust_features, labels = list(zip(*data))
    images = torch.stack(images, dim=0)
    robust_features = torch.stack(robust_features, dim=0)
    labels = torch.tensor(labels)
    return images, robust_features, labels


def get_disentangle_model(model_params):
    if model_params.get('r_ckpt') != None:
        r_encoder = prepare_model(args.model_path, args.model)
    else:
        r_encoder = get_model(model_params['r_arch'])
    # TODO: Fix Load Model
    nr_encoder = get_model(model_params['nr_arch'])
    detector = Detector(in_features=model_params['r_dim'] + model_params['nr_dim'])
    combiner = Combiner(in_features=model_params['r_dim'] + model_params['nr_dim'],
                        og_features=model_params['r_dim'])
    model = DisentangleModel(robust_encoder=r_encoder, nr_encoder=nr_encoder, detector=detector, combiner=combiner)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data_loader
    parser.add_argument('--root', default='./data', type=str)
    parser.add_argument('--robust_path', default='./data/robust_features_train.pkl', type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--shuffle', default=True, type=bool)
    parser.add_argument('--workers', default=10, type=int)
    # model
    parser.add_argument('--model', '-m', default='wrn-28-10', type=str, help='Name of the model')
    parser.add_argument('--nr_arch', default='wrn-28-10', type=str)
    parser.add_argument('--trial',default='disen_proto', type=str)
    parser.add_argument('--model_path', type=str, default='rst_adv/checkpoint-epoch200.pt')
    # optimizer
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float)
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='Learning rate')
    parser.add_argument('--lr_schedule', type=str, default='cosine',
                        choices=('trades', 'trades_fixed', 'cosine', 'wrn'),
                        help='Learning rate schedule')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='Use extragrdient steps')
    # log
    parser.add_argument('--out_root', default='./output', type=str)
    parser.add_argument('--save_freq', default=10, type=int)
    # cuda
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Disables CUDA training')
    parser.add_argument('--parallel', action='store_true')
    # freeze
    parser.add_argument('--nr_first', default=0, type=int,
                        help='only train nr until i-th epoch')
    parser.add_argument('--acc_first',default=0, type=int,
                        help='ignore nr constrint until i-th epoch')
    # loss
    parser.add_argument('--margin', default=0.9, type=float)



    args = parser.parse_args()
    # log and save
    out_dir = os.path.join(args.out_root, args.trial)
    ckpt_dir, summary_dir = create_trial_out_dir(out_dir)
    writer = SummaryWriter(log_dir=summary_dir)
    # cuda_settings
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    dl_kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    # data settings
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    dataset_params = {'root': args.root, 'transform': transform, 'robust_path':args.robust_path}
    # TODO: FIX COLLOATE FN
    loader_params = {'batch_size': args.batch_size, 'shuffle': args.shuffle,
                     'workers': args.workers, 'collate_fn':collate_fn}
    train_loader = get_loader(dataset_params=dataset_params, loader_params=loader_params)
    # model
    similarity = nn.CosineSimilarity()
    loss_params = {'margin': args.margin, 'similarity': similarity, 'alpha': 1}
    attack_params = {'eps': 0.031, 'step_size': 0.01, 'num_steps': 10}
    # TODO: Fix Parallel
    r_encoder = prepare_model(args.model_path, args.model)
    for p in r_encoder.parameters():
        p.requires_grad = False
    # TODO: Fix Load Model
    nr_encoder = get_model(args.nr_arch)
    detector = Detector(in_features=640 + 640)
    combiner = Combiner(in_features=640 + 640, og_features=640)
    model = DisentangleModel(robust_encoder=r_encoder, nr_encoder=nr_encoder, detector=detector, combiner=combiner,
                             attack_params=attack_params, loss_params=loss_params)
    # TODO: FIX OBJECT HAS NO ATTRIBUTE
    if args.parallel:
        device_ids = [i for i in range(torch.cuda.device_count())]
        model = torch.nn.DataParallel(model).cuda()

        cudnn.benchmark = True
        print(model.device_ids)
        #print(device_ids)
    #model = model.cuda()
    # optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay,
                          nesterov=args.nesterov)
    # TODO: FILL
    lr_params = {'lr':args.lr, 'lr_schedule':args.lr_schedule, 'epochs':args.epochs,
                 'nr_first':args.nr_first, 'acc_first':args.acc_first, 'parallel':args.parallel}

    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        lr = adjust_learning_rate(optimizer, epoch, lr_params)
        # adversarial training
        train(model, train_loader, device, loss_params, optimizer, attack_params, epoch, writer)
        if epoch % args.save_freq == 0 or epoch == args.epochs:
            torch.save(model.state_dict(),
                       os.path.join(ckpt_dir, 'ckpt_epoch_%d.pt'%epoch))

