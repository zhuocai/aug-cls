# external modules
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# torch related module
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import TensorDataset, DataLoader

import torchvision

import torchvision.transforms as transforms

# local folder
import resnet
import transform_layers

parser = argparse.ArgumentParser(description='PyTorch Image Classification')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--epochs', default=5, type=int, help='epochs')
parser.add_argument('--tmax', default=5, type=int, help='tmax')
parser.add_argument('--seed', default=2020, type=int, help='random seed')
parser.add_argument('--task', default=1, type=int, help='1 | 2 which task/label')
parser.add_argument('--transform', default=1, type=int, help='which transformation to apply')
parser.add_argument('--root_dir', default='../../data/q1_data', type=str,
                    help='path to data folder')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

batch_size = 128
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# data
root_dir = args.root_dir
train_x = np.load(osp.join(root_dir, 'train.npy'))
test_x = np.load(osp.join(root_dir, 'test.npy'))
train_num = train_x.shape[0]
test_num = test_x.shape[0]
train_x = train_x.reshape(train_num, 3, 32, 32) / 255.0
test_x = test_x.reshape(test_num, 3, 32, 32) / 255.0

train_label = pd.read_csv(osp.join(root_dir, 'train%d.csv' % args.task)).to_numpy()[:, 1].astype(int).ravel()
num_classes = 20 if args.task == 1 else 100
label_name = 'coarse' if args.task == 1 else 'fine'
log_name = label_name + '_%d' % args.transformation

mean = np.array([0.4914, 0.4822, 0.4465])
std = np.array([0.2023, 0.1994, 0.2010])

train_x = (train_x - np.einsum('ikl,j->ijkl', np.ones((train_x.shape[0], 32, 32)), mean)
           ) / np.einsum('ikl,j->ijkl', np.ones((train_x.shape[0], 32, 32)), std)

test_x = (test_x - np.einsum('ikl,j->ijkl', np.ones((test_x.shape[0], 32, 32)), mean)
          ) / np.einsum('ikl,j->ijkl', np.ones((test_x.shape[0], 32, 32)), std)

## train-val split
train_x, val_x, train_y, val_y = train_test_split(
    train_x, train_label, test_size=0.2, random_state=args.seed)

# leave data augmentations to later phase

# to Tensor
train_x = torch.Tensor(train_x)
val_x = torch.Tensor(val_x)
test_x = torch.Tensor(test_x)
train_y = torch.LongTensor(train_y)
val_y = torch.LongTensor(val_y)

trainset = TensorDataset(train_x, train_y)
valset = TensorDataset(val_x, val_y)
testset = TensorDataset(test_x)

train_loader = DataLoader(trainset, shuffle=True,
                          batch_size=128, num_workers=2, pin_memory=True)
val_loader = DataLoader(valset, shuffle=False,
                        batch_size=batch_size, num_workers=2, pin_memory=True)
test_loader = DataLoader(testset, shuffle=False,
                         batch_size=batch_size, num_workers=2, pin_memory=True)

model = resnet.ResNet18(num_channels=3, num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.tmax)

train_accs = []
val_accs = []


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    print('learning rate=%.4f' % optimizer.state_dict()['param_groups'][0]['lr'])
    train_bar = tqdm(enumerate(train_loader))
    for batch_idx, (inputs, targets) in train_bar:
        # print('inputs_size', inputs.size())
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = transform_layers.transform(inputs, args.transform)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        train_bar.set_postfix_str('Loss: %.4f | Acc: %.4f (%d/%d)'
                                  % (train_loss / (batch_idx + 1), correct / total, correct, total))
    train_accs.append(100. * correct / total)


def val(epoch):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        val_bar = tqdm(enumerate(val_loader))
        for batch_idx, (inputs, targets) in val_bar:
            # print('inputs_size', inputs.size())
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        val_bar.set_postfix_str('Loss: %.4f | Acc: %.4f (%d/%d)'
                                % (val_loss / (batch_idx + 1), correct / total, correct, total))
    val_accs.append(100. * correct / total)


def gen_res():
    model.eval()
    preds = []
    with torch.no_grad():
        test_bar = tqdm(enumerate(test_loader))
        for i, (inputs,) in test_bar:
            inputs = inputs.to(device)
            outputs = model(inputs)
            pred = outputs.argmax(dim=1)
            preds.append(pred.cpu().numpy())
    preds = np.concatenate(preds).ravel()
    df = pd.DataFrame({'image_id': np.arange(test_num),
                       label_name + '_label': preds})
    df.to_csv(log_name + '.csv', index=False)


for epoch in range(args.epochs):
    train(epoch)
    val(epoch)
    scheduler.step()

torch.save(model.state_dict(), log_name + '.pt')

with open(log_name + '.txt', 'w') as f:
    for i in range(args.epochs):
        f.write('train acc = %.4f, val acc = %.4f\n' % (train_accs[i], val_accs[i]))

gen_res()
