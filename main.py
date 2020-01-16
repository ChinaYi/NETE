import torch
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import os
import argparse
import numpy as np
import random
from tqdm import tqdm

import dataset
import model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 19980125
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--action', choices=['train', 'predict', 'test'], default='train')
parser.add_argument('--dataset', default="cholec80")
args = parser.parse_args()

learning_rate = 1e-4
epochs = 100
loss_layer = nn.CrossEntropyLoss()

num_stages = 4
num_layers = 10
num_f_maps = 64
dim = 2048
num_classes = len(dataset.phase2label_dicts[args.dataset])

def train(model, train_loader, validation_loader):
    global learning_rate, epochs
    model.to(device)
    save_dir = 'models/tcn/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for epoch in range(1, epochs + 1):
        if epoch % 50 == 0:
            learning_rate = learning_rate * 0.5
        model.train()
        correct = 0
        total = 0
        loss_item = 0
        optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=1e-5)
        for (video, labels, video_name) in tqdm(train_loader):
            video, labels = video.to(device), labels.to(device)
            outputs = model(video)

            loss = 0
            for p in outputs:
                loss += loss_layer(p.transpose(2, 1).contiguous().view(-1, num_classes), labels.view(-1))

            loss_item += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs[-1].data, 1)
            correct += ((predicted == labels).sum()).item()
            total += len(labels.shape[1])

        print('Train Epoch {}: Acc {}, Loss {}'.format(epoch, correct / total, loss_item / total))
        test(model, validation_loader)
        torch.save(model.state_dict(), save_dir + "/{}.model".format(epoch))

def test(model, test_loader):
    model.to(device)
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for (video, labels, video_name) in tqdm(test_loader):

            video, labels = video.to(device), labels.to(device)
            outputs = model(video)
            _, predicted = torch.max(outputs[-1].data, 1)
            correct += ((predicted == labels).sum()).item()
            total += len(labels.shape[1])

        print('Test: Acc {}'.format(correct / total))



causal_tcn = model.MultiStageCausalTCN(num_stages, num_layers, num_f_maps, dim, num_classes)

if args.action == 'train':
    video_traindataset = dataset.VideoDataset('cholec80', 'cholec80/train_dataset')
    video_train_dataloader = DataLoader(video_traindataset, batch_size=1, shuffle=True, drop_last=False)
    video_testdataset = dataset.VideoDataset('cholec80', 'cholec80/test_dataset')
    video_test_dataloader = DataLoader(video_testdataset, batch_size=1, shuffle=False, drop_last=False)
    train(causal_tcn, video_train_dataloader, video_test_dataloader)

if args.action == 'test':
    model_path = 'models/tcn/100.model'
    causal_tcn.load_state_dict(torch.load(model_path))
    video_testdataset = dataset.VideoDataset('cholec80', 'cholec80/test_dataset')
    video_test_dataloader = DataLoader(video_testdataset, batch_size=1, shuffle=False, drop_last=False)
    test(causal_tcn, video_test_dataloader)

if args.action == 'predict':
    pass