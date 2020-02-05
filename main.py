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

from dataset import *
import model
from utils import *

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
parser.add_argument('--sample_rate', default=2, type=int)
args = parser.parse_args()

learning_rate = 1e-4
epochs = 50
loss_layer = nn.CrossEntropyLoss()

num_stages = 2
num_layers = 12
num_f_maps = 64
dim = 2048
sample_rate = args.sample_rate
num_classes = len(phase2label_dicts[args.dataset])

def train(model, train_loader, validation_loader):
    global learning_rate, epochs
    mse_layer = nn.MSELoss(reduction='none')

    model.to(device)
    save_dir = 'models/tcn/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for epoch in range(1, epochs + 1):
        if epoch % 30 == 0:
            learning_rate = learning_rate * 0.5
        model.train()
        correct = 0
        total = 0
        loss_item = 0
        optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=1e-5)
        for (video, labels, mask, video_name) in tqdm(train_loader):
            labels = torch.Tensor(labels).long()
            mask = torch.Tensor(mask).long()
            mask = mask.to(device)
            video, labels = video.to(device), labels.to(device)
            outputs = model(video, None)

            loss = 0
            for stage, p in enumerate(outputs):
                loss += loss_layer(p.transpose(2, 1).contiguous().view(-1, num_classes), labels.view(-1))
                loss += 0.15 * torch.mean(torch.clamp(mse_layer(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16))

            loss_item += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs[-1].data, 1)
            correct += ((predicted == labels).sum()).item()
            total += labels.shape[0]

        print('Train Epoch {}: Acc {}, Loss {}'.format(epoch, correct / total, loss_item / total))
        test(model, validation_loader)
        torch.save(model.state_dict(), save_dir + "/{}.model".format(epoch))

def test(model, test_loader):
    model.to(device)
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for (video, labels, mask, video_name) in tqdm(test_loader):
            labels = torch.Tensor(labels).long()
            mask = torch.Tensor(mask).float()

            mask = mask.to(device)
            video, labels = video.to(device), labels.to(device)
            outputs = model(video, mask)
            _, predicted = torch.max(outputs[-1].data, 1)
            correct += ((predicted == labels).sum()).item()
            total += labels.shape[0]

        print('Test: Acc {}'.format(correct / total))


def predict(model, test_loader, argdataset, sample_rate, results_dir):
    model.eval()
    model.to(device)

    with torch.no_grad():
        for (video, labels, mask, video_name) in tqdm(test_loader):
            # labels = torch.Tensor(labels).long()
            mask = torch.Tensor(mask).float()
            mask = mask.to(device)
            video = video.to(device)
            re = model(video, mask)
            confidence, predicted = torch.max(F.softmax(re[-1].data,1), 1)
            predicted = predicted.squeeze(0).tolist()
            labels = [label.item() for label in labels]
            
            pic_file = video_name[0].split('.')[0] + '-vis.png'
            pic_path = os.path.join(results_dir, pic_file)
            mask = [m.item() for m in mask]
            confidence = confidence.squeeze(0).tolist()
            segment_bars(pic_path, labels, predicted, mask)
#             segment_bars_with_confidence_score(pic_path, confidence_score=confidence, labels=[labels, predicted, mask])
            
#             predicted_phases_txt = label2phase(predicted, phase2label_dict=phase2label_dicts[argdataset])
#             predicted_phases_expand = []
#             for i in predicted_phases_txt:
#                 predicted_phases_expand = np.concatenate((predicted_phases_expand, [i] * 5 * sample_rate)) # 5 is the resampled resolution
# 
# 
#             target_video_file = video_name[0].split('.')[0] + '_pred.txt'
#             gt_file = video_name[0].split('.')[0] + '-phase.txt'
# 
#             g_ptr = open(os.path.join(results_dir, gt_file), "r")
#             f_ptr = open(os.path.join(results_dir, target_video_file), 'w')
# 
#             gt = g_ptr.readlines()[1:] ##
#             predicted_phases_expand = predicted_phases_expand[0:len(gt)]
#             assert len(predicted_phases_expand) == len(gt)
# 
#             f_ptr.write("Frame\tPhase\n")
#             for index, line in enumerate(predicted_phases_expand):
#                 f_ptr.write('{}\t{}\n'.format(index, line))
#             f_ptr.close()

causal_tcn = model.MultiStageCausalTCN(num_stages, num_layers, num_f_maps, dim, num_classes)

if args.action == 'train':
    video_traindataset = VideoDataset('cholec80', 'cholec80/train_dataset', sample_rate)
    video_train_dataloader = DataLoader(video_traindataset, batch_size=1, shuffle=True, drop_last=False)
    video_testdataset = VideoDataset('cholec80', 'cholec80/test_dataset', sample_rate)
    video_test_dataloader = DataLoader(video_testdataset, batch_size=1, shuffle=False, drop_last=False)
    train(causal_tcn, video_train_dataloader, video_test_dataloader)

if args.action == 'test':
    model_path = 'models/tcn/50.model'
    causal_tcn.load_state_dict(torch.load(model_path))
    video_testdataset = VideoDataset('cholec80', 'cholec80/test_dataset', sample_rate)
    video_test_dataloader = DataLoader(video_testdataset, batch_size=1, shuffle=False, drop_last=False)
    test(causal_tcn, video_test_dataloader)

if args.action == 'predict':
    model_path = 'models/tcn/50.model'
    causal_tcn.load_state_dict(torch.load(model_path))
    video_testdataset =VideoDataset('cholec80', 'cholec80/test_dataset', sample_rate)
    video_test_dataloader = DataLoader(video_testdataset, batch_size=1, shuffle=False, drop_last=False)
    results_dir = os.path.join(args.dataset, 'eva/test_dataset')
    predict(causal_tcn,video_test_dataloader,args.dataset, sample_rate, results_dir)
    
if args.action == 'cross_validate':
    
