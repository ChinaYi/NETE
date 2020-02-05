import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import os
import argparse
import numpy as np
import random
from tqdm import tqdm

import dataset
from model import inception_v3

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
parser.add_argument('--action', choices=['train', 'extract', 'test', 'validate'], default='train')
parser.add_argument('--dataset', default="cholec80")
args = parser.parse_args()

learning_rate = 1e-4
epochs = 10
loss_layer = nn.CrossEntropyLoss()


def train(model, train_loader, validation_loader):
    global learning_rate
    save_dir = 'models/framewise_feature/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model.to(device)
    for epoch in range(1, epochs + 1):
        if epoch % 4 == 0:
            learning_rate = learning_rate * 0.5
        model.train()

        correct = 0
        total = 0
        loss_item = 0
        optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=1e-5)

        for (imgs, labels, img_names) in tqdm(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            feature, res = model(imgs) # of shape 64 x 7
            loss = loss_layer(res, labels)
            loss_item += loss.item()
            _, prediction = torch.max(res.data, 1)
            correct += ((prediction == labels).sum()).item()
            total += len(prediction)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Train Epoch {}: Acc {}, Loss {}'.format(epoch, correct/total, loss_item/total))
        test(model, validation_loader)
        torch.save(model.state_dict(), save_dir + "/{}.model".format(epoch))

    print('Training done!')


def test(model, test_loader):
    print('Testing...')
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    loss_item = 0
    with torch.no_grad():
        for (imgs, labels, img_names) in tqdm(test_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            feature, res = model(imgs)  # of shape 64 x 7
            loss = loss_layer(res, labels)
            loss_item += loss.item()
            _, prediction = torch.max(res.data, 1)
            correct += ((prediction == labels).sum()).item()
            total += len(prediction)
    print('Test: Acc {}, Loss {}'.format(correct / total, loss_item / total))



def extract(model, loader, save_path):
    model.eval()
    model.to(device)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with torch.no_grad():
        for (imgs, labels, img_names) in tqdm(loader):
            assert len(img_names) == 1 # batch_size = 1
            video, img_in_video = img_names[0].split('/')[-2], img_names[0].split('/')[-1] # video63 5730.jpg
            video_folder = os.path.join(save_path, video)
            if not os.path.exists(video_folder):
                os.makedirs(video_folder)
            feature_save_path = os.path.join(video_folder, img_in_video.split('.')[0] + '.npy')
            if os.path.exists(feature_save_path):
                continue

            imgs, labels = imgs.to(device), labels.to(device)
            features, res = model(imgs)
            features = features.to('cpu').numpy() # of shape 1 x 2048

            np.save(feature_save_path, features)




def imgf2videof(source_folder, target_folder):
    '''
        Merge the extracted img feature to video feature.
    '''
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    for video in os.listdir(source_folder):
        video_abs_path = os.path.join(source_folder, video)
        nums_of_imgs = len(os.listdir(video_abs_path))
        video_feature = []
        for i in range(nums_of_imgs):
            img_abs_path = os.path.join(video_abs_path, str(i) + '.npy')
            video_feature.append(np.load(img_abs_path))

        video_feature = np.concatenate(video_feature, axis=0)
        video_feature_save_path = os.path.join(target_folder, video + '.npy')
        np.save(video_feature_save_path, video_feature)
        print('{} done!'.format(video))


inception = inception_v3(pretrained=True, aux_logits=False)
fc_features = inception.fc.in_features
inception.fc = nn.Linear(fc_features, len(dataset.phase2label_dicts[args.dataset]))

if args.action == 'train':
    framewise_traindataset = dataset.FramewiseDataset('cholec80', 'cholec80/train_dataset')
    framewise_train_dataloader = DataLoader(framewise_traindataset, batch_size=64, shuffle=True, drop_last=False)

    framewise_testdataset = dataset.FramewiseDataset('cholec80', 'cholec80/test_dataset')
    framewise_test_dataloader = DataLoader(framewise_testdataset, batch_size=64, shuffle=True, drop_last=False)

    train(inception, framewise_train_dataloader, framewise_test_dataloader)

if args.action == 'test':
    model_path = 'models/framewise_feature/2.model'
    inception.load_state_dict(torch.load(model_path))
    framewise_testdataset = dataset.FramewiseDataset('cholec80', 'cholec80/test_dataset')
    framewise_test_dataloader = DataLoader(framewise_testdataset, batch_size=64, shuffle=True, drop_last=False)
    test(inception, framewise_test_dataloader)

if args.action == 'extract':
    model_path = 'models/framewise_feature/2.model'
    inception.load_state_dict(torch.load(model_path))
    framewise_testdataset = dataset.FramewiseDataset('cholec80', 'cholec80/train_dataset')
    framewise_test_dataloader = DataLoader(framewise_testdataset, batch_size=1, shuffle=False, drop_last=False)

    extract(inception, framewise_test_dataloader, 'cholec80/train_dataset/feature_folder/')
    imgf2videof('cholec80/train_dataset/feature_folder/', 'cholec80/train_dataset/video_feature_folder/')



