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
from sklearn.model_selection  import KFold

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
parser.add_argument('--action', default='base_train')
parser.add_argument('--dataset', default="cholec80")
parser.add_argument('--sample_rate', default=2, type=int)
args = parser.parse_args()

learning_rate = 1e-4
epochs = 200
loss_layer = nn.CrossEntropyLoss()
mse_layer = nn.MSELoss(reduction='none')


# num_stages = 2
num_layers = 12 #12
num_f_maps = 64
dim = 2048
sample_rate = args.sample_rate
num_classes = len(phase2label_dicts[args.dataset])

def step_refine_model(base_model, refine_model, video, labels, mask, learning_rate, optimizer):
    ## train refine model
    # let try more zeros
    # let try random noise
#     video = video.permute(0,2,1)
#     video = video * mask
#     random_noise = np.random.rand(1,2048,1)
#     random_noise = np.tile(random_noise, (1,1,len(mask)))
# #     random_noise = np.random.rand(1,2048, len(mask))
#     random_noise = torch.from_numpy(random_noise).float().to(device)
#     random_noise = random_noise * ((mask==0).float())
#     video = video + random_noise
#     video = video.permute(0,2,1)
    
#     additonal_mask = np.random.choice(2, len(mask), replace=True, p=[0.2,0.8])
#     additonal_mask = torch.from_numpy(additonal_mask).float().to(device)
#     
#     mask = additonal_mask * mask
#     import pdb
#     pdb.set_trace()
    base_model.eval()
    outputs = base_model(video, mask)
    
    _, predicted = torch.max(outputs.data, 1)
    ori_correct = ((predicted == labels).sum()).item()
    
    
    outputs_detach = F.softmax(outputs, dim=1).detach()
    outputs = refine_model(outputs_detach)
    

    loss = 0
    loss += loss_layer(outputs.transpose(2, 1).contiguous().view(-1, num_classes), labels.view(-1))
    loss += torch.mean(torch.clamp(mse_layer(F.log_softmax(outputs[:, :, 1:], dim=1), F.log_softmax(outputs.detach()[:, :, :-1], dim=1)), min=0, max=16))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    _, predicted = torch.max(outputs.data, 1)
    correct = ((predicted == labels).sum()).item()
    
    return loss.item(), ori_correct, correct
    

def refine_train(base_model, refine_model, train_loader, validation_loader):
    global learning_rate, epochs
    mse_layer = nn.MSELoss(reduction='none')
    
    base_model.to(device)
    refine_model.to(device)
    base_save_dir = 'models/base_tcn/'
    refine_save_dir = 'models/refine_tcn/'
    if not os.path.exists(base_save_dir):
        os.makedirs(base_save_dir)
    if not os.path.exists(refine_save_dir):
        os.makedirs(refine_save_dir)
    for epoch in range(1, epochs + 1):
        if epoch % 30 == 0:
            learning_rate = learning_rate * 0.5
        base_model.train()
        refine_model.train()
        correct = 0
        correct2 = 0
        correct_ori = 0
        total = 0
        loss_item = 0
        loss_item2 = 0
        optimizer = torch.optim.Adam(base_model.parameters(), learning_rate, weight_decay=1e-4)
        optimizer2 = torch.optim.Adam(refine_model.parameters(), learning_rate, weight_decay=1e-5)
        for (video, labels, mask, video_name) in (train_loader):
            labels = torch.Tensor(labels).long()
            mask = torch.Tensor(mask).float()
            video, labels = video.to(device), labels.to(device)
            mask = mask.to(device)
            
            outputs = base_model(video)
            
            loss = 0
            loss += loss_layer(outputs.transpose(2, 1).contiguous().view(-1, num_classes), labels.view(-1))
            loss += torch.mean(torch.clamp(mse_layer(F.log_softmax(outputs[:, :, 1:], dim=1), F.log_softmax(outputs.detach()[:, :, :-1], dim=1)), min=0, max=16))
            loss_item += loss.item()
            
            if epoch <= 100:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            correct += ((predicted == labels).sum()).item()
            total += labels.shape[0]
            
            tmp_loss, tmp_ori_correct, tmp_correct = step_refine_model(base_model, refine_model, video, labels, mask, learning_rate, optimizer2)
            correct_ori += tmp_ori_correct
            correct2 += tmp_correct
            loss_item2 += tmp_loss
            
            

        print('Train Epoch {}: Acc {}, Acc2 {}-{}, Loss {}, Loss2 {}'.
              format(epoch, correct / total, correct_ori/total, correct2 / total, loss_item / total, loss_item2 / total))
        refine_test(base_model, refine_model, validation_loader)
        torch.save(base_model.state_dict(), base_save_dir + '/{}.model'.format(epoch))
        torch.save(refine_model.state_dict(), refine_save_dir + '/{}.model'.format(epoch))


def refine_test(base_model, refine_model, test_loader):
    base_model.to(device)
    refine_model.to(device)
    base_model.eval()
    refine_model.eval()
    with torch.no_grad():
        correct1 = 0
        correct2 = 0
        total = 0
        for (video, labels, mask, video_name) in (test_loader):
            labels = torch.Tensor(labels).long()
            mask = torch.Tensor(mask).float()
            video, labels = video.to(device), labels.to(device)
            mask = mask.to(device)
#             outputs1 = base_model(video)
            outputs1 = base_model(video, mask)
            outputs1 = F.softmax(outputs1, dim=1)
            outputs2 = refine_model(outputs1)
            _, predicted1 = torch.max(outputs1.data, 1)
            _, predicted2 = torch.max(outputs2.data, 1)

            correct1 += ((predicted1 == labels).sum()).item()
            correct2 += ((predicted2== labels).sum()).item()
            total += labels.shape[0]

        print('Test: Acc1 {}, Acc2 {}'.format(correct1 / total, correct2/total))
    

def refine_predict(base_model, refine_model, test_loader):
    base_model.eval()
    refine_model.eval
    base_model.to(device)
    refine_model.to(device)
    

    with torch.no_grad():
        for (video, labels, mask, video_name) in tqdm(test_loader):
            video = video.to(device)
            mask = torch.Tensor(mask).float()
            mask = mask.to(device)
            
            
#             video = video.permute(0,2,1)
#             video = video * mask
#             random_noise = np.random.rand(1,2048,1)
#             random_noise = np.tile(random_noise, (1,1,len(mask)))
#         #     random_noise = np.random.rand(1,2048, len(mask))
#             random_noise = torch.from_numpy(random_noise).float().to(device)
#             random_noise = random_noise * ((mask==0).float())
#             video = video + random_noise
#             video = video.permute(0,2,1)
            
            re = refine_model(F.softmax(base_model(video, mask), dim=1))
#             re = base_model(video, mask)
            confidence, predicted = torch.max(F.softmax(re.data,1), 1)
            predicted = predicted.squeeze(0).tolist()
            labels = [label.item() for label in labels]
            
            pic_file = video_name[0].split('.')[0] + '-vis.png'
            pic_path = os.path.join(results_dir, pic_file)
            mask = [m.item() for m in mask]
            confidence = confidence.squeeze(0).tolist()
#             segment_bars(pic_path, labels, predicted, mask)
            segment_bars_with_confidence_score(pic_path, confidence_score=confidence, labels=[labels, predicted, mask])
    
    
    
def base_train(model, train_loader, validation_loader):
    global learning_rate, epochs
#     partial_entropy = nn.CrossEntropyLoss(reduction='none')
    model.to(device)
    save_dir = 'models/base_tcn/'
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
        for (video, labels, mask, video_name) in (train_loader):
            labels = torch.Tensor(labels).long()
            mask = torch.Tensor(mask).float()
            video, labels = video.to(device), labels.to(device)
            mask = mask.to(device)
#             outputs = model(video, mask)
            outputs = model(video)
            
            loss = 0
            loss += loss_layer(outputs.transpose(2, 1).contiguous().view(-1, num_classes), labels.view(-1))
            loss += torch.mean(torch.clamp(mse_layer(F.log_softmax(outputs[:, :, 1:], dim=1), F.log_softmax(outputs.detach()[:, :, :-1], dim=1)), min=0, max=16))
#             for stage, p in enumerate(outputs):
#                 if stage == 0: # The first stage
# #                     tmp = partial_entropy(p.transpose(2, 1).contiguous().view(-1, num_classes), labels.view(-1))
# #                     loss += torch.mean(tmp * mask)
#                     loss += loss_layer(p.transpose(2, 1).contiguous().view(-1, num_classes), labels.view(-1))
#                     loss += torch.mean(torch.clamp(mse_layer(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16))
# 
# #                 else:
# #                     loss += loss_layer(p.transpose(2, 1).contiguous().view(-1, num_classes), labels.view(-1))
# #                     loss += torch.mean(torch.clamp(mse_layer(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16))

            loss_item += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            correct += ((predicted == labels).sum()).item()
            total += labels.shape[0]

        print('Train Epoch {}: Acc {}, Loss {}'.format(epoch, correct / total, loss_item / total))
        base_test(model, validation_loader)
        torch.save(model.state_dict(), save_dir + '/{}.model'.format(epoch))


def base_test(model, test_loader):
    model.to(device)
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for (video, labels, mask, video_name) in (test_loader):
            labels = torch.Tensor(labels).long()
            mask = torch.Tensor(mask).float()
            video, labels = video.to(device), labels.to(device)
            mask = mask.to(device)
#             outputs = model(video, mask)
            outputs = model(video)
            _, predicted = torch.max(outputs.data, 1)
            correct += ((predicted == labels).sum()).item()
            total += labels.shape[0]

        print('Test: Acc {}'.format(correct / total))


def base_predict(model, test_loader, argdataset, sample_rate, results_dir):
    model.eval()
    model.to(device)

    with torch.no_grad():
        for (video, labels, mask, video_name) in tqdm(test_loader):
#             zero_one_mask = (mask!=7).to(device).float()    
            mask = torch.Tensor(mask).float()
            video = video.to(device)
            mask = mask.to(device)
            video = video.permute(0,2,1)
            video = video * mask
            random_noise = np.random.rand(1,2048,1)
            random_noise = np.tile(random_noise, (1,1,len(mask)))
        #     random_noise = np.random.rand(1,2048, len(mask))
            random_noise = torch.from_numpy(random_noise).float().to(device)
            random_noise = random_noise * ((mask==0).float())
            video = video + random_noise
            video = video.permute(0,2,1)
#             video = video.permute(0,2,1)
#             video = video * mask
#             
#             random_noise = np.random.rand(1, 2048, len(mask))
#             random_noise = torch.from_numpy(random_noise).float().to(device)
#             random_noise = random_noise * ((mask==0).float())
#             video = video + random_noise
#             video = video.permute(0,2,1)
            re = model(video)
            
#             re = model(video, mask)
            confidence, predicted = torch.max(F.softmax(re.data,1), 1)
            predicted = predicted.squeeze(0).tolist()
            labels = [label.item() for label in labels]
            
            pic_file = video_name[0].split('.')[0] + '-vis.png'
            pic_path = os.path.join(results_dir, pic_file)
            mask = [m.item() for m in mask]
            confidence = confidence.squeeze(0).tolist()
#             segment_bars(pic_path, labels, predicted, mask)
            segment_bars_with_confidence_score(pic_path, confidence_score=confidence, labels=[labels, predicted, mask])
            
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

base_model = model.BaseCausalTCN(num_layers, num_f_maps, dim, num_classes)
refine_model = model.RefineCausualTCN(4, num_f_maps, num_classes, num_classes)
# causal_tcn = model.MultiStageCausalTCN(num_stages, num_layers, num_f_maps, dim, num_classes)

if args.action == 'base_train':
    #
#     kf = KFold(10, shuffle=True, random_state=seed) # 10-fold cross validate
#     video_list = ['video{0:>2d}'.format(i) for i in range(41,81)]
#     for train_idx, test_idx in kf.split(video_list):
#         trainlist = ['video{:0>2d}'.format(i+40) for i in train_idx]
#         testlist = ['video{:0>2d}'.format(i+40) for i in test_idx]
#         
#         video_traindataset = VideoDataset('cholec80', 'cholec80/test_dataset', sample_rate, load_hard_frames=True, blacklist=testlist)
#         video_train_dataloader = DataLoader(video_traindataset, batch_size=1, shuffle=True, drop_last=False)
#         video_testdataset = VideoDataset('cholec80', 'cholec80/test_dataset', sample_rate, load_hard_frames=True, blacklist=trainlist)
#         video_test_dataloader = DataLoader(video_testdataset, batch_size=1, shuffle=False, drop_last=False)
#         base_train(base_model, video_train_dataloader, video_test_dataloader)
    
    video_traindataset = VideoDataset('cholec80', 'cholec80/train_dataset', sample_rate, load_hard_frames=True)
    video_train_dataloader = DataLoader(video_traindataset, batch_size=1, shuffle=True, drop_last=False)
    video_testdataset = VideoDataset('cholec80', 'cholec80/test_dataset', sample_rate, load_hard_frames=True)
    video_test_dataloader = DataLoader(video_testdataset, batch_size=1, shuffle=False, drop_last=False)
    base_train(base_model, video_train_dataloader, video_test_dataloader)

if args.action == 'base_test':
    model_path = 'models/base_tcn/100.model'
    base_model.load_state_dict(torch.load(model_path))
    video_testdataset = VideoDataset('cholec80', 'cholec80/test_dataset', sample_rate, load_hard_frames=True)
    video_test_dataloader = DataLoader(video_testdataset, batch_size=1, shuffle=False, drop_last=False)
    base_test(base_model, video_test_dataloader)

if args.action == 'base_predict':
    model_path = 'models/base_tcn/50.model'
    base_model.load_state_dict(torch.load(model_path))
    video_testdataset =VideoDataset('cholec80', 'cholec80/train_dataset', sample_rate, load_hard_frames=True)
    video_test_dataloader = DataLoader(video_testdataset, batch_size=1, shuffle=False, drop_last=False)
    results_dir = os.path.join(args.dataset, 'eva/test_dataset')
    base_predict(base_model,video_test_dataloader,args.dataset, sample_rate, results_dir)
    

if args.action == 'refine_train':
    video_traindataset = VideoDataset('cholec80', 'cholec80/train_dataset', sample_rate, load_hard_frames=True)
    video_train_dataloader = DataLoader(video_traindataset, batch_size=1, shuffle=True, drop_last=False)
    video_testdataset = VideoDataset('cholec80', 'cholec80/test_dataset', sample_rate, load_hard_frames=True)
    video_test_dataloader = DataLoader(video_testdataset, batch_size=1, shuffle=False, drop_last=False)
    refine_train(base_model, refine_model, video_train_dataloader, video_test_dataloader)
    

if args.action == 'refine_predict':
    base_model_path = 'models/base_tcn/100.model'
    refine_model_path = 'models/refine_tcn/100.model'
    base_model.load_state_dict(torch.load(base_model_path))
    refine_model.load_state_dict(torch.load(refine_model_path))
    video_testdataset =VideoDataset('cholec80', 'cholec80/train_dataset', sample_rate, load_hard_frames=True)
    video_test_dataloader = DataLoader(video_testdataset, batch_size=1, shuffle=False, drop_last=False)
    results_dir = os.path.join(args.dataset, 'eva/test_dataset')
    refine_predict(base_model, refine_model, video_test_dataloader)
    
if args.action == 'cross_validate':
    # useless
    kf = KFold(10, shuffle=True, random_state=seed) # 10-fold cross validate
    video_list = ['video{0:>2d}'.format(i) for i in range(1,41)]
    for train_idx, test_idx in kf.split(video_list):
        single_stage_tcn = model.MultiStageCausalTCN(1, num_layers, num_f_maps, dim, num_classes) # single stage
        trainlist = ['video{:0>2d}'.format(i+1) for i in train_idx]
        testlist = ['video{:0>2d}'.format(i+1) for i in test_idx]
        
        video_traindataset = VideoDataset('cholec80', 'cholec80/train_dataset', sample_rate, blacklist=testlist)
        video_testdataset = VideoDataset('cholec80', 'cholec80/train_dataset', sample_rate, blacklist=trainlist)
        cross_validate(single_stage_tcn, video_traindataset, video_testdataset, 'cholec80/train_dataset/hard_frames@{}'.format(sample_rate), 7)
#     if args.dataset == 'cholec80':
#         trainlist = ['video{0:>2d}'.format(i) for i in range(1,41)]
    
    
    
