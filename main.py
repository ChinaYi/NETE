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
parser.add_argument('--k', default=-100, type=int)
args = parser.parse_args()

learning_rate = 1e-4
epochs = 100
loss_layer = nn.CrossEntropyLoss()
mse_layer = nn.MSELoss(reduction='none')



num_stages = 4  # 4
num_layers = 12 # 12
num_f_maps = 64
dim = 2048
sample_rate = args.sample_rate
num_classes = len(phase2label_dicts[args.dataset])

    

def refine_train(base_model, refine_model, train_loader, validation_loader, save_dir='models/refine_model/'):
    global learning_rate
    epochs = 100
    mse_layer = nn.MSELoss(reduction='none')
    
    base_model.to(device)
    refine_model.to(device)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch in range(1, epochs + 1):
        if epoch % 30 == 0:
            learning_rate = learning_rate * 0.5
        
        refine_model.train()
        total = 0
        correct = 0
        loss_item = 0
        optimizer = torch.optim.Adam(refine_model.parameters(), learning_rate*10)
        for (video, labels, mask, video_name) in (train_loader):
            ## labels is two times longer than video
            labels = labels[::sample_rate]
            labels = torch.Tensor(labels).long()
            mask = torch.Tensor(mask).float()
            video, labels = video.to(device), labels.to(device)
            mask = mask.to(device)
            
            
            outputs, _ = refine_model(video)
            
            loss = 0
            for output in outputs:
                loss += loss_layer(output.view(-1, num_classes), labels.view(-1))
#                 loss += 0.5 * torch.mean(torch.clamp(mse_layer(F.log_softmax(output[:, 1:, :], dim=2), F.log_softmax(output.detach()[:, :-1, :], dim=2)), min=0, max=16))
            
            loss_item += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs[-1].data, 2)
            correct += ((predicted == labels).sum()).item()
            total += labels.shape[0]
            

        print('Train Epoch {}: Acc {}, Loss {}'.format(epoch, correct / total, loss_item / total))
        refine_test(base_model, refine_model, validation_loader)
        torch.save(refine_model.state_dict(), save_dir + '/{}.model'.format(epoch))


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
#             outputs1 = F.softmax(outputs1, dim=1)
            outputs1 = outputs1.permute(0,2,1)
            outputs2, _ = refine_model(outputs1)
            
            _, predicted1 = torch.max(outputs1.data, 2)
            _, predicted2 = torch.max(outputs2[-1].data, 2)

            correct1 += ((predicted1 == labels).sum()).item()
            correct2 += ((predicted2== labels).sum()).item()
            total += labels.shape[0]

        print('Test: Acc1 {}, Acc2 {}'.format(correct1 / total, correct2/total))
    

def refine_predict(base_model, refine_model, test_loader):
    base_model.eval()
    refine_model.eval()
    base_model.to(device)
    refine_model.to(device)
    
    pic_save_dir = 'results/refine_vis/'
    results_dir = 'cholec80/eva/test_dataset/'
#     results_dir = 'm2cai16/eva/test_dataset' # for m2cai16
    if not os.path.exists(pic_save_dir):
        os.makedirs(pic_save_dir)    
    with torch.no_grad():
        for (video, labels, mask, video_name) in tqdm(test_loader):
            video = video.to(device)
            mask = torch.Tensor(mask).float()
            mask = mask.to(device)
            
#             labels = labels[::sample_rate]
#             outputs, _ = refine_model(video)
            outputs, _ = refine_model(base_model(video, mask).permute(0,2,1))
            confidence, predicted = torch.max(F.softmax(outputs[-1].data,2), 2)
            predicted = predicted.squeeze(0).tolist()
            labels = [label.item() for label in labels]
            
            pic_file = video_name[0].split('.')[0] + '-vis.png'
            pic_path = os.path.join(pic_save_dir, pic_file)
            mask = [m.item() for m in mask]
            confidence = confidence.squeeze(0).tolist()
            
            
#             segment_bars(pic_path, labels, predicted, mask)
#             segment_bars_with_confidence_score(pic_path, confidence_score=confidence, labels=[labels, predicted, mask])
            
            predicted_phases_txt = label2phase(predicted, phase2label_dict=phase2label_dicts[args.dataset])
            predicted_phases_expand = []
            for i in predicted_phases_txt:
                predicted_phases_expand = np.concatenate((predicted_phases_expand, [i] * 5 * sample_rate)) # 5 is the resampled resolution
 
 
            target_video_file = video_name[0].split('.')[0] + '_pred.txt'
            gt_file = video_name[0].split('.')[0] + '-phase.txt'
#             gt_file = video_name[0].split('.')[0] + '.txt' # for m2cai16

 
            g_ptr = open(os.path.join(results_dir, gt_file), "r")
            f_ptr = open(os.path.join(results_dir, target_video_file), 'w')
 
            gt = g_ptr.readlines()[1:] ##
            predicted_phases_expand = predicted_phases_expand[0:len(gt)]
            assert len(predicted_phases_expand) == len(gt)
 
            f_ptr.write("Frame\tPhase\n")
            for index, line in enumerate(predicted_phases_expand):
                f_ptr.write('{}\t{}\n'.format(index, line))
            f_ptr.close()
    
    
def base_train(model, train_loader, validation_loader, load_hard_frames= False, save_dir = 'models/base_tcn'):
    global learning_rate, epochs
    model.to(device)
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
            outputs = model(video)
            
            loss = 0
            loss += loss_layer(outputs.transpose(2, 1).contiguous().view(-1, num_classes), labels.view(-1))
            loss += torch.mean(torch.clamp(mse_layer(F.log_softmax(outputs[:, :, 1:], dim=1), F.log_softmax(outputs.detach()[:, :, :-1], dim=1)), min=0, max=16))

            loss_item += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            correct += ((predicted == labels).sum()).item()
            total += labels.shape[0]

        print('Train Epoch {}: Acc {}, Loss {}'.format(epoch, correct / total, loss_item / total))
        base_test(model, validation_loader, load_hard_frames)
        torch.save(model.state_dict(), save_dir + '/{}.model'.format(epoch))


def base_test(model, test_loader, load_hard_frames):
    model.to(device)
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for (video, labels, mask, video_name) in (test_loader):
            labels = torch.Tensor(labels).long()
            
            # random_mask
            random_mask = np.random.choice(2, len(mask), replace=True, p=[0.3,0.7])
            random_mask = torch.from_numpy(random_mask).float().to(device)
            mask = torch.Tensor(mask).float()
            video, labels = video.to(device), labels.to(device)
            mask = mask.to(device)
            outputs = model(video, random_mask)
#             if load_hard_frames:
# #                 outputs = model(video, mask)
#                 outputs = model(video, random_mask)
#             else:
#                 outputs = model(video)
            _, predicted = torch.max(outputs.data, 1)
            correct += ((predicted == labels).sum()).item()
            total += labels.shape[0]
            
            feature_save_dir = 'cholec80/train_dataset/random_mask_video_feature@2020/'
            if not os.path.exists(feature_save_dir):
                os.makedirs(feature_save_dir)
             
            outputs_f = outputs.squeeze(0).permute(1,0).to('cpu').numpy() # of shape (l, c)
            video_name = video_name[0] # videoxx.npy
            feature_save_path = os.path.join(feature_save_dir, video_name)
            np.save(feature_save_path, outputs_f)
    
        print('Test: Acc {}'.format(correct / total))


def extract(model, data_loader, feature_save_dir, pic_save_dir):
    model.to(device)
    model.eval()
    if not os.path.exists(feature_save_dir):
        os.makedirs(feature_save_dir)
    if not os.path.exists(pic_save_dir):
        os.makedirs(pic_save_dir)
    with torch.no_grad():
        for (video, labels, mask, video_name) in (data_loader):
            labels = torch.Tensor(labels).long()
            mask = torch.Tensor(mask).float()
            video, labels = video.to(device), labels.to(device)
            mask = mask.to(device)
#             outputs = model(video, mask)
            outputs = model(video) # output of shape (bs, c, l)
            
            confidence, predicted = torch.max(F.softmax(outputs.data,1), 1)
            outputs_f = outputs.squeeze(0).permute(1,0).to('cpu').numpy() # of shape (l, c)
            
            video_name = video_name[0] # videoxx.npy
            
            feature_save_path = os.path.join(feature_save_dir, video_name)
            np.save(feature_save_path, outputs_f)
            
#             predicted = predicted.squeeze(0).tolist()
#             labels = [label.item() for label in labels]
#             mask = [m.item() for m in mask]
# 
#             pic_file = video_name.split('.')[0] + '-vis.png'
#             pic_path = os.path.join(pic_save_dir, pic_file)
#             
#             confidence = confidence.squeeze(0).tolist()
#             segment_bars_with_confidence_score(pic_path, confidence_score=confidence, labels=[labels, predicted, mask])
            
            print(video_name, ' done!')
            
            
            
def base_predict(model, test_loader, argdataset, sample_rate):
    model.eval()
    model.to(device)
    
    pic_save_dir = 'results/base_vis/'
#     results_dir = 'cholec80/eva/test_dataset/'
    results_dir = 'm2cai16/eva/test_dataset'  # for m2cai16
    if not os.path.exists(pic_save_dir):
        os.makedirs(pic_save_dir)    
    with torch.no_grad():
        for (video, labels, mask, video_name) in tqdm(test_loader):
#             zero_one_mask = (mask!=7).to(device).float()    
            mask = torch.Tensor(mask).float()
            video = video.to(device)
            mask = mask.to(device)
            re = model(video)
            
#             re = model(video, mask)
            confidence, predicted = torch.max(F.softmax(re.data,1), 1)
            predicted = predicted.squeeze(0).tolist()
            labels = [label.item() for label in labels]
            
            pic_file = video_name[0].split('.')[0] + '-vis.png'
            pic_path = os.path.join(pic_save_dir, pic_file)
            mask = [m.item() for m in mask]
            confidence = confidence.squeeze(0).tolist()
            
            alpha = 3
            beta = 0.95
            gamma = 30
#             beta2 = 0
            
            predicted, _ = PKI(confidence, predicted, transtion_prior_matrix, alpha, beta, gamma)
            
#             segment_bars(pic_path, labels, predicted, mask)
            segment_bars_with_confidence_score(pic_path, confidence_score=confidence, labels=[labels, predicted, mask])
            
           
            
            predicted_phases_txt = label2phase(predicted, phase2label_dict=phase2label_dicts[argdataset])
            predicted_phases_expand = []
            for i in predicted_phases_txt:
                predicted_phases_expand = np.concatenate((predicted_phases_expand, [i] * 5 * sample_rate)) # 5 is the resampled resolution
 
 
            target_video_file = video_name[0].split('.')[0] + '_pred.txt'
#             gt_file = video_name[0].split('.')[0] + '-phase.txt'
            gt_file = video_name[0].split('.')[0] + '.txt'  # for m2cai16
 
            g_ptr = open(os.path.join(results_dir, gt_file), "r")
            f_ptr = open(os.path.join(results_dir, target_video_file), 'w')
 
            gt = g_ptr.readlines()[1:] ##
            predicted_phases_expand = predicted_phases_expand[0:len(gt)]
            assert len(predicted_phases_expand) == len(gt)
 
            f_ptr.write("Frame\tPhase\n")
            for index, line in enumerate(predicted_phases_expand):
                f_ptr.write('{}\t{}\n'.format(index, line))
            f_ptr.close()


def grid_search(model, test_loader):
    model.to(device)
    model.eval()
    
    results_dict = {}
    with torch.no_grad():
        correct = 0
        total = 0
        for (video, labels, mask, video_name) in (test_loader):
            labels = torch.Tensor(labels).long()
            mask = torch.Tensor(mask).float()
            video, labels = video.to(device), labels.to(device)
            mask = mask.to(device)
#             if load_hard_frames:
#                 outputs = model(video, mask)
#             else:
#                 outputs = model(video)
            outputs = model(video)
            confidence, predicted = torch.max(F.softmax(outputs.data, dim=1), 1)
            predicted = predicted.tolist()[0]
            confidence = confidence.tolist()[0]
            gt = labels.tolist()
            
            results_dict[video_name[0].split('.')[0]] = [predicted, confidence, gt]
            
#     alpha_choices = [1,2,3,4,5,6,7,8,9,10,15,20]
#     beta_choices = [0,0.1,0.2,0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
#     beta2_choices = [0,0.1,0.2,0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
#     gamma_choices = [30,40,50,60,70,80,90,100]
    alpha_choices = [3]
    beta_choices = [0.95]
    gamma_choices = [30]
#     alpha2_choices = [1,5,6,7,8,9,10,11,12,13,14,15,20,25,30]
#     beta2_choices = [0.2,0.3,0.4,0.5,0.6,0.65,0.7,0.75,0.8,0.9,0.95]

    pic_dir = 'results/pki_vis/'
    if not os.path.exists(pic_dir):
        os.makedirs(pic_dir)
    c_best = 0
    c_beta = 0
    c_alpha = 0
    c_gamma = 0
    c_beta2 = 0
    c_alpha2 = 0
    for alpha in alpha_choices:
        for beta in beta_choices:
            for gamma in gamma_choices:
                for beta2 in beta2_choices:
                    for alpha2 in alpha2_choices:
                        correct = 0
                        total = 0
                    
                        for _, result in results_dict.items():
            #                 import pdb
            #                 pdb.set_trace()
                            refined_predicted, confidence = PKI(result[1], result[0], transtion_prior_matrix, alpha, beta, gamma)
#                             refined_predicted = PKI2(confidence, refined_predicted, alpha2, beta2)
    
                            pic_path = os.path.join(pic_dir, _+'.png')
    #                         segment_bars_with_confidence_score(pic_path, confidence_score=result[1], labels=[result[2], result[0], refined_predicted])
                            correct += np.sum(np.array(refined_predicted) == np.array(result[2]))
                            total += len(refined_predicted)
                        print('alpha: {} beta: {}, gamma:{} beta2:{} acc: {}'.format(alpha, beta, gamma, beta2, correct/total))
                        if correct / total > c_best:
                            c_best = correct/ total
                            c_alpha = alpha
                            c_beta = beta
                            c_gamma = gamma
                            c_beta2 = beta2
                            c_alpha2 = alpha2
    print('Best alpha :{} beta: {} gamma:{}  beta2:{} alpha2:{} best :{}'.format(c_alpha, c_beta, c_gamma, c_beta2, c_alpha2, c_best)) 
    
    
base_model = model.BaseCausalTCN(num_layers, num_f_maps, dim, num_classes)
# refine_model = model.RefineCausualTCN(4, num_f_maps, num_classes, num_classes)
refine_model = model.MultiStageRefineGRU(num_stage=num_stages, num_f_maps=128, num_classes=num_classes)
# causal_tcn = model.MultiStageCausalTCN(num_stages, num_layers, num_f_maps, dim, num_classes)

if args.action == 'base_train':
    load_hard_frames = True if args.dataset == 'cholec80' else False # we do not calculate hard framesfor m2cai16 dataset
    print('load hard frames ', load_hard_frames)
    video_traindataset = VideoDataset(args.dataset, '{}/train_dataset'.format(args.dataset), sample_rate, 'video_feature@2020', load_hard_frames=load_hard_frames)
    video_train_dataloader = DataLoader(video_traindataset, batch_size=1, shuffle=True, drop_last=False)
    video_testdataset = VideoDataset(args.dataset, '{}/test_dataset'.format(args.dataset), sample_rate, 'video_feature@2020', load_hard_frames=load_hard_frames)
    video_test_dataloader = DataLoader(video_testdataset, batch_size=1, shuffle=False, drop_last=False)
    
    base_model.load_state_dict(torch.load('models/base_tcn/100.model')) # fine_tune on the m2cai16
    model_save_dir = 'models/{}/base_tcn'.format(args.dataset)
    base_train(base_model, video_train_dataloader, video_test_dataloader, load_hard_frames=load_hard_frames, save_dir=model_save_dir)

if args.action == 'base_test':
    model_path = 'models/base_tcn/100.model' if args.dataset == 'cholec80' else 'models/m2cai16/base_tcn/100.model'
    load_hard_frames = True if args.dataset == 'cholec80' else False # we do not calculate hard framesfor m2cai16 dataset
    print(load_hard_frames)
    base_model.load_state_dict(torch.load(model_path))
    video_testdataset = VideoDataset(args.dataset, '{}/train_dataset'.format(args.dataset), sample_rate, 'video_feature@2020', load_hard_frames=load_hard_frames)
    video_test_dataloader = DataLoader(video_testdataset, batch_size=1, shuffle=False, drop_last=False)
    base_test(base_model, video_test_dataloader, load_hard_frames=load_hard_frames)

if args.action == 'base_predict':
    load_hard_frames = True if args.dataset == 'cholec80' else False # we do not calculate hard framesfor m2cai16 dataset
    model_path = 'models/base_tcn/100.model' if args.dataset == 'cholec80' else 'models/m2cai16/base_tcn/100.model' # for m2cai16

    base_model.load_state_dict(torch.load(model_path))
    video_testdataset =VideoDataset(args.dataset, '{}/test_dataset'.format(args.dataset), sample_rate, 'video_feature@2020', load_hard_frames=load_hard_frames)
    video_test_dataloader = DataLoader(video_testdataset, batch_size=1, shuffle=False, drop_last=False)
#     results_dir = os.path.join(args.dataset, 'eva/test_dataset')
    base_predict(base_model,video_test_dataloader,args.dataset, sample_rate)
    

if args.action == 'refine_train':
    base_model_path = 'models/base_tcn/100.model'
    base_model.load_state_dict(torch.load(base_model_path))
    
    # the sample rate for prob sequence is 1.
    video_traindataset = VideoDataset('cholec80', 'cholec80/train_dataset', 1, 'refine_model_video_feature@2020/epoch-100', load_hard_frames=True)
#     video_traindataset = VideoDataset('cholec80', 'cholec80/train_dataset', 1, 'mask_hard_frame_video_feature@2020', load_hard_frames=True)
#     video_traindataset = VideoDataset('cholec80', 'cholec80/train_dataset', 1, 'random_mask_video_feature@2020', load_hard_frames=True)
#     train_num = 96
#     for i in range(99,train_num, -1):
#         video_back_traindataset = VideoDataset('cholec80', 'cholec80/train_dataset', 1, 'refine_model_video_feature@2020/epoch-{}'.format(train_num), load_hard_frames=True)
#         video_traindataset.merge(video_back_traindataset) # add more training data
    video_back_traindataset = VideoDataset('cholec80', 'cholec80/train_dataset', 1, 'mask_hard_frame_video_feature@2020', load_hard_frames=True)
#     video_back_traindataset = VideoDataset('cholec80', 'cholec80/train_dataset', 1, 'random_mask_video_feature@2020', load_hard_frames=True)
    video_traindataset.merge(video_back_traindataset) # add more training data
#     video_back_traindataset_2 = VideoDataset('cholec80', 'cholec80/train_dataset', 1, 'random_mask_video_feature@2020', load_hard_frames=True)
#     video_traindataset.merge(video_back_traindataset_2)
    video_train_dataloader = DataLoader(video_traindataset, batch_size=1, shuffle=True, drop_last=False)
    
    
    video_testdataset = VideoDataset('cholec80', 'cholec80/test_dataset', sample_rate, 'video_feature@2020', load_hard_frames=True)
    video_test_dataloader = DataLoader(video_testdataset, batch_size=1, shuffle=False, drop_last=False)
    
    refine_train(base_model, refine_model, video_train_dataloader, video_test_dataloader)
    
if args.action == 'refine_test':
    base_model_path = 'models/base_tcn/100.model'
    base_model.load_state_dict(torch.load(base_model_path))

    refine_model = model.MultiStageRefineGRU(num_stage=5, num_f_maps=128, num_classes=num_classes)

#     video_testdataset =VideoDataset('cholec80', 'cholec80/train_dataset', 1, 'refine_model_video_feature@2020/epoch-100', load_hard_frames=True)
    video_testdataset =VideoDataset('cholec80', 'cholec80/test_dataset', sample_rate, 'video_feature@2020', load_hard_frames=True)

    video_test_dataloader = DataLoader(video_testdataset, batch_size=1, shuffle=False, drop_last=False)
    for i in range(1,50):
        refine_model_path = 'models/refine_model_stage5/{}.model'.format(i)
        refine_model.load_state_dict(torch.load(refine_model_path))
        print(i)
        refine_test(base_model, refine_model, video_test_dataloader)

if args.action == 'refine_predict':
    refine_model = model.MultiStageRefineGRU(num_stage=4, num_f_maps=128, num_classes=num_classes)
    load_hard_frames = True if args.dataset == 'cholec80' else False # we do not calculate hard framesfor m2cai16 dataset
    base_model_path = 'models/base_tcn/100.model'
    refine_model_path = 'models/refine_model/31.model' # 31 
    base_model.load_state_dict(torch.load(base_model_path))
    refine_model.load_state_dict(torch.load(refine_model_path))
    
#     video_testdataset =VideoDataset('cholec80', 'cholec80/train_dataset', 1, 'refine_model_video_feature@2020', load_hard_frames=True)
    video_testdataset =VideoDataset(args.dataset, '{}/test_dataset'.format(args.dataset), sample_rate, 'video_feature@2020', load_hard_frames=load_hard_frames)

    video_test_dataloader = DataLoader(video_testdataset, batch_size=1, shuffle=False, drop_last=False)
    refine_predict(base_model, refine_model, video_test_dataloader)
    
if args.action == 'cross_validate': # in order to get training data for the refine model
    # useless
    kf = KFold(10, shuffle=True, random_state=seed) # 10-fold cross validate
    video_list = ['video{0:>2d}'.format(i) for i in range(1,41)]
    for k, (train_idx, test_idx) in enumerate(kf.split(video_list)):
        if args.k != -100:
            if k != args.k:
                continue
        base_model = model.BaseCausalTCN(num_layers, num_f_maps, dim, num_classes)
        
        trainlist = ['video{:0>2d}'.format(i+1) for i in train_idx]
        testlist = ['video{:0>2d}'.format(i+1) for i in test_idx]
        
        print('split {}: {}'.format(k, '_'.join(testlist)))
        train_video_dir = 'cross_validate_video_feature@2020/{}/'.format('_'.join(testlist))
        test_video_dir = 'cross_validate_video_feature_for_validation@2020/'
        
        model_save_dir = 'models/cross_validate/base_tcn/{}'.format('_'.join(testlist))
#         video_traindataset = VideoDataset('cholec80', 'cholec80/train_dataset', sample_rate, train_video_dir, load_hard_frames=True)
        video_testdataset = VideoDataset('cholec80', 'cholec80/train_dataset', sample_rate, test_video_dir, blacklist=trainlist, load_hard_frames=True)
#         video_train_dataloader = DataLoader(video_traindataset, batch_size=1, shuffle=True, drop_last=False)
        video_test_dataloader = DataLoader(video_testdataset, batch_size=1, shuffle=False, drop_last=False)
        
#         for epoch in range(50, 101):
        base_model.load_state_dict(torch.load(model_save_dir + '/{}.model'.format(epochs)))
    #         base_train(base_model, video_train_dataloader, video_test_dataloader, save_dir=model_save_dir)
            
        feature_save_dir = 'cholec80/train_dataset/refine_model_video_feature@2020/epoch-{}/'.format(epochs)
        pic_save_dir = 'results/prior_train_data/'
        print('Training Done!')
        extract(base_model, video_test_dataloader, feature_save_dir, pic_save_dir)

    
if args.action == 'grid_search':
    load_hard_frames = True if args.dataset == 'cholec80' else False # we do not calculate hard framesfor m2cai16 dataset
    model_path = 'models/base_tcn/100.model' if args.dataset == 'cholec80' else 'models/m2cai16/base_tcn/100.model' # for m2cai16

    base_model.load_state_dict(torch.load(model_path))
    video_testdataset =VideoDataset(args.dataset, '{}/test_dataset'.format(args.dataset), sample_rate, 'video_feature@2020', load_hard_frames=load_hard_frames)
    video_test_dataloader = DataLoader(video_testdataset, batch_size=1, shuffle=False, drop_last=False)
#     results_dir = os.path.join(args.dataset, 'eva/test_dataset')
    grid_search(base_model,video_test_dataloader)
    
    
