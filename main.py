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

def refine_train(base_model, refine_model, train_loader, validation_loader, save_dir='models/refine_gru/', debug=False):
    global learning_rate
    epochs = 31
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
        optimizer = torch.optim.Adam(refine_model.parameters(), learning_rate*10) # 10 50
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
            loss_item += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs[-1].data, 2)
            correct += ((predicted == labels).sum()).item()
            total += labels.shape[0]
            

        print('Train Epoch {}: Acc {}, Loss {}'.format(epoch, correct / total, loss_item / total))
        if debug:
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
            outputs1 = base_model(video, mask)

            outputs1 = outputs1.permute(0,2,1)
            outputs2, _ = refine_model(outputs1)
            
            _, predicted1 = torch.max(outputs1.data, 2)
            _, predicted2 = torch.max(outputs2[-1].data, 2)

            correct1 += ((predicted1 == labels).sum()).item()
            correct2 += ((predicted2== labels).sum()).item()
            total += labels.shape[0]

        print('Test: Base model Acc {}, Prior Model Acc {}'.format(correct1 / total, correct2/total))
    

def refine_predict(base_model, refine_model, test_loader):
    base_model.eval()
    refine_model.eval()
    base_model.to(device)
    refine_model.to(device)
    
    pic_save_dir = 'results/{}/refine_vis/'.format(args.dataset)
    results_dir = '{}/eva/test_dataset/'.format(args.dataset)
    if not os.path.exists(pic_save_dir):
        os.makedirs(pic_save_dir)    
    with torch.no_grad():
        for (video, labels, mask, video_name) in tqdm(test_loader):
            video = video.to(device)
            mask = torch.Tensor(mask).float()
            mask = mask.to(device)

            base_model_output = base_model(video, mask)

            
            ###
            confidence, base_predicted = torch.max(F.softmax(base_model_output.data,1), 1)
            base_predicted = base_predicted.squeeze(0).tolist()
            confidence = confidence.squeeze(0).tolist()
             
 
#             alpha = 3
#             beta = 0.95
#             gamma = 30            
#             pki_predicted = PKI(confidence, base_predicted, transtion_prior_matrix, alpha, beta, gamma)
            
            
            
            outputs, _ = refine_model(base_model_output.permute(0,2,1))
            confidence, predicted = torch.max(F.softmax(outputs[-1].data,2), 2)
            predicted = predicted.squeeze(0).tolist()
            labels = [label.item() for label in labels]
            
            pic_file = video_name[0].split('.')[0] + '-vis.png'
            pic_path = os.path.join(pic_save_dir, pic_file)
            mask = [m.item() for m in mask]
            confidence = confidence.squeeze(0).tolist()
             
#             segment_bars(pic_path, labels, base_predicted, pki_predicted, predicted)
            segment_bars_with_confidence_score(pic_path, confidence_score=confidence, labels=[labels, base_predicted, predicted])
            
            predicted_phases_txt = label2phase(predicted, phase2label_dict=phase2label_dicts[args.dataset])
            predicted_phases_expand = []
            for i in predicted_phases_txt:
                predicted_phases_expand = np.concatenate((predicted_phases_expand, [i] * 5 * sample_rate)) # 5 is the resampled resolution
 
 
            target_video_file = video_name[0].split('.')[0] + '_pred.txt'
            if args.dataset == 'cholec80':
                gt_file = video_name[0].split('.')[0] + '-phase.txt'
            if args.dataset == 'm2cai16':
                gt_file = video_name[0].split('.')[0] + '.txt'
                

 
            g_ptr = open(os.path.join(results_dir, gt_file), "r")
            f_ptr = open(os.path.join(results_dir, target_video_file), 'w')
 
            gt = g_ptr.readlines()[1:] ##
            predicted_phases_expand = predicted_phases_expand[0:len(gt)]
            assert len(predicted_phases_expand) == len(gt)
 
            f_ptr.write("Frame\tPhase\n")
            for index, line in enumerate(predicted_phases_expand):
                f_ptr.write('{}\t{}\n'.format(index, line))
            f_ptr.close()

    
    
def base_train(model, train_loader, validation_loader, save_dir = 'models/base_tcn', debug = False):
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
            loss += loss_layer(outputs.transpose(2, 1).contiguous().view(-1, num_classes), labels.view(-1)) # cross_entropy loss
            loss += torch.mean(torch.clamp(mse_layer(F.log_softmax(outputs[:, :, 1:], dim=1), F.log_softmax(outputs.detach()[:, :, :-1], dim=1)), min=0, max=16)) # smooth loss

            loss_item += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            correct += ((predicted == labels).sum()).item()
            total += labels.shape[0]

        print('Train Epoch {}: Acc {}, Loss {}'.format(epoch, correct / total, loss_item / total))
        if debug:
            base_test(model, validation_loader)
        torch.save(model.state_dict(), save_dir + '/{}.model'.format(epoch))

def base_test(model, test_loader, save_prediction=False, random_mask=False):
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        correct = 0
        total = 0
        for (video, labels, mask, video_name) in (test_loader):
            labels = torch.Tensor(labels).long()
            if random_mask:
                # random_mask
                mask = np.random.choice(2, len(mask), replace=True, p=[0.3,0.7])
                mask = torch.from_numpy(mask).float().to(device)
            else:
                mask = torch.Tensor(mask).float().to(device)
            video, labels = video.to(device), labels.to(device)
            mask = mask.to(device)
            outputs = model(video, mask)
            _, predicted = torch.max(outputs.data, 1)
            correct += ((predicted == labels).sum()).item()
            total += labels.shape[0]
            
            if save_prediction:
                if random_mask:
                    feature_save_dir = '{}/train_dataset/random_mask_type@2020/'.format(args.dataset)
                else:
                    feature_save_dir = '{}/train_dataset/mask_hard_frame_type@2020/'.format(args.dataset)
                if not os.path.exists(feature_save_dir):
                    os.makedirs(feature_save_dir)
                  
                outputs_f = outputs.squeeze(0).permute(1,0).to('cpu').numpy() # of shape (l, c)
                video_name = video_name[0] # videoxx.npy
                feature_save_path = os.path.join(feature_save_dir, video_name)
                np.save(feature_save_path, outputs_f)
    
        print('Test: Acc {}'.format(correct / total))

def base_predict(model, test_loader, argdataset, sample_rate, pki = False):
    model.eval()
    model.to(device)
    pic_save_dir = 'results/{}/base_vis/'.format(args.dataset)
    results_dir = '{}/eva/test_dataset/'.format(args.dataset)
    if not os.path.exists(pic_save_dir):
        os.makedirs(pic_save_dir)

    with torch.no_grad():
        for (video, labels, mask, video_name) in tqdm(test_loader):
            mask = torch.Tensor(mask).float()
            video = video.to(device)
            mask = mask.to(device)
            re = model(video, mask)


            confidence, predicted = torch.max(F.softmax(re.data,1), 1)
            predicted = predicted.squeeze(0).tolist()
            confidence = confidence.squeeze(0).tolist()
            labels = [label.item() for label in labels]
            
            pic_file = video_name[0].split('.')[0] + '-vis.png'
            pic_path = os.path.join(pic_save_dir, pic_file)
            segment_bars_with_confidence_score(pic_path, confidence_score=confidence, labels=[labels, predicted])

            if pki:
                # best hyper by grid search
                alpha = 3
                beta = 0.95
                gamma = 30            
                predicted, _ = PKI(confidence, predicted, transtion_prior_matrix, alpha, beta, gamma)
                        
            
            predicted_phases_txt = label2phase(predicted, phase2label_dict=phase2label_dicts[argdataset])
            predicted_phases_expand = []
            for i in predicted_phases_txt:
                predicted_phases_expand = np.concatenate((predicted_phases_expand, [i] * 5 * sample_rate)) # we downsample the framerate from 25fps to 5fps
 
            
            target_video_file = video_name[0].split('.')[0] + '_pred.txt'
            if args.dataset == 'm2cai16':
                gt_file = video_name[0].split('.')[0] + '.txt'
            else:
                gt_file = video_name[0].split('.')[0] + '-phase.txt'
 
            g_ptr = open(os.path.join(results_dir, gt_file), "r")
            f_ptr = open(os.path.join(results_dir, target_video_file), 'w')
 
            gt = g_ptr.readlines()[1:] ##
            predicted_phases_expand = predicted_phases_expand[0:len(gt)]
            assert len(predicted_phases_expand) == len(gt)
 
            f_ptr.write("Frame\tPhase\n")
            for index, line in enumerate(predicted_phases_expand):
                f_ptr.write('{}\t{}\n'.format(index, line))
            f_ptr.close()


def cross_validate(model, data_loader, feature_save_dir):
    model.to(device)
    model.eval()
    if not os.path.exists(feature_save_dir):
        os.makedirs(feature_save_dir)
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
            print(video_name, ' done!')
            
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
            outputs = model(video, mask)
            confidence, predicted = torch.max(F.softmax(outputs.data, dim=1), 1)
            predicted = predicted.tolist()[0]
            confidence = confidence.tolist()[0]
            gt = labels.tolist()
            
            results_dict[video_name[0].split('.')[0]] = [predicted, confidence, gt]
            
    alpha_choices = [1,2,3,4,5,6,7,8,9,10,15,20]
    beta_choices = [0,0.1,0.2,0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    gamma_choices = [30,40,50,60,70,80,90,100]
#     alpha_choices = [3]
#     beta_choices = [0.95]
#     gamma_choices = [30]

    c_best = 0
    c_beta = 0
    c_alpha = 0
    c_gamma = 0
    for alpha in alpha_choices:
        for beta in beta_choices:
            for gamma in gamma_choices:
                correct = 0
                total = 0
            
                for _, result in results_dict.items():
                    refined_predicted = PKI(result[1], result[0], transtion_prior_matrix, alpha, beta, gamma)
                    correct += np.sum(np.array(refined_predicted) == np.array(result[2]))
                    total += len(refined_predicted)
                print('alpha: {} beta: {}, gamma:{} beta2:{} acc: {}'.format(alpha, beta, gamma, correct/total))
                if correct / total > c_best:
                    c_best = correct/ total
                    c_alpha = alpha
                    c_beta = beta
                    c_gamma = gamma
                    
    print('Best alpha :{} beta: {} gamma:{} best :{}'.format(c_alpha, c_beta, c_gamma, c_best)) 
    
    
base_model = model.BaseCausalTCN(num_layers, num_f_maps, dim, num_classes)
refine_model = model.MultiStageRefineGRU(num_stage=num_stages, num_f_maps=128, num_classes=num_classes) #128

if args.action == 'base_train':
    video_traindataset = VideoDataset(args.dataset, '{}/train_dataset'.format(args.dataset), sample_rate, 'video_feature@2020')
    video_train_dataloader = DataLoader(video_traindataset, batch_size=1, shuffle=True, drop_last=False)
    video_testdataset = VideoDataset(args.dataset, '{}/test_dataset'.format(args.dataset), sample_rate, 'video_feature@2020')
    video_test_dataloader = DataLoader(video_testdataset, batch_size=1, shuffle=False, drop_last=False)
    
    model_save_dir = 'models/{}/base_tcn'.format(args.dataset)
    base_train(base_model, video_train_dataloader, video_test_dataloader, save_dir=model_save_dir, debug=True)


if args.action == 'base_predict':
    model_path = 'saved_models/{}/base_tcn/base_tcn.model'.format(args.dataset) # use the saved model
#     model_path = 'models/{}/base_tcn/100.model'.format(args.dataset) # use your model
    base_model.load_state_dict(torch.load(model_path))
    video_testdataset =VideoDataset(args.dataset, '{}/test_dataset'.format(args.dataset), sample_rate, 'video_feature@2020')
    video_test_dataloader = DataLoader(video_testdataset, batch_size=1, shuffle=False, drop_last=False)
    base_predict(base_model,video_test_dataloader,args.dataset, sample_rate)
    

if args.action == 'refine_train':
    base_model_path = 'saved_models/{}/base_tcn/base_tcn.model'.format(args.dataset) # use the saved model
#     base_model_path = 'models/{}/base_tcn/100.model'.format(args.dataset)  # use your model
    base_model.load_state_dict(torch.load(base_model_path))
    
    # the sample rate for prob sequence is 1.
    video_traindataset = VideoDataset(args.dataset, '{}/train_dataset'.format(args.dataset), 1, 'cross_validate_type@2020')
    video_back_traindataset = VideoDataset(args.dataset, '{}/train_dataset'.format(args.dataset), 1, 'mask_hard_frame_type@2020')
    video_traindataset.merge(video_back_traindataset) # add more training data
#     video_back_traindataset_2 = VideoDataset('cholec80', 'cholec80/train_dataset', 1, 'random_mask_type@2020')
#     video_traindataset.merge(video_back_traindataset_2)
    video_train_dataloader = DataLoader(video_traindataset, batch_size=1, shuffle=True, drop_last=False)
    
    video_testdataset = VideoDataset(args.dataset, '{}/test_dataset'.format(args.dataset), sample_rate, 'video_feature@2020')
    video_test_dataloader = DataLoader(video_testdataset, batch_size=1, shuffle=False, drop_last=False)
    refine_train(base_model, refine_model, video_train_dataloader, video_test_dataloader, save_dir='models/{}/refine_gru/'.format(args.dataset), debug=True)
    

if args.action == 'refine_predict':
    base_model_path = 'saved_models/{}/base_tcn/base_tcn.model'.format(args.dataset)
    refine_model_path = 'saved_models/{}/refine_gru/refine_gru.model'.format(args.dataset)
#     
#     base_model_path = 'models/{}/base_tcn/100.model'.format(args.dataset) # your model
#     refine_model_path = 'models/{}/refine_gru/31.model'.format(args.dataset) # your model, 31
    base_model.load_state_dict(torch.load(base_model_path))
    refine_model.load_state_dict(torch.load(refine_model_path))
    
    video_testdataset =VideoDataset(args.dataset, '{}/test_dataset'.format(args.dataset), sample_rate, 'video_feature@2020')
    video_test_dataloader = DataLoader(video_testdataset, batch_size=1, shuffle=False, drop_last=False)
    refine_predict(base_model, refine_model, video_test_dataloader)
    
    

if args.action == 'cross_validate_type': # get cross-validate-type disturbed prediction sequence
    kf = KFold(10, shuffle=True, random_state=seed) # 10-fold cross validate
    if args.dataset == 'cholec80':
        video_list = ['video{0:>2d}'.format(i) for i in range(1,41)]
    else:
        video_list = ['workflow_video_{:0>2d}'.format(i) for i in range(1, 28)]

    for k, (train_idx, test_idx) in enumerate(kf.split(video_list)):
        if args.k != -100:
            if k != args.k:
                continue
        base_model = model.BaseCausalTCN(num_layers, num_f_maps, dim, num_classes)
        
        if args.dataset == 'cholec80':
            trainlist = ['video{:0>2d}'.format(i+1) for i in train_idx]
            testlist = ['video{:0>2d}'.format(i+1) for i in test_idx]
        elif args.dataset == 'm2cai16':
            trainlist = ['workflow_video_{:0>2d}'.format(i+1) for i in train_idx]
            testlist = ['workflow_video_{:0>2d}'.format(i+1) for i in test_idx]
        
        
        print('split {}: {}'.format(k, '_'.join(testlist)))
        train_video_dir = 'cross_validate_video_feature@2020/{}/'.format('_'.join(testlist)) # pre-extracted inceptionv3 feature
        test_video_dir = 'cross_validate_video_feature_for_validation@2020/' # pre-extracted inceptionv3 feature
        
        model_save_dir = 'models/{}/cross_validate/base_tcn/{}'.format(args.dataset, '_'.join(testlist))
        video_traindataset = VideoDataset(args.dataset, '{}/train_dataset'.format(args.dataset), sample_rate, train_video_dir)
        video_testdataset = VideoDataset(args.dataset, '{}/train_dataset'.format(args.dataset), sample_rate, test_video_dir, blacklist=trainlist)
        video_train_dataloader = DataLoader(video_traindataset, batch_size=1, shuffle=True, drop_last=False)
        video_test_dataloader = DataLoader(video_testdataset, batch_size=1, shuffle=False, drop_last=False)
        
        saved_model_dir = 'saved_models/{}/cross_validate/base_tcn/{}/base_tcn.model'.format(args.dataset, '_'.join(testlist))
        base_model.load_state_dict(torch.load(saved_model_dir))  # load saved model
#         base_model.load_state_dict(torch.load(model_save_dir + '/{}.model'.format(epochs))) # load your model
#         base_train(base_model, video_train_dataloader, video_test_dataloader, save_dir=model_save_dir) # train your model
            
        feature_save_dir = '{}/train_dataset/cross_validate_type@2020/'.format(args.dataset)
        print('Training Done!')
        cross_validate(base_model, video_test_dataloader, feature_save_dir)


if args.action == 'mask_hard_frame_type': # get mask_hard_frame_type disturbed prediction sequence
    model_path = 'saved_models/{}/base_tcn/base_tcn.model'.format(args.dataset) # use the saved model
#     model_path = 'models/{}/base_tcn/100.model'.format(args.dataset) # use your model
    base_model.load_state_dict(torch.load(model_path))
    video_testdataset =VideoDataset(args.dataset, '{}/train_dataset'.format(args.dataset), sample_rate, 'video_feature@2020')
    video_test_dataloader = DataLoader(video_testdataset, batch_size=1, shuffle=False, drop_last=False)
    base_test(base_model, video_test_dataloader, save_prediction=True, random_mask=False)
    
if args.action == 'random_mask_type':
    model_path = 'saved_models/{}/base_tcn/base_tcn.model'.format(args.dataset) # use the saved model
#     model_path = 'models/{}/base_tcn/100.model'.format(args.dataset) # use your model
    base_model.load_state_dict(torch.load(model_path))
    video_testdataset =VideoDataset(args.dataset, '{}/train_dataset'.format(args.dataset), sample_rate, 'video_feature@2020')
    video_test_dataloader = DataLoader(video_testdataset, batch_size=1, shuffle=False, drop_last=False)
    base_test(base_model, video_test_dataloader, save_prediction=True, random_mask=True)

if args.action == 'grid_search': # grid search the best hyper for PKI
    model_path = 'saved_models/{}/base_tcn/base_tcn.model'.format(args.dataset) # use the saved model
#     model_path = 'models/base_tcn/100.model' # use your model

    base_model.load_state_dict(torch.load(model_path))
    video_testdataset =VideoDataset(args.dataset, '{}/test_dataset'.format(args.dataset), sample_rate, 'video_feature@2020')
    video_test_dataloader = DataLoader(video_testdataset, batch_size=1, shuffle=False, drop_last=False)
    grid_search(base_model,video_test_dataloader)
    
if args.action == 'vis_disturbed_sequence':
    sequence_dir = 'random_mask_type@2020'
    pic_dir = 'disturbed_vis/{}/{}/'.format(args.dataset, sequence_dir)
    if not os.path.exists(pic_dir):
        os.makedirs(pic_dir)
    video_traindataset = VideoDataset(args.dataset, '{}/train_dataset'.format(args.dataset), 1, sequence_dir)
    video_train_dataloader = DataLoader(video_traindataset, batch_size=1, shuffle=True, drop_last=False)
    correct = total = 0
    for (video, labels, mask, video_name) in (video_train_dataloader):
        ## labels is sample_rate times longer than video
        labels = labels[::sample_rate]

        video_name = video_name[0].split('.')[0]

        confidence, original_predicted = torch.max(F.softmax(video.data,2), 2)
        
        correct += (original_predicted==torch.Tensor(labels).long()).sum().item()
        total += len(labels)
        

        confidence = confidence.squeeze(0).tolist()
        original_predicted= original_predicted.squeeze(0).tolist()

        labels = [label.item() for label in labels]

        segment_bars_with_confidence_score(pic_dir + video_name + '.png', confidence_score=confidence, labels=[labels, original_predicted])
        print(video_name, ' done!')
    print(correct / total)
        

        

    
    
