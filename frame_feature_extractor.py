# This file contains the code about:
# 1. Extract inceptionv3 features for the video frame.
# 2. Find out hard frames.
# This step is not necessary if you download the extracted feature we provided.
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
from sklearn.model_selection  import KFold


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
parser.add_argument('--action', choices=['train', 'extract', 'test', 'hard_frame'], default='train')
parser.add_argument('--dataset', default="cholec80", choices=['cholec80','m2cai16'])
parser.add_argument('--target', type=str, default='train_set')
parser.add_argument('--k', type=int, default=-100)
args = parser.parse_args()

learning_rate = 1e-4
epochs = 3
loss_layer = nn.CrossEntropyLoss()


def train(model, save_dir, train_loader, validation_loader):
    global learning_rate
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model.to(device)
    for epoch in range(1, epochs + 1):
        if epoch % 2 == 0:
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
#         test(model, validation_loader)
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


def extract(model, loader, save_path, record_err= False):
    model.eval()
    model.to(device)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    err_dict = {}
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
            
            _,  prediction = torch.max(res.data, 1)
            if record_err and (prediction == labels).sum().item() == 0:
                # hard frames
                if video not in err_dict.keys():
                    err_dict[video] = []
                else:
                    err_dict[video].append(int(img_in_video.split('.')[0]))

            features = features.to('cpu').numpy() # of shape 1 x 2048

            np.save(feature_save_path, features)
    
    return err_dict



def imgf2videof(source_folder, target_folder):
    '''
        Merge the extracted img feature to video feature.
    '''
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    for video in os.listdir(source_folder):
        video_feature_save_path = os.path.join(target_folder, video + '.npy')
        video_abs_path = os.path.join(source_folder, video)
        nums_of_imgs = len(os.listdir(video_abs_path))
        video_feature = []
        for i in range(nums_of_imgs):
            img_abs_path = os.path.join(video_abs_path, str(i) + '.npy')
            video_feature.append(np.load(img_abs_path))

        video_feature = np.concatenate(video_feature, axis=0)
        
        np.save(video_feature_save_path, video_feature)
        print('{} done!'.format(video))




if __name__ == '__main__':
    inception = inception_v3(pretrained=True, aux_logits=False)
    fc_features = inception.fc.in_features
    inception.fc = nn.Linear(fc_features, len(dataset.phase2label_dicts[args.dataset]))
    if args.action == 'train':
        # 
        
        framewise_traindataset = dataset.FramewiseDataset(args.dataset, '{}/train_dataset'.format(args.dataset))
        framewise_train_dataloader = DataLoader(framewise_traindataset, batch_size=64, shuffle=True, drop_last=False)
    
        framewise_testdataset = dataset.FramewiseDataset(args.dataset, '{}/test_dataset'.format(args.dataset))
        framewise_test_dataloader = DataLoader(framewise_testdataset, batch_size=64, shuffle=True, drop_last=False)
    
        train( inception, 'models/{}/inceptionv3'.format(args.dataset), framewise_train_dataloader, framewise_test_dataloader)
    
    if args.action == 'test':
        model_path = 'saved_models/{}/inceptionv3/inceptionv3.model'.format(args.dataset) # use the saved model
#         model_path = 'models/{}/inceptionv3/2.model'.format(args.dataset) # use your own model
        inception.load_state_dict(torch.load(model_path))
        framewise_testdataset = dataset.FramewiseDataset(args.dataset, '{}/test_dataset'.format(args.dataset))
        framewise_test_dataloader = DataLoader(framewise_testdataset, batch_size=64, shuffle=True, drop_last=False)
        test(inception, framewise_test_dataloader)
    
    if args.action == 'extract': # extract inception feature
        model_path = 'saved_models/{}/inceptionv3/inceptionv3.model'.format(args.dataset) # use the saved model
#         model_path = 'models/{}/inceptionv3/2.model'.format(args.dataset) # use your own model
        inception.load_state_dict(torch.load(model_path))
        
        if args.target == 'train_set':
            framewise_testdataset = dataset.FramewiseDataset(args.dataset, '{}/train_dataset'.format(args.dataset))
            framewise_test_dataloader = DataLoader(framewise_testdataset, batch_size=1, shuffle=False, drop_last=False)
        
            extract(inception, framewise_test_dataloader, '{}/train_dataset/frame_feature@2020/'.format(args.dataset))
            imgf2videof('{}/train_dataset/frame_feature@2020/'.format(args.dataset), '{}/train_dataset/video_feature@2020/'.format(args.dataset))
        else:
            framewise_testdataset = dataset.FramewiseDataset(args.dataset, '{}/test_dataset'.format(args.dataset))
            framewise_test_dataloader = DataLoader(framewise_testdataset, batch_size=1, shuffle=False, drop_last=False)
        
            extract(inception, framewise_test_dataloader, '{}/test_dataset/frame_feature@2020/'.format(args.dataset))
            imgf2videof('{}/test_dataset/frame_feature@2020/'.format(args.dataset), '{}/test_dataset/video_feature@2020/'.format(args.dataset))
        
    if args.action == 'hard_frame' and args.target == 'train_set':
        kf = KFold(10, shuffle=True, random_state=seed) # 10-fold cross validate
        if args.dataset == 'cholec80':
            video_list = ['video{:0>2d}'.format(i) for i in range(1,41)]
        elif args.dataset == 'm2cai16':
            video_list = ['workflow_video_{:0>2d}'.format(i) for i in range(1, 28)]
        for k, (train_idx, test_idx) in enumerate(kf.split(video_list)):
            if args.k != -100:
                if k != args.k:
                    continue
            
            inception = inception_v3(pretrained=True, aux_logits=False)
            fc_features = inception.fc.in_features
            inception.fc = nn.Linear(fc_features, len(dataset.phase2label_dicts[args.dataset]))
            
            if args.dataset == 'cholec80':
                trainlist = ['video{:0>2d}'.format(i+1) for i in train_idx]
                testlist = ['video{:0>2d}'.format(i+1) for i in test_idx]
            elif args.dataset == 'm2cai16':
                trainlist = ['workflow_video_{:0>2d}'.format(i+1) for i in train_idx]
                testlist = ['workflow_video_{:0>2d}'.format(i+1) for i in test_idx]
            
            framewise_traindataset = dataset.FramewiseDataset(args.dataset, '{}/train_dataset'.format(args.dataset), blacklist=testlist)
            framewise_testdataset = dataset.FramewiseDataset(args.dataset, '{}/train_dataset'.format(args.dataset), blacklist=trainlist)
            
            framewise_train_dataloader = DataLoader(framewise_traindataset, batch_size=64, shuffle=True, drop_last=False)
            framewise_test_dataloader = DataLoader(framewise_testdataset, batch_size=1, shuffle=True, drop_last=False)
            
            model_save_dir = 'models/{}/cross_validate/inceptionv3/'.format(args.dataset) + '_'.join(testlist)
            print('Cross Validate {}, save dir '.format(k) + model_save_dir)
             
            save_model_dir = 'saved_models/{}/cross_validate/inceptionv3/'.format(args.dataset) + + '_'.join(testlist) +'/inceptionv3.model'
            inception.load_state_dict(torch.load(save_model_dir))
#             inception.load_state_dict(torch.load(model_save_dir + '/3.model'))  # load your model
            
#             train(inception, model_save_dir , framewise_train_dataloader, framewise_test_dataloader) # train your model

            test_extraction_path = '{}/train_dataset/cross_validate_frame_feature_for_validatation@2020/'.format(args.dataset)
            train_extraction_path = '{}/train_dataset/cross_validate_frame_feature@2020/{}/'.format(args.dataset, '_'.join(testlist))
            
            print('Training Done! Extract feature to {} and {}'.format(test_extraction_path, train_extraction_path))
            
            framewise_train_dataloader = DataLoader(framewise_traindataset, batch_size=1, shuffle=True, drop_last=False)
            extract(inception, framewise_train_dataloader, train_extraction_path, False)
            imgf2videof(train_extraction_path, '{}/train_dataset/cross_validate_video_feature@2020/{}/'.format(args.dataset, '_'.join(testlist)))
            err_dict = extract(inception, framewise_test_dataloader, test_extraction_path, True)
            imgf2videof(test_extraction_path, '{}/train_dataset/cross_validate_video_feature_for_validation@2020/'.format(args.dataset))
            
            
            print('Make Hard Frame files at {}/train_dataset/hard_frames@2020/'.format(args.dataset))
             
            if not os.path.exists('{}/train_dataset/hard_frames@2020'.format(args.dataset)):
                os.makedirs('{}/train_dataset/hard_frames@2020'.format(args.dataset))
            for video in testlist:
                anno_file = '{}/train_dataset/annotation_folder/{}.txt'.format(args.dataset,video)
                hard_frame_file = '{}/train_dataset/hard_frames@2020/{}.txt'.format(args.dataset, video)
                with open(anno_file, 'r') as f:
                    gt_lines = f.readlines()
                for err_img in err_dict[video]:
                    ori = gt_lines[err_img]
                    idx, _ = ori.split('\t')
                    assert int(idx) == err_img
                    phase_txt = 'HardFrame\n'
                    gt_lines[err_img] = idx + '\t' + phase_txt
                with open(hard_frame_file, 'w') as f:
                    for line in gt_lines:
                        f.write(line)
        
        
    if args.action == 'hard_frame' and args.target == 'test_set':
        negative_index = 8 if args.dataset=='m2cai16' else 7
        print('hard frame index: ', negative_index)
        inception = inception_v3(pretrained=True, aux_logits=False)
        fc_features = inception.fc.in_features
        inception.fc = nn.Linear(fc_features, len(dataset.phase2label_dicts[args.dataset]) + 1) # plus HardFrame
        
        
        framewise_traindataset = dataset.FramewiseDataset(args.dataset, '{}/train_dataset'.format(args.dataset), label_folder = 'hard_frames@2020')
        framewise_train_dataloader = DataLoader(framewise_traindataset, batch_size=64, shuffle=True, drop_last=False)
    
        framewise_testdataset = dataset.FramewiseDataset(args.dataset, '{}/test_dataset'.format(args.dataset))
        framewise_test_dataloader = DataLoader(framewise_testdataset, batch_size=1, shuffle=False, drop_last=False)
        
        inception.load_state_dict(torch.load('saved_models/{}/inceptionv3_w_hard/inceptionv3.model'.format(args.dataset))) # load saved model
#         inception.load_state_dict(torch.load('models/{}/inceptionv3_w_hard/3.model'.format(args.dataset))) # load your model
#         train(inception, 'models/{}/inceptionv3_w_hard'.format(args.dataset),framewise_train_dataloader, framewise_test_dataloader) # train your model
        
        print('Training Done! Detect hard frames in test dataset')
        inception.eval()
        inception.to(device)
        
        hard_frame_dict = {}
        with torch.no_grad():
            for (imgs, labels, img_names) in tqdm(framewise_test_dataloader):
                assert len(img_names) == 1 # batch_size = 1
                video, img_in_video = img_names[0].split('/')[-2], img_names[0].split('/')[-1] # video63 5730.jpg
                imgs, labels = imgs.to(device), labels.to(device)
                features, res = inception(imgs)
                
                _,  prediction = torch.max(res.data, 1)
                if prediction == negative_index:
                    # hard frames
                    if video not in hard_frame_dict.keys():
                        hard_frame_dict[video] = []
                        hard_frame_dict[video].append(int(img_in_video.split('.')[0]))
                    else:
                        hard_frame_dict[video].append(int(img_in_video.split('.')[0]))
        
#         import pickle
#         f_ptr = open('hard_frame_test_dict.pkl','wb')
#         pickle.dump(hard_frame_dict, f_ptr)
        
        if not os.path.exists('{}/test_dataset/hard_frames@2020'.format(args.dataset)):
            os.makedirs('{}/test_dataset/hard_frames@2020'.format(args.dataset))
        
        if args.dataset == 'cholec80':
            testlist = ['video{:0>2d}'.format(i) for i in range(41,81)]
        elif args.dataset == 'm2cai16':
            testlist = ['test_workflow_video_{:0>2d}'.format(i) for i in range(1,15)]
        for video in testlist:
            anno_file = '{}/test_dataset/annotation_folder/{}.txt'.format(args.dataset, video)
            hard_frame_file = '{}/test_dataset/hard_frames@2020/{}.txt'.format(args.dataset, video)
            with open(anno_file, 'r') as f:
                gt_lines = f.readlines()
            for hard_frame in hard_frame_dict[video]:
                ori = gt_lines[hard_frame]
                idx, _ = ori.split('\t')
                assert int(idx) == hard_frame
                phase_txt = 'HardFrame\n'
                gt_lines[hard_frame] = idx + '\t' + phase_txt
            with open(hard_frame_file, 'w') as f:
                for line in gt_lines:
                    f.write(line)
    
        
    
        
        

    
   


