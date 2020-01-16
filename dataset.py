from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader
from torchvision import transforms
import os
import numpy as np

phase2label_dicts = {
    'cholec80':{'Preparation':0,
    'CalotTriangleDissection':1,
    'ClippingCutting':2,
    'GallbladderDissection':3,
    'GallbladderPackaging':4,
    'CleaningCoagulation':5,
    'GallbladderRetraction':6}
}


def phase2label(phases, phase2label_dict):
    labels = [phase2label_dict[phase] for phase in phases]
    return labels

def label2phase(labels, phase2label_dict):
    label2phase_dict = {phase2label_dict[k]:k for k in phase2label_dict.keys()}
    phases = [label2phase_dict[label] for label in labels]
    return phases

class FramewiseDataset(Dataset):
    def __init__(self, dataset, root):
        self.dataset = dataset
        self.imgs = []
        self.labels = []

        label_folder = os.path.join(root, 'annotation_folder')
        video_folder = os.path.join(root, 'image_folder')
        for v in os.listdir(video_folder):
            v_abs_path = os.path.join(video_folder, v)
            v_label_file_abs_path = os.path.join(label_folder, v + '.txt')
            labels = self.read_labels(v_label_file_abs_path)
            images = os.listdir(v_abs_path)

            assert len(labels) == len(images)
            for image in images:
                image_index = int(image.split('.')[0])
                self.imgs.append(os.path.join(v_abs_path, image))
                self.labels.append(labels[image_index])
        self.transform = self.get_transform()

        print('FramewiseDataset: Load dataset {} with {} images.'.format(self.dataset, self.__len__()))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        img, label, img_path = self.transform(default_loader(self.imgs[item])), self.labels[item], self.imgs[item]
        return img, label, img_path

    def get_transform(self):
        return transforms.Compose([
                transforms.Resize((299,299)),
                transforms.ToTensor()
        ])

    def read_labels(self, label_file):
        with open(label_file,'r') as f:
            phases = [line.strip().split('\t')[1] for line in f.readlines()]
            labels = phase2label(phases, phase2label_dicts[self.dataset])
        return labels

class VideoDataset(Dataset):
    def __init__(self, dataset, root):
        self.dataset = dataset
        self.videos = []
        self.labels = []
        self.video_names = []

        video_feature_folder = os.path.join(root, 'video_feature_folder')
        label_folder = os.path.join(root, 'annotation_folder')
        for v_f in os.listdir(video_feature_folder):
            v_f_abs_path = os.path.join(video_feature_folder, v_f)
            v_label_file_abs_path = os.path.join(label_folder, v_f.split('.')[0] + '.txt')
            labels = self.read_labels(v_label_file_abs_path)

            self.videos.append(np.load(v_f_abs_path))
            self.labels.append(labels)
            self.video_names.append(v_f)

        print('VideoDataset: Load dataset {} with {} images.'.format(self.dataset, self.__len__()))

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, item):
        video, label, video_name = self.videos[item], self.labels[item], self.video_names[item]
        return video, label, video_name

    def read_labels(self, label_file):
        with open(label_file,'r') as f:
            phases = [line.strip().split('\t')[1] for line in f.readlines()]
            labels = phase2label(phases, phase2label_dicts[self.dataset])
        return labels

if __name__ == '__main__':
    '''
        UNIT TEST
    '''
    framewisedataset_cholec80 = FramewiseDataset('cholec80','cholec80/train_dataset')
    framewisedataloader_cholec80 = DataLoader(framewisedataset_cholec80, batch_size=64, shuffle=True, drop_last=False)

    videodataset_cholec80 = VideoDataset('cholec80', 'cholec80/train_dataset')
    videodataloader_cholec80 = DataLoader(videodataset_cholec80, batch_size=1, shuffle=True, drop_last=False)
