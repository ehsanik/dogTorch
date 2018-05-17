import json
import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
import pdb
import skimage.io as io
import h5py
import matplotlib.pyplot as plt

from PIL import Image


def parse_file(dataset_adr, categories):
    dataset = []
    with open(dataset_adr) as f:
        for line in f:
            line = line[:-1].split('/')
            category = '/'.join(line[2:-1])
            file_name = '/'.join(line[2:])
            if not category in categories:
                continue
            dataset.append([file_name, category])
    return dataset


def get_class_names(path):
    classes = []
    with open(path) as f:
        for line in f:
            categ = '/'.join(line[:-1].split('/')[2:])
            classes.append(categ)
    class_dic = {classes[i]: i for i in range(len(classes))}
    return class_dic


class SunDataset(data.Dataset):

    CLASS_WEIGHTS = None

    def __init__(self, args, train=True):
        self.root_dir = args.data
        root_dir = self.root_dir
        if train:
            self.data_set_list = os.path.join(root_dir,
                                              args.trainset_image_list)
        else:
            self.data_set_list = os.path.join(root_dir, args.testset_image_list)

        self.categ_dict = get_class_names(
            os.path.join(root_dir, 'ClassName.txt'))

        self.data_set_list = parse_file(self.data_set_list, self.categ_dict)

        self.args = args
        self.read_features = args.read_features

        self.features_dir = args.features_dir
        if train:
            self.transform = transforms.Compose([
                transforms.RandomSizedCrop(args.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.Scale((args.image_size, args.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Scale((args.image_size, args.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

    def get_relative_centroids(self):
        return None

    def __len__(self):
        return len(self.data_set_list)

    def load_and_resize(self, img_name):
        with open(img_name, 'rb') as fp:
            image = Image.open(fp).convert('RGB')
        return self.transform(image)

    def __getitem__(self, idx):
        file_name, categ = self.data_set_list[idx]
        try:
            image = self.load_and_resize(
                os.path.join(self.root_dir, 'all_data', file_name + '~'))
        except Exception:
            image = self.load_and_resize(
                os.path.join(self.root_dir, 'all_data', file_name))
        if not categ in self.categ_dict:
            pdb.set_trace()
        label = self.categ_dict[categ]
        label = torch.Tensor([label]).long()
        return (image, label, 0, 0, [file_name])
