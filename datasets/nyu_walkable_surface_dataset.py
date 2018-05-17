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
from .walkable_nyu_datafile import train_set_list, test_set_list, val_set_list


def _category_weights():

    category_sizes = torch.Tensor([1.0 - 0.09, 0.09])
    weight = 1.0 / category_sizes
    return weight


class NyuDataset(data.Dataset):
    CLASS_WEIGHTS = _category_weights()

    def __init__(self, args, train=True):
        self.root_dir = args.data

        if train:
            self.data_set_list = train_set_list
        elif args.use_test_for_val:
            self.data_set_list = test_set_list
        else:
            self.data_set_list = val_set_list

        self.data_set_list = ['%06d.png' % (x) for x in self.data_set_list]
        self.args = args
        self.read_features = args.read_features

        self.features_dir = args.features_dir
        self.transform = transforms.Compose([
            transforms.Scale((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.transform_segmentation = transforms.Compose([
            transforms.Scale((args.segmentation_size, args.segmentation_size)),
            transforms.ToTensor(),
        ])

    def get_relative_centroids(self):
        return None

    def clean_mask(self, output):
        cleaned = output.clone()
        cleaned[output > 0.5] = 1.0
        cleaned[output < 0.5] = 0.0
        return cleaned

    def __len__(self):
        return len(self.data_set_list)

    def load_and_resize(self, img_name):
        with open(img_name, 'rb') as fp:
            image = Image.open(fp).convert('RGB')
        return self.transform(image)

    def load_and_resize_segmentation(self, img_name):
        with open(img_name, 'rb') as fp:
            image = Image.open(fp).convert('RGB')
        return self.clean_mask(self.transform_segmentation(image)[0:1])

    def __getitem__(self, idx):
        fid = self.data_set_list[idx]
        if self.read_features:
            features = []
            for i in range(self.sequence_length):
                feature_path = os.path.join(
                    self.features_dir,
                    self.frames_metadata[fid + i]['cur_frame'] + '.pytar')
                features.append(torch.load(feature_path))
            input = torch.stack(features)
        else:
            image = self.load_and_resize(
                os.path.join(self.root_dir, 'images', fid))
            segment = self.load_and_resize_segmentation(
                os.path.join(self.root_dir, 'walkable', fid))

        # The two 0s are just place holders. They can be replaced by any values
        return (image, segment, 0, 0, ['images' + fid])
