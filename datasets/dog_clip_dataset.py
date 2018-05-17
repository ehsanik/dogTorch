import json
import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random

from PIL import Image
import pdb


def _read_labels(json_file, imus, sequence_length):
    """Returns a list of all frames, and a list of where each data point (whose
    length is sequence_length) in the list of frames."""
    with open(json_file, 'r') as fp:
        dataset_meta = json.load(fp)
    frames = []
    idx_to_fid = []
    centroids = {
        'absolute_centroids':
        torch.Tensor([dataset_meta['absolute_centroids'][imu] for imu in imus]),
        'difference_centroids':
        torch.Tensor(
            [dataset_meta['difference_centroids'][imu] for imu in imus]),
    }
    for clip in dataset_meta['clips']:
        frame_clips = [{
            'cur_frame':
            frame_meta['filename'],
            'prev_frame':
            frame_meta['prev-frame'],
            'labels':
            torch.LongTensor(
                [frame_meta['imu-diff-clusters'][imu] for imu in imus]),
            'diffs':
            torch.FloatTensor(
                [frame_meta['imu-diff-values'][imu] for imu in imus]),
            'absolute_cur_imus':
            torch.FloatTensor(
                [frame_meta['absolute_cur_imus'][imu] for imu in imus]),
            'absolute_prev_imus':
            torch.FloatTensor(
                [frame_meta['absolute_prev_imus'][imu] for imu in imus]),
        } for frame_meta in clip['frames']]
        for i in range(len(frame_clips) - sequence_length + 1):
            idx_to_fid.append(i + len(frames))
        frames += frame_clips
    return frames, idx_to_fid, centroids


def _category_weights():

    category_sizes = torch.Tensor(
        [[131., 199., 177., 157., 3446., 1689., 14838.,
          186.], [379., 366., 1705., 1297., 9746., 873., 1475.,
                  4982.], [232., 257., 241., 3422., 126., 11225., 5105., 215.],
         [137., 115., 142., 3192., 3066., 10036., 3983.,
          152.], [781., 594., 1183., 4753., 664., 9038., 1394.,
                  2416.], [140., 214., 150., 4644., 11075., 1974., 182., 2444.],
         [169., 133., 150., 104., 2090., 12701., 2157., 3319.]])

    weight = category_sizes.sum(1, keepdim=True) / category_sizes
    return weight


class DogClipDataset(data.Dataset):
    CLASS_WEIGHTS = _category_weights()

    def __init__(self, args, train=True):
        root_dir = args.data
        if train or args.read_feature_and_image:
            json_file = os.path.join(root_dir, args.trainset_image_list)
        elif args.use_test_for_val:
            json_file = os.path.join(root_dir, args.testset_image_list)
        else:
            json_file = os.path.join(root_dir, args.valset_image_list)

        self.num_classes = args.num_classes
        self.sequence_length = args.sequence_length
        self.experiment_type = args.experiment_type
        self.regression = args.regression
        self.end2end = args.end2end
        self.absolute_regress = args.absolute_regress
        self.args = args
        self.single_image_feature = args.single_image_feature

        self.read_features = args.read_features
        self.frames_metadata, self.idx_to_fid, self.centroids = _read_labels(
            json_file, args.imus, args.sequence_length)

        self.root_dir = root_dir
        self.features_dir = args.features_dir
        self.transform = transforms.Compose([
            transforms.Scale((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def get_relative_centroids(self):
        return self.centroids['difference_centroids']

    def __len__(self):
        return len(self.idx_to_fid)

    def load_and_resize(self, img_name):
        with open(img_name, 'rb') as fp:
            image = Image.open(fp).convert('RGB')
        return self.transform(image)

    def get_relpath(self, idx):
        fid = self.idx_to_fid[idx]
        return self.frames_metadata[fid]['cur_frame']

    def __getitem__(self, idx):
        current_images_files = []
        fid = self.idx_to_fid[idx]
        if self.read_features:
            features = []
            for i in range(self.sequence_length):
                feature_path = os.path.join(
                    self.features_dir,
                    self.frames_metadata[fid + i]['cur_frame'] + '.pytar')
                feature = torch.load(feature_path)
                if self.single_image_feature:
                    feature = feature[:feature.size(0) // 2]
                features.append(feature)
            input = torch.stack(features)

        elif not self.read_features:
            images = []
            for i in range(self.sequence_length):
                img = self.load_and_resize(
                    os.path.join(self.root_dir, 'images',
                                 self.frames_metadata[fid + i]['cur_frame']))
                images.append(img)
            if self.end2end:
                input = torch.stack(images, 0)
            elif self.regression:
                input = torch.cat(images)
            else:
                input = torch.cat(images)

        labels = []
        absolute_cur_imus = []
        absolute_prev_imus = []
        for i in range(self.sequence_length):
            if self.absolute_regress:
                labels.append(
                    self.frames_metadata[fid + i]['absolute_cur_imus'])
            elif self.regression:
                labels.append(self.frames_metadata[fid + i]['diffs'])
            else:
                labels.append(self.frames_metadata[fid + i]['labels'])
            absolute_cur_imus.append(
                self.frames_metadata[fid + i]['absolute_cur_imus'])
            absolute_prev_imus.append(
                self.frames_metadata[fid + i]['absolute_prev_imus'])
            current_images_files.append(
                self.frames_metadata[fid + i]['cur_frame'])

        labels = torch.stack(labels)
        absolute_cur_imus = torch.stack(absolute_cur_imus)
        absolute_prev_imus = torch.stack(absolute_prev_imus)
        if self.args.read_feature_and_image:
            features = []
            for i in range(self.sequence_length):
                image = self.load_and_resize(
                    os.path.join(self.root_dir, 'images',
                                 self.frames_metadata[fid + i]['cur_frame']))
                features.append(image)
            labels = torch.stack(features)
        return (input, labels, absolute_prev_imus, absolute_cur_imus,
                current_images_files)
