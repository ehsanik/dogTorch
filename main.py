"""
=================
Command line argument parser and loading the models.
=================
"""

import argparse
import datasets
import datetime
import logging
import models
import pprint
import os
import random
import sys
import torch
import pdb

from torch.autograd import Variable
from training import solver


def model_class(class_name):
    if class_name not in models.__all__:
        raise argparse.ArgumentTypeError("Invalid model {}; choices: {}".format(
            class_name, models.__all__))
    return getattr(models, class_name)


def dataset_class(class_name):
    if class_name not in datasets.__all__:
        raise argparse.ArgumentTypeError(
            "Invalid dataset {}; choices: {}".format(class_name,
                                                     datasets.__all__))
    return getattr(datasets, class_name)


def setup_logging(filepath, verbose):
    logFormatter = logging.Formatter(
        '%(levelname)s %(asctime)-20s:\t %(message)s')
    rootLogger = logging.getLogger()
    if verbose:
        rootLogger.setLevel(logging.DEBUG)
    else:
        rootLogger.setLevel(logging.INFO)
    logging.getLogger('PIL').setLevel(logging.WARNING)

    # Setup the logger to write into file
    fileHandler = logging.FileHandler(filepath)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    # Setup the logger to write into stdout
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)


def get_non_default_flags_str(args, parser, *ignore):
    flags = []
    for key, val in sorted(vars(args).items()):
        if key in ignore:
            continue
        if isinstance(val, type):
            val = val.__name__
        if val != parser.get_default(key):
            flags.append(key + '-' + str(val).replace(' ', '#'))
    return '+'.join(flags)


def parse_args():
    parser = argparse.ArgumentParser(description='Dog project training script')
    parser.add_argument(
        'mode', default='train', nargs='?',
        choices=('train', 'test', 'save_feats', 'perplexity',
                 'nearest_neighbor'))
    parser.add_argument('--data', metavar='DIR', default='data',
                        help='path to dataset')
    parser.add_argument('--save', metavar='DIR', default='cache',
                        help='path to cache directory')
    parser.add_argument('--dataset', default='DogClipDataset',
                        help='Dataset to '
                        'use for training/test.', type=dataset_class)
    parser.add_argument(
        '--arch', '-a', metavar='ARCH', default='AlexNetImage2IMU',
        help='model to use for training/test.', type=model_class)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('--verbose', action='store_true',
                        help='Level of logging the outputs')
    parser.add_argument('--epochs', default=90000, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--break-batch', default=1, type=int,
                        help='break batches with this factor to fit to memory.')
    parser.add_argument('--lrm', default=0.1, type=float, help='learning rate '
                        'multiplier.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-1, type=float,
                        metavar='W', help='weight decay (default: 1e-1)')
    parser.add_argument('--reload', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint')
    parser.add_argument('--no-strict', action='store_false', dest='strict',
                        help='Loading the weights from another model.')
    parser.add_argument('--no-pretrain', action='store_false', dest='pretrain',
                        help='Initialize the model with random intialization')
    parser.add_argument('--image_size', default=224, type=int,
                        help='Input image size')
    parser.add_argument(
        '--segmentation_size', default=56, type=int, help=
        'Segmentation size for fully convolutional network for walkable surface estimation'
    )
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--trainset_image_list', default='train.json',
                        help='Train dataset annotation file')
    parser.add_argument('--testset_image_list', default='test.json',
                        help='Test dataset annotation file')
    parser.add_argument('--valset_image_list', default='val.json',
                        help='Validation dataset annotation file')
    parser.add_argument('--num_classes', default=8, type=int,
                        help='Number of classes per IMU')
    parser.add_argument('--imus', nargs='+', default=list(range(6)), type=int,
                        help='List of IMUs to train')
    parser.add_argument('--features_dir', default='data/features', 
                        help='Address to read image features for LSTM networks')
    parser.add_argument(
        '--read_features', action='store_true',
        help='Indicate whether read features or use original image')
    parser.add_argument('--read_feature_and_image', action='store_true')
    parser.add_argument('--use_test_for_val', action='store_true',
                        help='Use this option to do final evaluation')
    parser.add_argument('--no_angle_metric', action='store_true',
                        help='Skip angular metric for faster training')
    parser.add_argument(
        '--regression', action='store_true',
        help='Regressing the IMU values instead of classification')
    parser.add_argument(
        '--end2end', action='store_true',
        help='End to end training image feature learning and IMU prediction')
    parser.add_argument(
        '--absolute_regress', action='store_true',
        help='Regressing the absolute imu values instead of difference')
    parser.add_argument('--single_image_feature', action='store_true',
                        help='Use single image feature')
    parser.add_argument('--save_qualitative', action='store_true')
    parser.add_argument(
        '--detach_resnet_end2end', action='store_true', help=
        'In the end2end network stop backpropagating through feature learning network'
    )
    parser.add_argument('--attention', dest='use_attention',
                        action='store_true',
                        help='Use attention network for LSTM')
    parser.add_argument('--experiment_type', default='imu2imu')
    parser.add_argument('--input_length', default=0, type=int,
                        help='Length of the input sequence')
    parser.add_argument('--output_length', default=1, type=int,
                        help='Length of the output sequence')
    parser.add_argument('--sequence_length', default=1, type=int,
                        help='Length of the sequence involved in training')
    parser.add_argument('--image_feature', default=1024, type=int,
                        help='Size of Image features')
    parser.add_argument('--hidden_size', default=512, type=int,
                        help='Size of hidden layers in LSTM')
    parser.add_argument('--num_layers', default=1, type=int,
                        help='Number of layers of LSTM')
    parser.add_argument('--step_size', default=10, type=int,
                        help='Step size for reducing the learning rate')
    parser.add_argument('--teacher_forcing', default=0, type=float,
                        help='Ratio of teacher forcing')
    parser.add_argument('--dropout_ratio', default=0.5, type=float)
    parser.add_argument(
        '--planning_distance', default=3, type=int,
        help='Indicates the length of the predicting sequence in Planning network'
    )
    parser.add_argument('--save_frequency', default=1, type=int,
                        help='Frequency of saving the model, per epoch')
    parser.add_argument(
        '--detach_level', default=5, type=int,
        help='Indicates how far in FCN network the output will be backpropagated'
    )  #5 means detach everything 1 means detach nothing #Just for the FCN network

    args = parser.parse_args()

    args.imu_feature = len(args.imus) * args.num_classes
    assert args.batch_size % args.break_batch == 0, "--batch-size must be "\
        "divisible by --break-batch."
    if args.absolute_regress:
        assert args.regression, "Regression must also be true"

    # Make log directory
    timestamp = str(datetime.datetime.now()).replace(' ', '#').replace(':', '.')
    args.save = os.path.join(
        args.save, args.arch.__name__,
        get_non_default_flags_str(args, parser, 'data', 'save', 'arch',
                                  'reload'), timestamp)
    os.makedirs(args.save, exist_ok=True)
    setup_logging(os.path.join(args.save, 'log.txt'), args.verbose)

    logging.info('Command: {}'.format(' '.join(sys.argv)))
    logging.info('Command line arguments parsed: {}'.format(
        pprint.pformat(vars(args))))

    return args


def get_data_loaders(args):
    train_dataset = args.dataset(args, train=True)
    val_dataset = args.dataset(args, train=False)
    # Do not shuffle dataset in save_feats mode to get consistent order of
    # inputs for saving features.
    shuffle = (args.mode != 'save_feats')
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size // args.break_batch,
        shuffle=shuffle, num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size // args.break_batch,
        shuffle=shuffle, num_workers=args.workers, pin_memory=True)
    args.train_loader = train_loader
    return train_loader, val_loader


def get_model_and_loss(args):
    model = args.arch(args).cuda()
    if args.reload is not None:
        model.load_state_dict(torch.load(args.reload), strict=args.strict)
    loss = model.loss().cuda()
    logging.info('Model: {}'.format(model))
    logging.info('Loss: {}'.format(loss))
    return model, loss


def main():
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    logging.info('Reading dataset metadata')
    train_loader, val_loader = get_data_loaders(args)
    args.relative_centroids = train_loader.dataset.get_relative_centroids()

    logging.info('Constructing model')
    model, loss = get_model_and_loss(args)

    if args.mode == 'train':
        optimizer = model.optimizer()
        for i in range(args.epochs):
            solver.train_one_epoch(model, loss, optimizer, train_loader, i + 1,
                                   args)
            solver.test_one_epoch(model, loss, val_loader, i + 1, args)
            if i % args.save_frequency == 0:

                torch.save(
                    model.state_dict(),
                    os.path.join(args.save,
                                 'model_state_{:02d}.pytar'.format(i + 1)))
    elif args.mode == 'test':
        solver.test_one_epoch(model, loss, val_loader, 0, args)
    elif args.mode == 'save_feats':
        solver.save_features(model, [train_loader, val_loader], args)
    elif args.mode == 'perplexity':
        solver.perplexity(model, val_loader, args)
    elif args.mode == 'nearest_neighbor':
        solver.nearest_neighbor(train_loader, val_loader, args)
    else:
        raise NotImplementedError("Unsupported mode {}".format(args.mode))


if __name__ == '__main__':
    main()
