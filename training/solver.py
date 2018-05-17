"""
=================
Scripts for training and testing the network 
=================
"""

import logging
import math
import os
import time
import torch
import pdb
import matplotlib.pyplot as plt
import numpy as np
import json

from multiprocessing import pool
from torch.autograd import Variable
from training import metrics


def train_one_epoch(model, loss, optimizer, data_loader, epoch, args):
    # Prepare model and optimizer
    model.train()
    loss.train()
    lr = model.learning_rate(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # Setup average meters
    data_time_meter = metrics.AverageMeter()
    batch_time_meter = metrics.AverageMeter()
    loss_meter = metrics.AverageMeter()
    accuracy_metric = [m(args) for m in model.metric]

    stat_all_five = {}
    stat_each_indiv = {}

    # Iterate over data
    timestamp = time.time()

    all_joints = {}
    all_sequences = {}

    for i, (input, target, prev_absolutes, next_absolutes,
            file_names) in enumerate(data_loader):

        # Move data to gpu
        batch_size = input.size(0)
        input = Variable(input.cuda(async=True))
        target = Variable(target.cuda(async=True))
        prev_absolutes = prev_absolutes.cuda(async=True)
        next_absolutes = next_absolutes.cuda(async=True)
        data_time_meter.update(time.time() - timestamp, batch_size)

        # Forward pass
        output, target_output, output_indices = model(input, target)

        try:
            output_indices = output_indices.cuda()
        except Exception:
            output_indices = output_indices
        target = target[:, output_indices]

        loss_output = loss(output, target)

        # Backward pass and update weights
        loss_output.backward()

        if i % args.break_batch == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Bookkeeping on loss, accuracy, and batch time
        loss_meter.update(loss_output.data[0], batch_size)
        for acc in accuracy_metric:
            acc.record_output(output.data, output_indices, target.data,
                              prev_absolutes, next_absolutes, batch_size)
        batch_time_meter.update(time.time() - timestamp)

        # Log report
        logging.info('Epoch: [{}][{}/{}]\t'.format(
            epoch, (i // args.break_batch) +
            1, math.ceil(len(data_loader) / args.break_batch)) +
                     'Time {batch_time.val:.2f} ({batch_time.avg:.2f})   '
                     'Data {data_time.val:.2f} ({data_time.avg:.2f})   '
                     'Loss {loss.val:.6f} ({loss.avg:.6f})   '
                     '{accuracy_report}   LR {lr}'.format(
                         batch_time=batch_time_meter, data_time=data_time_meter,
                         loss=loss_meter, accuracy_report='\t'.join(
                             [ac.report() for ac in accuracy_metric]), lr=lr))
        timestamp = time.time()

    training_summary = (
        'Epoch: [{}] -- TRAINING SUMMARY\t'.format(epoch) +
        'Time {batch_time.sum:.2f}   Data {data_time.sum:.2f}   '
        'Loss {loss.avg:.6f}   {accuracy_report}'.format(
            batch_time=batch_time_meter, data_time=data_time_meter,
            loss=loss_meter, accuracy_report='\n'.join(
                [ac.final_report() for ac in accuracy_metric])))
    logging.info(training_summary)
    logging.info('Full train result is at {}'.format(
        os.path.join(args.save, 'train.log')))
    with open(os.path.join(args.save, 'train.log'), 'a') as fp:
        fp.write('{}\n'.format(training_summary))


def test_one_epoch(model, loss, data_loader, epoch, args):
    # Prepare model and loss
    model.eval()
    loss.eval()

    # Setup average meters
    data_time_meter = metrics.AverageMeter()
    batch_time_meter = metrics.AverageMeter()
    loss_meter = metrics.AverageMeter()
    accuracy_metric = [m(args) for m in model.metric]

    data_for_visualization = {}

    # Iterate over data
    timestamp = time.time()
    for i, (input, target, prev_absolutes, next_absolutes,
            current_images_files) in enumerate(data_loader):
        # Move data to gpu
        batch_size = input.size(0)
        input = Variable(input.cuda(async=True), volatile=True)
        target = Variable(target.cuda(async=True), volatile=True)
        prev_absolutes = prev_absolutes.cuda(async=True)
        next_absolutes = next_absolutes.cuda(async=True)
        data_time_meter.update(time.time() - timestamp)

        # Forward pass
        output, target_output, output_indices = model(input, target)

        try:
            output_indices = output_indices.cuda()
        except Exception:
            output_indices = output_indices
        target = target[:, output_indices]

        loss_output = loss(output, target)

        # Bookkeeping on loss, accuracy, and batch time
        loss_meter.update(loss_output.data[0], batch_size)
        for ac in accuracy_metric:
            if type(ac) == metrics.AngleEvaluationMetric:
                absolutes_predict = ac.record_output(
                    output.data, output_indices, target.data, prev_absolutes,
                    next_absolutes, batch_size)
            else:
                ac.record_output(output.data, output_indices, target.data,
                                 prev_absolutes, next_absolutes, batch_size)

        if args.save_qualitative:
            if not (args.regression):
                converted_output = torch.max(output, -1)[1]
            else:
                converted_output = output.clone()

        batch_time_meter.update(time.time() - timestamp)
        # Log report
        logging.info('Test Epoch: [{}][{}/{}]\t'.format(
            epoch, (i // args.break_batch) +
            1, math.ceil(len(data_loader) / args.break_batch)) +
                     'Time {batch_time.val:.2f} ({batch_time.avg:.2f})   '
                     'Data {data_time.val:.2f} ({data_time.avg:.2f})   '
                     'Loss {loss.val:.6f} ({loss.avg:.6f})   '
                     '{accuracy_report}'.format(
                         batch_time=batch_time_meter, data_time=data_time_meter,
                         loss=loss_meter, accuracy_report='\t'.join(
                             [ac.report() for ac in accuracy_metric])))
        timestamp = time.time()

    testing_summary = (
        'Epoch: [{}] -- TESTING SUMMARY\t'.format(epoch) +
        'Time {batch_time.sum:.2f}   Data {data_time.sum:.2f}   '
        'Loss {loss.avg:.6f}   {accuracy_report}'.format(
            batch_time=batch_time_meter, data_time=data_time_meter,
            loss=loss_meter, accuracy_report='\n'.join(
                [ac.final_report() for ac in accuracy_metric])))
    logging.info(testing_summary)
    logging.info('Full test result is at {}'.format(
        os.path.join(args.save, 'test.log')))
    with open(os.path.join(args.save, 'test.log'), 'a') as fp:
        fp.write('{}\n'.format(testing_summary))


def _save_tensor(tensor_path_pair):
    tensor, path = tensor_path_pair
    logging.debug('Saving feature to {}.'.format(path))
    torch.save(tensor, path)


def save_features(model, data_loaders, args):
    model.eval()
    os.makedirs(args.features_dir, exist_ok=True)
    thread_pool = pool.ThreadPool(args.workers)
    for data_loader in data_loaders:
        data_index = 0
        for input, target, prev_absolutes, next_absolutes, _ in data_loader:
            input = Variable(input.cuda(async=True), volatile=True)
            features = model.feats(input).data.cpu()
            features_to_save = []
            for feature in features:
                relpath = data_loader.dataset.get_relpath(data_index)
                feature_path = os.path.join(args.features_dir,
                                            relpath + '.pytar')
                features_to_save.append((feature, feature_path))
                data_index += 1
            thread_pool.map(_save_tensor, features_to_save)


def perplexity(model, data_loader, args):
    model.eval()
    # Setup average meters
    data_time_meter = metrics.AverageMeter()
    batch_time_meter = metrics.AverageMeter()
    perplexity_meter = metrics.AverageMeter()

    timestamp = time.time()
    for i, (input, target, prev_absolutes, next_absolutes,
            _) in enumerate(data_loader):
        batch_size = input.size(0)
        input = Variable(input.cuda(async=True), volatile=True)
        target = Variable(target.cuda(async=True), volatile=True)
        data_time_meter.update(time.time() - timestamp)

        perplexity = model.perplexity(input, target).data
        perplexity_meter.update(perplexity, batch_size)

        batch_time_meter.update(time.time() - timestamp)
        # Log report
        logging.info(
            'Perplexity: [{}/{}]\t'.format(
                (i // args.break_batch) +
                1, math.ceil(len(data_loader) / args.break_batch)) +
            'Time {batch_time.val:.2f} ({batch_time.avg:.2f})   '
            'Data {data_time.val:.2f} ({data_time.avg:.2f})   '
            'Perplexity [{cur_perplexity}] ([{avg_perplexity}])'.format(
                batch_time=batch_time_meter, data_time=data_time_meter,
                cur_perplexity=', '.join(
                    '{:.2f}'.format(val)
                    for val in perplexity_meter.val), avg_perplexity=', '.join(
                        '{:.2f}'.format(avg) for avg in perplexity_meter.avg)))
    logging.info(
        'Final Perplexity: \tTime {batch_time.sum:.2f}   '
        'Data {data_time.sum:.2f}   Perplexity [{perplexity}] Average Perplexity {avg_perplexity}'.
        format(
            batch_time=batch_time_meter, data_time=data_time_meter,
            perplexity=', '.join(
                '{:.2f}'.format(avg) for avg in perplexity_meter.avg),
            avg_perplexity=perplexity_meter.avg.mean()))
