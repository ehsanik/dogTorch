"""
=================
This model is a recurrent model for the planning task, given two non-consecutive frames, predict a sequence of actions taking the dog from the first frame to the second frame. 
=================
"""

import torch
import torch.nn as nn
import pdb
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from torchvision.models import alexnet as torchvision_alexnet
from extensions.multi_label_cross_entropy import MultiLabelCrossEntropyLoss
from .basemodel import BaseModel
from torch.autograd import Variable
from .lstm import Lstm
from training import metrics


def _target_to_hot_vectors(target, num_classes):
    hot_vectors = torch.cuda.FloatTensor(*target.size(), num_classes).zero_()
    hot_vectors.scatter_(2, target.data.unsqueeze(2), 1)
    hot_vectors = Variable(hot_vectors)
    return hot_vectors


class LstmImg2ActionPlanning(BaseModel):
    metric = [
        metrics.SequenceMultiClassMetric, metrics.AngleEvaluationMetric,
        metrics.AllAtOnce
    ]

    def __init__(self, args):
        super(LstmImg2ActionPlanning, self).__init__()

        assert args.planning_distance == args.output_length

        self.input_length = args.input_length
        self.output_length = args.output_length
        self.sequence_length = args.sequence_length
        self.class_weights = args.dataset.CLASS_WEIGHTS[torch.LongTensor(
            args.imus)]
        self.planning_distance = args.planning_distance
        self.imus = args.imus
        self.hidden_size = args.hidden_size
        self.num_classes = args.num_classes

        self.embedding_input = nn.Linear(args.image_feature * 2,
                                         args.hidden_size)
        self.embedding_target = nn.Linear(args.num_classes * len(args.imus),
                                          args.hidden_size)
        self.lstm = nn.LSTMCell(args.hidden_size, args.hidden_size,
                                args.num_layers)
        self.out = nn.Linear(args.hidden_size, args.imu_feature)

        for i in self.imus:
            setattr(self, 'imu{}'.format(i),
                    nn.Linear(args.imu_feature, args.num_classes))

    def forward(self, input, target):
        imu_start_index = 1  #inclusive
        imu_end_index = imu_start_index + self.planning_distance  #exclusive
        assert imu_start_index > 0 and imu_end_index < self.sequence_length
        input_start = input[:, :imu_start_index, :512]
        input_end = input[:, imu_end_index:, :512]
        target = target[:, imu_start_index:imu_end_index]
        output_indices = list(range(imu_start_index, imu_end_index))

        full_input = torch.cat([input_start, input_end], 1)
        full_input = full_input.view(full_input.size(0), -1)
        embedded_input = self.embedding_input(full_input)
        batch_size = embedded_input.size(0)
        hidden, cell = self.initHiddenCell(batch_size)
        lstm_outputs = []
        for output_num in range(self.planning_distance):

            hidden, cell = self.lstm(embedded_input, (hidden, cell))

            output = (self.out(hidden))
            imu_out = []
            for i in self.imus:
                imu_i = getattr(self, 'imu{}'.format(i))
                imu_out.append(imu_i(output))

            imu_soft = [
                F.softmax(imu).view(batch_size, 1, self.num_classes)
                for imu in imu_out
            ]
            imu_out_reshaped = [(imu).view(batch_size, 1, self.num_classes)
                                for imu in imu_out]
            cleaned_output = torch.cat(imu_soft, 1)
            converted_output = torch.cat(imu_out_reshaped, 1)
            lstm_outputs.append(converted_output)

            embedded_input = self.embedding_target(
                cleaned_output.view(batch_size, -1))

        full_output = torch.stack(lstm_outputs)
        return full_output.transpose(
            0, 1), target, torch.LongTensor(output_indices)

    def initHiddenCell(self, batch_size):
        result = Variable(torch.zeros(batch_size, self.hidden_size))
        cell = Variable(torch.zeros(batch_size, self.hidden_size))
        return result.cuda(), cell.cuda()

    def loss(self):
        return MultiLabelCrossEntropyLoss(self.class_weights)

    def optimizer(self):
        return optim.Adam(self.parameters(), lr=0.001)

    def perplexity(self, input, target):
        imu_start_index = 1  #inclusive
        imu_end_index = imu_start_index + self.planning_distance  #exclusive
        assert imu_start_index > 0 and imu_end_index < self.sequence_length
        input_start = input[:, :imu_start_index]
        input_end = input[:, imu_end_index:]
        target = target[:, imu_start_index:imu_end_index]
        output_indices = list(range(imu_start_index, imu_end_index))

        full_input = torch.cat([input_start, input_end], 1)
        full_input = full_input.view(full_input.size(0), -1)
        embedded_input = self.embedding_input(full_input)
        batch_size = embedded_input.size(0)
        hidden, cell = self.initHiddenCell(batch_size)
        lstm_outputs = []
        for output_num in range(self.planning_distance):
            hidden, cell = self.lstm(embedded_input, (hidden, cell))

            output = (self.out(hidden))
            imu_out = []
            for i in self.imus:
                imu_i = getattr(self, 'imu{}'.format(i))
                imu_out.append(imu_i(output))

            imu_soft = [
                F.softmax(imu).view(batch_size, 1, self.num_classes)
                for imu in imu_out
            ]
            imu_out_reshaped = [(imu).view(batch_size, 1, self.num_classes)
                                for imu in imu_out]
            cleaned_output = _target_to_hot_vectors(target[:, output_num],
                                                    self.num_classes)
            converted_output = torch.cat(imu_soft, 1)
            lstm_outputs.append(converted_output)

            embedded_input = self.embedding_target(
                cleaned_output.view(batch_size, -1))

        probabilities = torch.stack(lstm_outputs)
        probabilities = probabilities.transpose(0, 1)
        gt_probabilities = probabilities.gather(3,
                                                target.unsqueeze(3)).squeeze(3)
        gt_avg_log_probability = gt_probabilities.log().mean(1)
        perplexity = gt_avg_log_probability.exp().mean(0)
        return 100 * perplexity

    def learning_rate(self, epoch):
        base_lr = 0.001
        decay_rate = 0.1
        step = 90
        assert 1 <= epoch
        if 1 <= epoch <= step:
            return base_lr
        elif step <= epoch <= step * 2:
            return base_lr * decay_rate
        elif step * 2 <= epoch <= step * 3:
            return base_lr * decay_rate * decay_rate
        else:
            return base_lr * decay_rate * decay_rate * decay_rate
