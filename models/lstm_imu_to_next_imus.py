"""
=================
This model is a Recurrent model for Acting like a dog. Given a sequence of images, predict a sequence of joint movements of the dog for the next time frames. 
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


class LstmImu2NextImus(BaseModel):
    metric = [metrics.SequenceMultiClassMetric, metrics.AngleEvaluationMetric]

    def __init__(self, args):
        super(LstmImu2NextImus, self).__init__()

        assert (args.sequence_length <= args.input_length + args.output_length)

        self.input_length = args.input_length
        self.output_length = args.output_length
        self.imu_feature = args.imu_feature
        self.num_classes = args.num_classes
        self.imus = args.imus
        self.class_weights = args.dataset.CLASS_WEIGHTS[torch.LongTensor(
            args.imus)]

        self.embedding_input = nn.Linear(args.imu_feature, args.hidden_size)
        self.lstm = Lstm(args)

    def forward(self, input, target):
        input = target[:, :self.input_length]
        output_indices = list(
            range(target.size(1) - self.output_length, target.size(1)))
        target = target[:, -self.output_length:]

        input = self.convert_imus(input)

        input = input.transpose(0, 1)
        embedded_input = self.embedding_input(input)
        full_output = self.lstm(embedded_input, None)
        return full_output.transpose(
            0, 1), target, torch.LongTensor(output_indices)

    def loss(self):
        return MultiLabelCrossEntropyLoss(self.class_weights)

    def optimizer(self):
        return optim.Adam(self.parameters(), lr=0.001)

    def convert_imus(self, input):
        size = input.size()
        num_imus = len(self.imus)
        converted = torch.FloatTensor(size[0], size[1], num_imus,
                                      self.num_classes).cuda()
        converted.zero_()
        converted.scatter_(3, input.data.unsqueeze(3), 1.0)
        converted = Variable(converted).view(size[0], size[1], self.imu_feature)
        return converted

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
