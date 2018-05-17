"""
=================
This model is designed for the scene categorization task. (Representation learning)
=================
"""

import torch
import torch.nn as nn
import pdb

from torchvision.models import resnet18 as torchvision_resnet18
from training import metrics
from .basemodel import BaseModel
from torch.nn import init
from torchvision.models.resnet import BasicBlock, ResNet


class ResNetSceneCategorization(BaseModel):
    metric = [metrics.ClassificationMetric]

    def __init__(self, args):
        super(ResNetSceneCategorization, self).__init__()

        self.class_weights = args.dataset.CLASS_WEIGHTS

        self.args = args
        self.dropout = nn.Dropout()

        resnet_model = torchvision_resnet18(pretrained=args.pretrain)
        # Remove the last fully connected layer.
        del resnet_model.fc

        self.resnet = resnet_model
        self.fc = nn.Linear(512, 397)

    def resnet_features(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)
        return x

    def forward(self, input, target):
        assert target.size(1) == 1
        output_indices = 0
        batch_size = input.size(0)
        resnet_output = self.resnet_features(input)
        resnet_output.detach_()
        output = self.fc(resnet_output.view(batch_size, -1))
        return output, target, output_indices

    def loss(self):
        return nn.CrossEntropyLoss()

    def learning_rate(self, epoch):
        assert 1 <= epoch
        lrm = 0.1
        step = 30
        if 1 <= epoch <= step:
            return lrm
        elif step + 1 <= epoch <= 2 * step:
            return lrm * 0.1
        elif 2 * step + 1 <= epoch <= 3 * step:
            return lrm * 0.01
        else:
            return lrm * 0.001
