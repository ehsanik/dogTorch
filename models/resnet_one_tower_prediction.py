"""
=================
This model is a non-recurrent model for Acting like a dog. Given a sequence of images, predict a sequence of joint movements of the dog for the next time frames. 
=================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18 as torchvision_resnet18
from extensions.multi_label_cross_entropy import MultiLabelCrossEntropyLoss
from training import metrics
from .basemodel import BaseModel
import pdb


class ResNet18Image2IMUOneTowerPrediction(BaseModel):
    metric = [
        metrics.SequenceMultiClassMetric, metrics.AngleEvaluationMetric,
        metrics.AllAtOnce
    ]

    def __init__(self, args):
        super(ResNet18Image2IMUOneTowerPrediction, self).__init__()

        self.class_weights = args.dataset.CLASS_WEIGHTS[torch.LongTensor(
            args.imus)]

        resnet_model = torchvision_resnet18(pretrained=args.pretrain)
        # Remove the last fully connected layer.
        del resnet_model.fc
        self.resnet = resnet_model

        num_features = 512
        num_frames = args.input_length
        self.args = args

        self.resnet.conv1 = nn.Conv2d(num_frames * 3, 64, kernel_size=(7, 7),
                                      stride=(2, 2), padding=(3, 3), bias=False)

        num_classes = args.num_classes
        # Make num_imu fc layers
        self.imus = args.imus
        for i in self.imus:
            setattr(self, 'imu{}'.format(i), nn.Linear(num_features,
                                                       num_classes))
        self.dropout = nn.Dropout()

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

        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return x

    def forward(self, input, target):

        input = input[:, :self.args.input_length * 3]
        target = target[:, -self.args.output_length:]

        features = self.resnet_features(input)
        output_indices = list(
            range(self.args.sequence_length - self.args.output_length,
                  self.args.sequence_length))
        # Iterate over fully connecteds for each imu, perform forward pass and
        # record the output.

        all_output = []
        for imu_id in range(self.args.output_length):
            imu_out = []
            for i in self.imus:
                imu_i = getattr(self, 'imu{}'.format(i))
                imu_out.append(imu_i(features))
            output = torch.stack(imu_out, dim=1).unsqueeze(1)
            all_output.append(output)
        # Add a singleton dim at 1 for sequence length, which is always 1 in
        # this model.
        all_output = torch.cat(all_output, dim=1)
        return all_output, target, torch.LongTensor(output_indices)

    def loss(self):
        return MultiLabelCrossEntropyLoss(self.class_weights)

    def perplexity(self, input, target):
        output, target, _ = self.forward(input, target)
        output = F.softmax(output.transpose(1, 3)).transpose(1, 3)
        probabilities = output.gather(3, target.unsqueeze(3)).squeeze(3)
        perplexity = probabilities.log().mean(1).exp().mean(0)
        return 100 * perplexity

    def learning_rate(self, epoch):
        assert 1 <= epoch
        if 1 <= epoch <= 30:
            return 0.01
        elif 31 <= epoch <= 60:
            return 0.001
        elif 61 <= epoch <= 90:
            return 0.0001
        else:
            return 0.00001
