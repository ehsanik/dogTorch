"""
=================
This model is a non-recurrent model for Inferring the action of the dog using a Resnet-18 network. Given a sequence of images (can be more than 2), infer the imu changes corresponding to those frames.
=================
"""

import torch
import torch.nn as nn

from torchvision.models import resnet18 as torchvision_resnet18
from extensions.multi_label_cross_entropy import MultiLabelCrossEntropyLoss
from training import metrics
from .basemodel import BaseModel
import pdb


class ResNet18Image2IMUOneTower(BaseModel):
    metric = [metrics.SequenceMultiClassMetric]

    def __init__(self, args):
        super(ResNet18Image2IMUOneTower, self).__init__()
        assert args.sequence_length == 1, "ResNet18Image2IMU supports seq-len=1"

        self.class_weights = args.dataset.CLASS_WEIGHTS[torch.LongTensor(
            args.imus)]

        resnet_model = torchvision_resnet18(pretrained=args.pretrain)
        # Remove the last fully connected layer.
        del resnet_model.fc
        self.resnet = resnet_model

        self.resnet.conv1 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2),
                                      padding=(3, 3), bias=False)

        num_features = 512
        num_frames = 1
        num_classes = args.num_classes
        # Make num_imu fc layers
        self.imus = args.imus
        for i in self.imus:
            setattr(self, 'imu{}'.format(i),
                    nn.Linear(num_frames * num_features, num_classes))
        self.dropout = nn.Dropout()

    def feats(self, x):
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
        features = self.feats(input)
        output_indices = list(range(0, (target.size(1))))
        # Iterate over fully connecteds for each imu, perform forward pass and
        # record the output.
        imu_out = []
        for i in self.imus:
            imu_i = getattr(self, 'imu{}'.format(i))
            imu_out.append(imu_i(features))
        # Add a singleton dim at 1 for sequence length, which is always 1 in
        # this model.
        return torch.stack(
            imu_out,
            dim=1).unsqueeze(1), target, torch.LongTensor(output_indices)

    def loss(self):
        return MultiLabelCrossEntropyLoss(self.class_weights)

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
