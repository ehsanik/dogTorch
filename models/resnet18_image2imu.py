"""
=================
This model is a non-recurrent model for Inferring the action of the dog using a Resnet-18 network. Given two images, infer the imu changes corresponding to those two frames. 
=================
"""

import torch
import torch.nn as nn
import pdb

from torchvision.models import resnet18 as torchvision_resnet18
from extensions.multi_label_cross_entropy import MultiLabelCrossEntropyLoss
from training import metrics
from .basemodel import BaseModel


class ResNet18Image2IMU(BaseModel):
    metric = [
        metrics.AllAtOnce, metrics.AngleEvaluationMetric,
        metrics.SequenceMultiClassMetric
    ]

    def __init__(self, args):
        super(ResNet18Image2IMU, self).__init__()
        assert args.sequence_length == 2, "ResNet18Image2IMU supports seq-len=2"
        assert args.input_length == 2, "input length not currect"
        assert args.output_length == 1, "output length not currect"

        self.class_weights = args.dataset.CLASS_WEIGHTS[torch.LongTensor(
            args.imus)]
        self.output_length = args.output_length
        self.lr = args.lrm
        self.step_size = args.step_size
        self.decay = args.weight_decay

        resnet_model = torchvision_resnet18(pretrained=args.pretrain)
        # Remove the last fully connected layer.
        del resnet_model.fc
        self.resnet = resnet_model

        num_features = 512
        num_frames = 2
        num_classes = args.num_classes
        # Make num_imu fc layers
        self.imus = args.imus
        for i in self.imus:
            setattr(self, 'imu{}'.format(i),
                    nn.Linear(num_frames * num_features, num_classes))
        self.dropout = nn.Dropout(p=args.dropout_ratio)

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

    def feats(self, x):
        frame1 = x[:, :3, :, :]
        frame2 = x[:, 3:, :, :]
        return torch.cat(
            [self.resnet_features(frame1),
             self.resnet_features(frame2)], 1)

    def forward(self, input, target):
        features = self.feats(input)
        output_indices = list(
            range(target.size(1) - self.output_length, target.size(1)))
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
        base_lr = self.lr
        decay_rate = self.decay
        step = self.step_size
        assert 1 <= epoch
        if 1 <= epoch <= step:
            return base_lr
        elif step <= epoch <= step * 2:
            return base_lr * decay_rate
        elif step * 2 <= epoch <= step * 3:
            return base_lr * decay_rate * decay_rate
        else:
            return base_lr * decay_rate * decay_rate * decay_rate
