"""
=================
This model is a Fully Convolutional Network for predicting the walkable surfaces using a single 3 channel image.
=================
"""

import torch
import torch.nn as nn
import pdb

from torchvision.models import resnet18 as torchvision_resnet18
from extensions.weighted_binary_cross_entropy import WeightedCrossEntropy, SeGANLoss
from training import metrics
from .basemodel import BaseModel
from torch.nn import init
from torchvision.models.resnet import BasicBlock, ResNet


# This function is borrowed from https://github.com/Kaixhin/FCN-semantic-segmentation/blob/master/model.py
def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, bias=False,
         transposed=False):
    """
    Returns 2D convolutional layer with space-preserving padding
    """
    if transposed:
        layer = nn.ConvTranspose2d(
            in_planes, out_planes, kernel_size=kernel_size, stride=stride,
            padding=1, output_padding=1, dilation=dilation, bias=bias)
        # Bilinear interpolation init
        w = torch.Tensor(kernel_size, kernel_size)
        centre = kernel_size % 2 == 1 and stride - 1 or stride - 0.5
        for y in range(kernel_size):
            for x in range(kernel_size):
                w[y, x] = (1 - abs((x - centre) / stride)) * (1 - abs(
                    (y - centre) / stride))
        layer.weight.data.copy_(
            w.div(in_planes).repeat(in_planes, out_planes, 1, 1))
    else:
        padding = (kernel_size + 2 * (dilation - 1)) // 2
        layer = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                          stride=stride, padding=padding, dilation=dilation,
                          bias=bias)
    if bias:
        init.constant(layer.bias, 0)
    return layer


class FullyConvolutional(BaseModel):
    metric = [metrics.IouSegmentation]

    def __init__(self, args):
        super(FullyConvolutional, self).__init__()

        self.class_weights = args.dataset.CLASS_WEIGHTS

        self.args = args
        self.lr = args.lrm
        self.decay = args.weight_decay
        self.step_size = args.step_size

        resnet_model = torchvision_resnet18(pretrained=args.pretrain)
        # Remove the last fully connected layer.
        del resnet_model.fc
        self.resnet = resnet_model

        num_features = 512
        num_frames = 2

        self.imus = args.imus
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=args.dropout_ratio)
        self.conv4 = conv(512, 256, stride=2, transposed=True, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv3 = conv(256, 128, stride=2, transposed=True, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv2 = conv(128, 64, stride=2, transposed=True, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv1 = conv(64, 1, kernel_size=3)

    def resnet_features(self, x):

        detach_level = self.args.detach_level

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x1 = self.resnet.layer1(x)
        if detach_level > 1:
            x1.detach_()
        x2 = self.resnet.layer2(x1)
        if detach_level > 2:
            x2.detach_()
        x3 = self.resnet.layer3(x2)
        if detach_level > 3:
            x3.detach_()
        x4 = self.resnet.layer4(x3)
        if detach_level > 4:
            x4.detach_()

        return x1, x2, x3, x4

    def forward(self, input, target):
        x1, x2, x3, x4 = self.resnet_features(input)
        output_indices = list(range(0, (target.size(1))))
        x = self.bn4(self.relu(self.conv4(x4)))
        x = self.bn3(self.relu(self.conv3(x3 + x)))
        x = self.bn2(self.relu(self.conv2(x2 + x)))
        x = self.conv1(x1 + x)
        x = self.dropout(x)
        output = x
        return output, target, torch.LongTensor(output_indices)

    def loss(self):
        return SeGANLoss(self.class_weights.cuda())

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
