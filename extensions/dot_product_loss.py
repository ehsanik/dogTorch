import torch.nn as nn
import torch
import pdb


class DotProductLoss(nn.Module):

    def __init__(self):
        super(DotProductLoss, self).__init__()

    def forward(self, output, target):
        return -torch.dot(target.view(-1), output.view(-1)) / target.nelement()
