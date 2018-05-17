import torch.nn as nn
import torch
import pdb


class WeightedCrossEntropy(nn.Module):

    def __init__(self, weights=None, size_average=True):
        super(WeightedCrossEntropy, self).__init__()
        self.weights = weights
        self.size_average = size_average
        assert (self.size_average == True)  # Not implemented for the other case

    def forward(self, output, target):
        loss = nn.CrossEntropyLoss(self.weights, self.size_average)
        output_one = output.view(-1)
        output_zero = 1 - output_one
        output_converted = torch.stack([output_zero, output_one], 1)
        target_converted = target.view(-1).long()
        return loss(output_converted, target_converted)


class SeGANLoss(nn.Module):

    def __init__(self, weights=None, size_average=True):
        super(SeGANLoss, self).__init__()
        self.weights = weights
        self.size_average = size_average
        assert (self.size_average == True)  # Not implemented for the other case

    def forward(self, output, target):
        background = target == 0
        foreground = target == 1
        loss = nn.BCEWithLogitsLoss(size_average=self.size_average)
        background_loss = loss(output[background], target[background])
        foreground_loss = loss(output[foreground], target[foreground])
        return background_loss + foreground_loss
