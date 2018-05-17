import torch.nn as nn


class MultiLabelCrossEntropyLoss(nn.Module):

    def __init__(self, weights, size_average=True):
        super(MultiLabelCrossEntropyLoss, self).__init__()
        self.criterions = [
            nn.CrossEntropyLoss(weight, size_average)
            for weight in weights.cuda()
        ]
        self.size_average = size_average

    def forward(self, output, target):
        assert output.dim() == 4
        assert target.dim() == 3
        if output.size(0) != target.size(0):
            raise ValueError("Inequal batch size ({} vs. {})".format(
                output.size(0), target.size(0)))
        if output.size(1) != target.size(1):
            raise ValueError("Inequal sequence length ({} vs. {})".format(
                output.size(1), target.size(1)))
        if output.size(2) != target.size(2):
            raise ValueError("Inequal number of labels ({} vs. {})".format(
                output.size(2), target.size(2)))
        if output.size(2) != len(self.criterions):
            raise ValueError("Unexpected number of labels ({} vs. {})".format(
                output.size(2), len(self.criterions)))

        loss = 0
        for i, criterion in enumerate(self.criterions):
            # Merge sequence length to batch_size
            label_output = output[:, :, i, :].contiguous().view(
                -1, output.size(3))
            label_target = target[:, :, i].contiguous().view(-1)
            loss += criterion(label_output, label_target)
        if self.size_average:
            loss /= len(self.criterions)
        return loss
