import torch
import torch.nn as nn

from extensions.multi_label_cross_entropy import MultiLabelCrossEntropyLoss
from torch.autograd import Variable
from training import metrics
from .basemodel import BaseModel


class NearestNeighbor(BaseModel):
    metric = [metrics.SequenceMultiClassMetric]#, metrics.AngleEvaluationMetric]

    def __init__(self, args):
        super(NearestNeighbor, self).__init__()
        self.class_weights = args.dataset.CLASS_WEIGHTS[torch.LongTensor(
            args.imus)]
        self.input_length = args.input_length
        self.output_length = args.output_length
        self.train_loader = args.train_loader
        self.num_classes = args.num_classes

    def forward(self, input, target):
        if self.training:
            raise RuntimeError("Can't do forward pass in training.")
        input = input[:, :self.input_length]
        output_indices = list(range(target.size(1) - self.output_length, target.size(1)))
        target = target[:,-self.output_length:,:]
        input = input.data.contiguous().view(input.size(0), -1)
        new_tensor = input.new
        min_distances = new_tensor(input.size(0)).fill_(1e9)
        best_labels = new_tensor(target.size()).long().fill_(-1)
        for i, (train_data, train_label, _, _) in enumerate(self.train_loader):
            train_data = train_data[:, :self.input_length].contiguous()
            train_data = train_data.view(train_data.size(0), -1).cuda(async=True)
            train_label = train_label[:, -self.output_length:].contiguous()
            train_label = train_label.cuda(async=True)
            distances = -torch.mm(input, train_data.t())
            cur_min_distances, min_indices = distances.min(1)
            min_indices = min_indices[:,None,None].expand_as(best_labels)
            cur_labels = train_label.gather(0, min_indices)
            old_new_distances = torch.stack([min_distances, cur_min_distances])
            min_distances, picker = old_new_distances.min(0)
            picker = picker[:,None,None].expand_as(best_labels)
            best_labels = (1 - picker) * best_labels + picker * cur_labels
        output = new_tensor(*best_labels.size(), self.num_classes).fill_(0)
        output.scatter_(3, best_labels.unsqueeze(3), 1)
        output = Variable(output)
        return output, target, torch.LongTensor(output_indices)

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
