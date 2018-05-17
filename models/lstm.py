"""
=================
This model is an LSTM layer that takes input and outputs a sequence of imu values.
=================
"""

import torch
import torch.nn as nn
import pdb

from torchvision.models import alexnet as torchvision_alexnet
from extensions.multi_label_cross_entropy import MultiLabelCrossEntropyLoss
from .basemodel import BaseModel
from torch.autograd import Variable


def _target_to_hot_vectors(target, num_classes):
    hot_vectors = torch.cuda.FloatTensor(*target.size(), num_classes).zero_()
    hot_vectors.scatter_(2, target.data.unsqueeze(2), 1)
    hot_vectors = Variable(hot_vectors)
    return hot_vectors


class Lstm(BaseModel):

    def __init__(self, args):
        super(Lstm, self).__init__()

        self.class_weights = args.dataset.CLASS_WEIGHTS[torch.LongTensor(
            args.imus)]
        self.num_classes = args.num_classes
        self.imus = args.imus
        self.output_length = args.output_length
        self.num_layers = args.num_layers
        self.hidden_size = args.hidden_size
        self.imu_feature = args.imu_feature
        self.use_attention = args.use_attention
        self.regression = args.regression

        self.encoder = nn.LSTM(args.hidden_size, args.hidden_size,
                               args.num_layers)
        if self.use_attention:
            self.embedding_output = nn.Linear(
                args.imu_feature + args.hidden_size, args.hidden_size)
        else:
            self.embedding_output = nn.Linear(args.imu_feature,
                                              args.hidden_size)
        self.decoder = nn.LSTMCell(args.hidden_size, args.hidden_size,
                                   args.num_layers)
        self.out = nn.Linear(args.hidden_size, args.imu_feature)
        if self.use_attention:
            self.attender = nn.Linear(args.hidden_size, args.input_length)

        for i in self.imus:
            setattr(self, 'imu{}'.format(i),
                    nn.Linear(args.imu_feature, args.num_classes))

        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.hidden_linear = nn.Linear(args.hidden_size, args.hidden_size)
        self.cell_linear = nn.Linear(args.hidden_size, args.hidden_size)

    def forward(self, input, target=None):
        batch_size = input.size(1)

        hidden, cell = self.initHiddenCell(batch_size)
        output_encoder, (hidden, cell) = self.encoder(input, (hidden, cell))
        full_output = []

        memory = output_encoder.transpose(0, 1)
        hidden = self.hidden_linear(hidden).squeeze(0)
        cell = self.cell_linear(cell).squeeze(0)

        decoder_input = self.initDecoderInput(batch_size)
        decoder_input_size = decoder_input.size()

        for seq_index in range(self.output_length):

            decoder_input = decoder_input.transpose(0, 1)
            decoder_input = decoder_input.squeeze(0)
            if self.use_attention:
                pdb.set_trace()
                attention = self.softmax(self.attender(hidden)).unsqueeze(1)
                attended_memory = torch.bmm(attention, memory).squeeze(1)
                decoder_input = torch.cat([decoder_input, attended_memory], 1)
            decoder_input = self.embedding_output(decoder_input)
            decoder_input = self.dropout(decoder_input)
            decoder_input = self.relu(decoder_input)
            hidden, cell = self.decoder(decoder_input, (hidden, cell))
            output_gru = hidden

            output = (self.out(output_gru))
            output = output.view(batch_size, self.num_classes * len(self.imus))
            if self.regression:
                cleaned_output = output.view(batch_size, len(self.imus),
                                             self.num_classes)
                not_cleaned_output = cleaned_output
            else:
                imu_out = []
                for i in self.imus:
                    imu_i = getattr(self, 'imu{}'.format(i))
                    imu_out.append(imu_i(output))

                imu_soft = [
                    self.softmax(imu).view(batch_size, 1, self.num_classes)
                    for imu in imu_out
                ]
                cleaned_output = torch.cat(imu_soft, 1)
                not_cleaned_output = torch.cat(imu_out, 1).view(
                    batch_size, len(self.imus), self.num_classes)

            full_output.append(not_cleaned_output)

            if target is None:  #this means teacher forcing is disabled
                decoder_input = cleaned_output.view(decoder_input_size)

        full_output = torch.stack(full_output)
        return full_output

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

    def initHiddenCell(self, batch_size):
        result = Variable(
            torch.zeros(self.num_layers, batch_size, self.hidden_size))
        cell = Variable(
            torch.zeros(self.num_layers, batch_size, self.hidden_size))
        return result.cuda(), cell.cuda()

    def initDecoderInput(self, batch_size):
        res = Variable(torch.zeros(batch_size, 1, self.imu_feature))
        return res.cuda()
