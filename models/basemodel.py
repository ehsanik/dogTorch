"""
=================
This is the basic model containing place holder/implementation for necessary functions.

All the models in this project inherit from this class
=================
"""

import torch.nn as nn
import torch.optim as optim
import logging


class BaseModel(nn.Module):

    def loss(self):
        return nn.CrossEntropyLoss()

    def optimizer(self):
        # Learning rate here is just a place holder. This will be overwritten
        # at training time.
        return optim.SGD(self.parameters(), lr=0.1)

    def learning_rate(self, epoch):
        assert 1 <= epoch
        if 1 <= epoch <= 30:
            return 0.1
        elif 31 <= epoch <= 60:
            return 0.01
        elif 61 <= epoch <= 90:
            return 0.001
        else:
            return 0.0001

    def evaluation_report(self, output, target):
        raise NotImplementedError

    def load_state_dict(self, state_dict, strict=True):
        """Copies parameters and buffers from :attr:`state_dict` into
        this module and its descendants. If :attr:`strict` is ``True`` then
        the keys of :attr:`state_dict` must exactly match the keys returned
        by this module's :func:`state_dict()` function.
        Arguments:
            state_dict (dict): A dict containing parameters and
                persistent buffers.
            strict (bool): Strictly enforce that the keys in :attr:`state_dict`
                match the keys returned by this module's `:func:`state_dict()`
                function.
        """
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    raise RuntimeError(
                        'While copying the parameter named {}, '
                        'whose dimensions in the model are {} and '
                        'whose dimensions in the checkpoint are {}.'.format(
                            name, own_state[name].size(), param.size()))
            elif strict:
                raise KeyError('unexpected key "{}" in state_dict'.format(name))
            else:
                logging.warning(
                    'Parameter {} not found in own state'.format(name))
        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError(
                    'missing keys in state_dict: "{}"'.format(missing))
        else:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                logging.warning(
                    'missing keys in state_dict: "{}"'.format(missing))
