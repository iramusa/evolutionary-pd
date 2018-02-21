#!/usr/bin/env python3

import torch
from torch import nn
import copy
import numpy as np

from nets import PDBrain


class Player(object):
    def __init__(self):
        pass

    def play(self, last_move, state=None):
        pass

    def replicate(self):
        return copy.deepcopy(self)

    def save(self, filepath):
        torch.save(self, filepath)


LSTM_WIDTH = 10
LSTM_DEPTH = 1
FC_WIDTH = 10


class NeuralPlayer(Player):
    def __init__(self):
        super(Player, self).__init__()
        self.net = PDBrain()
        self.state = None

    def play(self, last_move):
        new_move_p, self.state = self.net(last_move, self.state)

        if np.random.rand() > new_move_p.data.cpu().numpy()[0, 0]:
            new_move = 1
        else:
            new_move = 0

        return new_move

    def replicate(self):
        new_player = NeuralPlayer()
        new_player.net.load_state_dict(self.net.state_dict())
        new_player.net.perturb_weights()
        return new_player


class TitTatPlayer(Player):
    def __init__(self):
        super(Player, self).__init__()

    def play(self, last_move):
        last_move_float = last_move.data.cpu().numpy()[0, 0]
        # print(last_move_float)
        if last_move_float > 0.4:
            return 1
        else:
            return 0

