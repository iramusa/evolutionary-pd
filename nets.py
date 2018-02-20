#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.autograd import Variable

FC_WIDTH = 10
PERTURBATION_NOISE = 0.005


def perturb_parameters(m):
    classname = m.__class__.__name__
    if 'Linear' in classname or 'GRU' in classname:
        print("perturbing", m)
        m.weight.data += m.weight.data.normal_(0.0, PERTURBATION_NOISE)


class PDBrain(nn.Module):
    def __init__(self):
        super(PDBrain, self).__init__()
        self.fc_seq = nn.Sequential(
            nn.Linear(1, FC_WIDTH),
            nn.Sigmoid(),
            nn.Linear(FC_WIDTH, FC_WIDTH),
            nn.Sigmoid(),
        )
        self.recurrent = nn.GRU(FC_WIDTH, 1, num_layers=1)

    def forward(self, last_move, state):
        h = self.fc_seq(last_move)
        new_move, state = self.recurrent(h.view(1, 1, 10), state)
        return new_move, state

    def perturb_weights(self):
        for p in self.parameters():
            noise = torch.normal(means=torch.zeros(p.size()),
                                 std=torch.ones(p.size()) * PERTURBATION_NOISE)
            p.data = p.data + noise


if __name__ == "__main__":
    move_p1 = 1
    move_tensor_p1 = Variable(torch.FloatTensor(1, 1))

    net = PDBrain()

    # move_tensor_p1.data.fill_(move_p1)
    # move_p1, state = net(move_tensor_p1, None)
    # move_p1 = net(move_tensor_p1, None)
    # print(move_p1)

    print([p for p in net.parameters()])
    print("BREAKREAKREAKREAKREAKREA")
    net.perturb_weights()
    print([p for p in net.parameters()])
