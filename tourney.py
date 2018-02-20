#!/usr/bin/env python3

import numpy as np
import torch
from torch.autograd import Variable

from players import NeuralPlayer

PAYOFF_MATRIX = np.array([[1, 5],
                         [0, 4]])

N_PLAYERS = 100
PD_ITERS = 40

flag = 0


class Tourney(object):
    def __init__(self, players):
        self.players = []
        self.scores = {}
        self.coop_count = 0
        self.iter_count = 0

        for player in players:
            self.players.append(player)
            self.scores.update({player: 0})

    def print_tourney_stats(self):
        print("Max score: ", np.max(list(self.scores.values())))
        print("Mean score: ", np.mean(list(self.scores.values())))
        print("Min score: ", np.min(list(self.scores.values())))
        print("Cooperation rate: ", self.coop_count/self.iter_count)
        print("Games played: ", self.iter_count)
        print(" ")

    def play_n_rounds(self, n):
        for _ in range(n):
            self.play_round()

    def play_round(self):
        players_left = list(range(len(self.players)))

        while len(players_left) >= 2:
            p1_no = np.random.randint(len(players_left))
            p1_no = players_left.pop(p1_no)
            p1 = self.players[p1_no]

            p2_no = np.random.randint(len(players_left))
            p2_no = players_left.pop(p2_no)
            p2 = self.players[p2_no]

            score_p1, score_p2 = self.play_game(p1, p2)
            self.scores[p1] += score_p1
            self.scores[p2] += score_p2

    def play_game(self, p1, p2):
        global flag
        score_p1 = score_p2 = 0
        move_p1 = move_p2 = 0.5

        move_tensor_p1 = Variable(torch.FloatTensor(1, 1))
        move_tensor_p2 = Variable(torch.FloatTensor(1, 1))

        for i in range(PD_ITERS):
            self.iter_count += 1

            move_tensor_p1.data.fill_(move_p1)
            move_tensor_p2.data.fill_(move_p2)

            move_p1 = p1.play(move_tensor_p2)
            move_p2 = p2.play(move_tensor_p1)

            if move_p1 == move_p2 == 1:
                self.coop_count += 1

            if flag == 0:
                print("p1: ", move_p1, "p2: ", move_p2)
            score_p1 += PAYOFF_MATRIX[move_p1, move_p2]
            score_p2 += PAYOFF_MATRIX[move_p2, move_p1]

        flag = 1
        return score_p1, score_p2


if __name__ == "__main__":
    players = [NeuralPlayer() for i in range(N_PLAYERS)]
    t = Tourney(players)
    print("Tourney starting")

    t.play_round()
    print("Round is over")
    print("Results")
    print(t.scores)
