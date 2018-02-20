#!/usr/bin/env python3

import numpy as np
import torch

from players import NeuralPlayer, TitTatPlayer
from tourney import Tourney

N_PLAYERS = 100
ROUNDS_BETWEEN_REPLICATIONS = 2
N_TOURNEYS = 20


class Population(object):
    def __init__(self):
        self.population = []
        self.fitness = []

    def save_population(self):
        torch.save(self.population, open("population.torch", 'wb'))

    def load_population(self):
        self.population = torch.load(open("population.torch", 'rb'))

    def get_fitness(self):
        t = Tourney(self.population)
        t.play_n_rounds(ROUNDS_BETWEEN_REPLICATIONS)
        self.fitness = []
        for player in self.population:
            self.fitness.append(t.scores[player])

        t.print_tourney_stats()

    def replicate_players(self):
        # probability of picking a player is a function of fitness
        fit = np.array(self.fitness)/ROUNDS_BETWEEN_REPLICATIONS
        # fit = fit - np.min(fit)
        fit = fit / np.sum(fit)

        indices = np.random.choice(np.arange(len(self.population)), size=N_PLAYERS, replace=True, p=fit)

        # destroys some of the players, some objects are referenced multiple times
        self.population = [self.population[i] for i in indices]

        # mutate replicas
        for i in range(len(self.population)):
            self.population[i] = self.population[i].replicate()


if __name__ == "__main__":
    pop = Population()
    pop.load_population()

    # for i in range(N_PLAYERS):
    #     pop.population.append(NeuralPlayer())

    # for i in range(50):
    #     pop.population.append(TitTatPlayer())

    for i in range(N_TOURNEYS):
        print(i)
        pop.get_fitness()
        pop.replicate_players()

    pop.save_population()