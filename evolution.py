import copy
import pickle

from player import Player
import numpy as np
from config import CONFIG


class Evolution():

    def __init__(self, mode):
        self.mode = mode
        self.mutation_rate = 0.1

    # calculate fitness of players
    def calculate_fitness(self, players, delta_xs):
        for i, p in enumerate(players):
            p.fitness = delta_xs[i]

    def mutate(self, child , sigma):

        # TODO
        # child: an object of class `Player`
        for layer in child.nn.parameters:
            chance = np.random.random()
            if chance < self.mutation_rate:
                size = layer.shape
                new_weight = np.random.normal(0,sigma,size = size)
                layer += new_weight



    def q_tournament_selection(self, players, k, q=3):
        selected_players = []
        for _ in range(k) :
            q_players = []
            for i in range(q):
                q_players.append(players[round(np.random.random() * (len(players)-1))])
            selected_players.append(max(q_players,))
        return selected_players

    def crossover(self, parent1, parent2):
        child1 = Player(self.mode)
        child1.nn = copy.deepcopy(parent1.nn)
        child1.fitness = parent1.fitness
        child2 = Player(self.mode)
        child2.nn = copy.deepcopy(parent2.nn)
        child2.fitness = parent2.fitness

        nn_p_number = len(parent1.nn.parameters)

        for i in range(nn_p_number):
            length_m = parent1.nn.parameters[i].shape
            length = length_m[0]*length_m[1]
            layer1 = parent1.nn.parameters[i].reshape(1,length)
            layer2 = parent2.nn.parameters[i].reshape(1,length)
            index = round(np.random.random() * length)
            layer3 = np.array(list(layer1[:index]) + list(layer2[index:])).reshape(length_m)
            layer4 = np.array(list(layer2[:index]) + list(layer1[index:])).reshape(length)
            child1.nn.parameters[i] = layer3
            child2.nn.parameters[i] = layer4

        return child1, child2

    def generate_new_population(self, num_players, prev_players=None):

        # in first generation, we create random players
        if prev_players is None:
            return [Player(self.mode) for _ in range(num_players)]

        else:

            # TODO
            # num_players example: 150
            # prev_players: an array of `Player` objects
            parents = self.q_tournament_selection(prev_players, num_players)
            np.random.shuffle(parents)

            fit_sum = 0
            new_players = []
            for i in range(0, len(parents), 2) :
                fit_sum += prev_players[i].fitness
                fit_sum += prev_players[i+1].fitness
                new_players += self.crossover(parents[i], parents[i+1])
            for i in range(num_players):
                new_players [i] = self.mutate(new_players[i],prev_players[i].fitness/fit_sum)
            return new_players


            # TODO (additional): a selection method other than `fitness proportionate`
            # TODO (additional): implementing crossover

            new_players = prev_players
            return new_players

    def next_population_selection(self, players, num_players):

        # TODO
        # num_players example: 100
        # players: an array of `Player` objects

        # TODO (additional): a selection method other than `top-k`
        # TODO (additional): plotting
        players = sorted(players,key = lambda x : x[1].fitness,reverse=True)
        players = self.q_tournament_selection(players,num_players)
        self.update_learning_curve(players)

        return players[: num_players]

    def update_learning_curve(self, players):
        fitnesses = list(map(lambda i: i.fitness, players))

        result = []
        try:
            with open('learning_curve', 'rb') as f:
                result = pickle.load(f)
        except:
            pass

        maximum_fitness = np.max(fitnesses)
        minimum_fitness = np.min(fitnesses)
        average_fitness = np.average(fitnesses)

        with open('learning_curve', 'wb') as f:
            result.append([maximum_fitness, average_fitness, minimum_fitness])
            pickle.dump(result, f)