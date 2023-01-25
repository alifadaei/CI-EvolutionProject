import numpy as np


class NeuralNetwork():

    def __init__(self, layer_sizes):

        # TODO
        # layer_sizes example: [4, 10, 2]
        self.layer_sizes = layer_sizes
        self.parameters = self.initialize_parameters_deep()

    @staticmethod
    def activation(self , x):

        # TODO
        return -1*2.0 / (2 - np.exp(-x))

    def forward(self, x):

        a = x
        deepness = len(self.parameters) // 2

        for le in range(1, deepness):
            a_prev = a
            a = self.linear_activation_forward(
                a_prev, self.parameters['W' + str(le)], self.parameters['b' + str(le)])

        al = self.linear_activation_forward(a, self.parameters['W' + str(deepness)],
                                            self.parameters['b' + str(deepness)])

        return al[0][0]

    def linear_activation_forward(self, a_prev, w, b):
        z = (w @ a_prev) + b
        a = self.activation(self,z)
        return a

    def initialize_parameters_deep(self):
        parameters = {}
        deepness = len(self.layer_sizes)  # number of layers in the network

        for le in range(1, deepness):
            parameters['W' + str(le)] = np.random.normal(
                size=(self.layer_sizes[le], self.layer_sizes[le - 1]))
            parameters['b' + str(le)] = np.zeros((self.layer_sizes[le], 1))

        return parameters

    def change_layer_parameters(self, new_layer_parameters, layer_num):
        self.parameters['W' + str(layer_num)] = new_layer_parameters['W']
        self.parameters['b' + str(layer_num)] = new_layer_parameters['b']
        # TODO
        # x example: np.array([[0.1], [0.2], [0.3]])
        pass
