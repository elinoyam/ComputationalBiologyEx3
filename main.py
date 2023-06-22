# Ex 3
import random
import numpy as np # need to install


class genetic_algorithm:

    def execute(pop_size, generations, threshold, X, y, network):
        class Agent:
            def __init__(self, network):
                pass

class neural_network:
    def __init__(self, network):
        self.weights = []
        self.activations = []
        for layer in network:
            if layer[0] != None:
                input_size = layer[0]
            else:
                input_size = network[network.index(layer) - 1][1]
            output_size = layer[1]
            activation = layer[2]
            self.weights.append(np.random.randn(input_size, output_size))
            self.activations.append(activation)

        def propagate(self, data):
            input_data = data
            for i in range(len(self.weights)):
                z = np.dot(input_data, self.weights[i])
                a = self.activations[i](z)
                input_data = a
            yhat = a
            return yhat

        self.neural_network = neural_network(network)
        self.fitness = 0

        