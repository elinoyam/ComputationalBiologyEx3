import random

import numpy as np



def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class neural_network:
    def randomize_weights(self, network):
        # create randomized weights to the model
        self.weights = []
        self.activations = []

        if network == None: # if the structure of the network is not defined - randomize the structure as well
            network = []
            # randomize the network shape
            number_of_layers = random.randint(1,4) # generate number of hidden layers
            network.append([16, random.randint(1,10), sigmoid]) # add input layer (always starts with 16 like the input length)
            for index in range(number_of_layers):
                network.append([None, random.randint(1,10), sigmoid])
            network.append([None, 1, sigmoid]) # output layer (always ends with 1 like the label length)

        for index,layer in enumerate(network):
            if layer[0] != None: # if the input size is defined (it's in the 0 index)
                input_size = layer[0]
            else: # if the input size is not defined will set it to be like the output size of the layer before
                input_size = network[index - 1][1]
            output_size = layer[1]
            activation = layer[2]
            self.weights.append(np.random.randn(input_size, output_size)) # random numbers as weights
            self.activations.append(activation)

    def __init__(self, network,*, layers_weights = None, layers_activations = None):
        if layers_weights == None and layers_activations == None: # need new and random weights
            self.randomize_weights(network)
        else:
            self.weights = []
            self.activations = []
            for index in range(network): # here network mean number of layers
                self.weights.append(np.array(layers_weights[index]))
                self.activations.append(sigmoid if layers_activations is None else layers_activations[index]) # sigmoid is the default activation function

    def propagate(self, data):
        # create a label for the all data (must have length of 16)
        input_data = data
        for i in range(len(self.weights)):
            z = np.dot(input_data, self.weights[i]) # matrix multiplication
            a = self.activations[i](z) # for each of the values - calculate the activation function value
            input_data = a
        yhat = a # the last label - still need to be converted to 1/0 by rounding the result

        return yhat

    def save_model(self, file_name):
        # save the model structure and weights so it could be loaded and used in runnet
        with open(file_name, "w") as file:
            # first write the number of layers
            file.write(str(len(self.weights)) + '\n')
            for layer in self.weights: # for each layer
                file.write(str(len(layer)) + "\n") # write the number of rows in the layer matrix
                for row in layer:
                    # write each row in the matrix in a different line with it's weights seperated by ','
                    file.write(",".join(map(str, row)))
                    file.write('\n')

        print("Model was saved in '" + file_name + "' file.")