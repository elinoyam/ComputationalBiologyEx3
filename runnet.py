# In this program will receive two files
# one with the model settings
# the other with strings to test

import sys
from neural_network import neural_network
import numpy as np

if __name__ == "__main__":
    #wanted_model = sys.argv[1]
    wanted_model = input("insert which model to run (select 1 or 0)\n")
    if wanted_model != "0" and wanted_model != "1":
        raise Exception("The input for the wanted model must be 1 or 0 only.")

    # open wnet file with the model setting
    with open("wnet"+ wanted_model + ".txt", "r") as model_settings:
        # read the model structure settings and build the model accordingly
        model_layers = int(model_settings.readline()) # number of layers

        # create a neural network model with model_layers layers
        model_weights = []
        for layer in range(model_layers): # start to read each layer
            rows_number = int(model_settings.readline()) # read the number of rows in the layer matrix
            rows = []

            for row in range(rows_number):
                # each row is seperated in a different line and the weights of the row is seperated by ','
                weights_line = model_settings.readline()
                rows.append(np.array([float(weight) for weight in weights_line.split(',') if (weight != '' and weight != '\n')]))

            model_weights.append(rows)

        model = neural_network(model_layers, layers_weights=model_weights) # create the model



    # for self testing:
    # X = []
    # Y = []
    # with open("testnet" + wanted_model + ".txt", "r") as test_file:
    #     # go through each line and give the string an assignment (0/1)
    #     # for each string write the assigned label in the "labels.txt" file
    #     line = test_file.readline()
    #     while line != "" and line != "\n":
    #         input, label = line[:-2], line[-2:-1]
    #         X.append([int(j) for j in input])
    #         Y.append(int(label))
    #         line = test_file.readline()
    #
    #     inputs = np.array(X)
    #     real_labels = np.array([Y])
    #
    #     labels = model.propagate(inputs)
    #     labels = np.round(labels)
    #     with open("labels" + wanted_model + ".txt", "w") as result_file:
    #         for index, label in enumerate(labels):
    #             # write the result label to the result_file
    #             result_file.write(str(int(label)) + "=" + str(real_labels[0][index]) + "==" + str(label == real_labels[0][index]) +  "\n" )


    # for submission:
    X = []
    removed_characters = []
    counter = 0;
    with open("testnet" + wanted_model + ".txt", "r") as test_file:
        line = test_file.readline()
        while line != "" and line != '\n':
            line = line.rstrip('\n')  # Remove newline character
            #last_character = line[-1]
            #removed_characters.append(last_character)
            #line = line[:-1]
            X.append([int(j) for j in line])
            line = test_file.readline()
            counter = counter + 1

        inputs = np.array(X)
        labels = model.propagate(inputs)
        labels = np.round(labels)
        labels = np.squeeze(labels).astype(int)
        labels = labels.astype(str)

        #removed_characters_array = np.array(removed_characters)
        # write the result to the labels file
        #num_equal_elements = np.sum(labels == removed_characters_array)
        #print("Accuracy is: "+ str(num_equal_elements/counter*100))
        with open("labels" + wanted_model + ".txt", "w") as result_file:
            for label in labels:
                result_file.write(str(int(label.item())) + "\n")