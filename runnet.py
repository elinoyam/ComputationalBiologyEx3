# In this program will receive two files
# one with the model settings
# the other with strings to test

# import neural_network



# open wnet file with the model setting and build the model
with open("wnet.txt","r") as model_settings:
    # read the model structure settings and build the model accordingly
    model_layers = int(model_settings.readline())
    # create a neural network model with model_layers layers
    # model = neural_network(model_layers)
    for layer in range(model_layers):
        layer_weights_line = model_settings.readline()
        layer_weights = [float(weight) for weight in layer_weights_line.split(' ')]
        # model.update_layer(layer, layer_weights)



# get the test file to open and label
with open("testnet0.txt", "r") as test_file:
    # go through each line and give the string an assignment (0/1)
    # for each string write the assigned label in the "labels.txt" file
    with open("labels0.txt", "w") as result_file:
        for line in range (20000):
            string_to_test = test_file.readline()
            label = ""
            # label = model.assign(line)
            # write the result label to the result_file
            result_file.write(label + "\n" )