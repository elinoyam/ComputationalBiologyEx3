import random
import sys
import numpy as np
from neural_network import neural_network, sigmoid


MUTATION_RATE = 0.5
RESET_RATE = 0.8
SELECTION_PERCENTAGE = 0.4

class genetic_algorithm:

    def execute(pop_size, generations, threshold, X, y, network):
        class Agent:
            def __init__(self, network):

                self.neural_network = neural_network(network)
                self.fitness = 0

        def generate_agents(population, network):
            return [Agent(network) for _ in range(population)];

        def fitness(agents, X, y):
            for agent in agents:
                yhat = agent.neural_network.propagate(X)
                yhat = [[round(result[0])] for result in yhat] # get the real label (0/1)
                cost = (yhat - y.T) ** 2 # MSE - Mean Square Error
                # the fitness is equal to the number of correct guesses divided by the number of samples
                agent.fitness = (len(cost) - sum(cost)) / len(cost)

            return agents

        def selection(agents):
            global SELECTION_PERCENTAGE
            agents = sorted(agents, key=lambda agent: agent.fitness, reverse=True) # sort the models by their fitness score in descending order

            agents = agents[:int(SELECTION_PERCENTAGE * len(agents))] # get the SELECTION_PERCENTAGE % of the bests models
            return agents

        def unflatten(flattened, shapes):
            # return the weights to the original matrix shapes
            newarray = []
            index = 0
            for shape in shapes:
                size = np.product(shape) # how many weights in the matrix
                newarray.append(flattened[index: index + size].reshape(shape))  # reshape those weights
                index += size
            return newarray

        def crossover(agents, network, pop_size):
            offspring = []
            for _ in range((pop_size - len(agents)) // 2):
                parent1 = random.choice(agents)
                parent2 = random.choice(agents)

                shapes = None
                is_parent1_fitter = None
                if parent1.fitness > parent2.fitness: # if the first parent have better fitness score
                    shapes = [np.shape(layer) for layer in parent1.neural_network.weights] # the children will receive parent1 weights shapes
                    is_parent1_fitter = True
                else:
                    shapes = [np.shape(layer) for layer in parent2.neural_network.weights] # the children will receive parent2 weights shapes
                    is_parent1_fitter = False

                # build the children network structure - according to the fitter parent
                better_network = []
                better_network.append([shapes[0][0], shapes[0][1], sigmoid])
                for index in range(1, len(shapes)):
                    better_network.append([shapes[index][0], shapes[index][1], sigmoid])
                child1 = Agent(better_network)
                child2 = Agent(better_network)

                genes1 = np.concatenate([a.flatten() for a in parent1.neural_network.weights])
                genes2 = np.concatenate([a.flatten() for a in parent2.neural_network.weights])
                larger_weights = genes1 if len(genes1) > len(genes2) else genes2

                smallest_length = (len(genes1) - 1) if len(genes1) < len(genes2) else (len(genes2) - 1)
                shapes_sum = (len(genes1)) if is_parent1_fitter else (len(genes2))
                split_max = min(smallest_length, shapes_sum)
                split = random.randint(0, split_max)

                child1_genes = genes1[0:split].tolist() + genes2[split:split_max].tolist()
                if len(child1_genes) < shapes_sum:
                    child1_genes += larger_weights[split_max:shapes_sum].tolist()

                child1_genes = np.array(child1_genes)

                child2_genes=genes2[0:split].tolist() + genes1[split:split_max].tolist()
                if len(child2_genes) < shapes_sum:
                    child2_genes+= larger_weights[split_max:shapes_sum].tolist()

                child2_genes = np.array(child2_genes)

                child1.neural_network.weights = unflatten(child1_genes, shapes)
                child2.neural_network.weights = unflatten(child2_genes, shapes)

                offspring.append(child1)
                offspring.append(child2)
            agents.extend(offspring)
            return agents

        def mutation(agents):
            for agent in agents:
                if random.uniform(0.0, 1.0) <= MUTATION_RATE: # mutate only MUTATION_RATE % of the population
                    weights = agent.neural_network.weights
                    shapes = [a.shape for a in weights]
                    flattened = np.concatenate([a.flatten() for a in weights])

                    if random.uniform(0.0,1.0) <= RESET_RATE:
                        # change 1 value to random value
                        randintg = random.randint(0, len(flattened) - 1)
                        flattened[randintg] = np.random.randn()
                    else:
                        # reset part of the weights in random indices
                        indices = np.random.choice(np.arange(flattened.size), replace=False,
                                                   size=int(flattened.size * 0.2))
                        flattened[indices] = 0

                    agent.neural_network.weights = unflatten(flattened, shapes) # return to the original shape
            return agents


        # MAIN LOOP
        global SELECTION_PERCENTAGE, MUTATION_RATE
        last_fitness = 0
        for i in range(generations):
            if i == 0:
                agents = generate_agents(pop_size, network)

            agents = fitness(agents, X, y)
            agents = selection(agents)
            agents = crossover(agents, network, pop_size)
            agents = mutation(agents)
            # agents = fitness(agents, X, y)
            if any(agent.fitness > threshold for agent in agents):
                print('Threshold met at generation ' + str(i) + ' !')

            if i % 10 == 0:
                if agents[0].fitness - last_fitness < 0.001:
                    MUTATION_RATE = max(MUTATION_RATE + 0.25, 1)
                    SELECTION_PERCENTAGE = max(0.2, SELECTION_PERCENTAGE - 0.1)
                else:
                    MUTATION_RATE = 0.5 # default value
                    SELECTION_PERCENTAGE = 0.4
                print('Generation', str(i), ':')
                print('The Best agent has fitness ' + str(agents[0].fitness) + 'at generation ' + str(i) + '.')
                print('The Worst agent has fitness ' + str(agents[-1].fitness) + 'at generation ' + str(i) + '.')
                last_fitness = agents[0].fitness

        agents = fitness(agents, X, y)
        best_agent = sorted(agents, key=lambda agent: agent.fitness, reverse=True)[0]
        return best_agent


if __name__ == "__main__":
    if len(sys.argv) == 0:
        raise Exception("You must insert which model to run (select 1 or 0).")

    wanted_model = sys.argv[1]

    if wanted_model != "0" and wanted_model != "1":
        raise Exception("The input for the wanted model must be 1 or 0 only.")

    # open the test file and split it into learn db and test db
    X = []
    Y = []

    with open("nn" + wanted_model + ".txt", "r") as strings_file:
        line = strings_file.readline()
        while line != "" and line != "\n":
            input, label = line.split('   ')
            X.append([int(j) for j in input])
            Y.append(int(label))
            line = strings_file.readline()

        split_test_index = round(len(X) * 0.8)
        build_inputs = np.array(X[0:split_test_index])
        test_inputs = np.array(X[split_test_index:])
        build_labels = np.array([Y[:split_test_index]])
        test_labels = np.array([Y[split_test_index:]])

    # network = [[16,10,sigmoid], [10, 2, sigmoid], [2,1,sigmoid]]
    network = None
    ga = genetic_algorithm

    agent = ga.execute(200,300,0.99,build_inputs,build_labels,network)
    weights = agent.neural_network.weights
    print(agent.fitness)

    results = agent.neural_network.propagate(test_inputs)
    labels = labels = np.round(results)

    # print(final_results)
    # print(test_labels[0])

    # check if same labeling:
    # is_same = True
    # diff = 0
    # for i in range(len(final_results)):
    #     if final_results[i] != test_labels[0][i]:
    #         is_same = False
    #         diff +=1

    # print("diff is = " + str(diff) + " so accurecy = " + str((len(final_results) - float(diff)) / len(final_results)))
    # if is_same:
    #     print("Results are Goooooooddddd!!")

    agent.neural_network.save_model("wnet" + wanted_model + ".txt")


