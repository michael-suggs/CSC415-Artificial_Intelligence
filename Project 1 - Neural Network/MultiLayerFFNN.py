import numpy as np
__author__ = "Michael Suggs // mjs3607@uncw.edu"


class MultiLayerFFNN:
    training_sets = []
    output = []
    testing_sets = []
    testing_output = []

    """
    """
    def __init__(self, num_inputs, hidden_per_layer, hidden_depth, num_outputs, target_tsse, eta):
        self.input_vector = np.zeros([num_inputs, 1])
        self.input_weights = 2 * np.random.rand(num_inputs, hidden_per_layer) - 1

        # self.hidden_vector = np.zeros([num_hidden_nodes, hidden_depth])
        # self.hidden_weights = 2 * np.random.rand(num_hidden_nodes, num_outputs) - 1
        # self.hidden_bias = np.ones([num_hidden_nodes, hidden_depth])
        # self.hidden_bias_weights = 2 * np.random.rand(num_hidden_nodes, hidden_depth) - 1

        self.hidden_vector_list = [np.zeros([hidden_per_layer[i],
                                             1]) for i in range(hidden_depth)]
        self.hidden_weights_list = [(2 * np.random.rand(hidden_per_layer[i],
                                                        hidden_per_layer[i + 1]) - 1)
                                    for i in range(hidden_depth - 1)]
        self.hidden_weights_list.append((2 * np.random.rand(hidden_per_layer[-1], num_outputs) - 1))

        self.hidden_bias_list = [(np.ones(hidden_per_layer[i], 1))
                                 for i in range(hidden_depth)]
        self.hidden_bias_weights_list = [(2 * np.random.rand(hidden_per_layer[i], 1) - 1)
                                         for i in range(hidden_depth)]

        # self.output_vector = np.zeros([num_outputs, 1])
        self.target_output_vector = np.zeros([num_outputs, 1])
        self.output_bias = np.ones([num_outputs, 1])
        self.output_bias_weights = 2 * np.random.rand(num_outputs, 1) - 1

        self.target_tsse = target_tsse
        self.tsse = 0
        # self.target_tsse = self.rmse_to_tsse(target_rmse, num_outputs)
        self.eta = eta

    """
    """
    def rmse_to_tsse(self, target_rmse, num_outputs):
        tsse = ((target_rmse**2) * len(self.training_sets
                        * num_outputs))/2
        return tsse

    """
    """
    def train(self, epochs):
        current_epoch = 0

        while current_epoch < epochs or self.tsse > self.target_tsse:
            self.output.clear()
            np.random.shuffle(self.training_sets)

            for elem in self.training_sets:
                for i in range(len(elem[0])):
                    for j in range(len(elem[0][i])):
                        self.input_vector[i*len(elem[0][i])+j] = elem[0][i][j]

                for i in range(len(elem[1])):
                    self.target_output_vector[i] = float(elem[1][i])

                out = self.feedforward()
                self.output.append((elem[0], out))
                self.backpropagate(out)

            if current_epoch % 250 == 0 and current_epoch != 0:
                self.tsse = self.calc_tsse()
                # print(current_epoch, ": ", self.tsse)
                # print(self.output)

                # if current_epoch > epochs:
                #     print(self.input_weights)
                #     print(self.hidden_weights)

            current_epoch += 1

        print("Final Epoch: ", current_epoch)

    """
    """
    def feedforward(self):
        output_vector = np.zeros([self.target_output_vector.shape[0], 1])

        input_transpose = self.input_vector.transpose()
        input_weight_dot = input_transpose.dot(self.input_weights).T

        # calculates the output of each hidden node
        for i in range(self.hidden_vector.shape[0]):
            self.hidden_vector[i] = self.calc_hidden_output(i, input_weight_dot)

        hidden_transpose = self.hidden_vector.transpose()
        hidden_weight_dot = hidden_transpose.dot(self.hidden_weights).T

        # calculates the final output of the network
        for i in range(output_vector.shape[0]):
            output_vector[i] = self.calc_output(i, hidden_weight_dot)

        return output_vector

    """
    """
    def backpropagate(self, output_vector):
        output_error = np.zeros((output_vector.shape[0], 1))
        for i in range(output_vector.shape[0]):
            output_error[i] = (self.target_output_vector[i] - output_vector[i])\
                              * output_vector[i] * (1 - output_vector[i])

        for i in range(self.output_bias_weights.shape[0]):
            self.output_bias_weights[i] += self.eta * output_error[i] * self.output_bias[i]

        output_hidden_dot = np.dot(self.hidden_weights, output_error)
        hidden_error = np.zeros((self.hidden_vector.shape[0], 1))
        for i in range(self.hidden_vector.shape[0]):
            hidden_error[i] = self.hidden_vector[i] \
                              * (1 - self.hidden_vector[i]) \
                              * (output_hidden_dot.sum(axis=1)[i])
                            # ^ Sum of each row of that dot product
                            # each row corresponds to all output nodes for a given hidden node

        for i in range(self.hidden_bias_weights.shape[0]):
            self.hidden_bias_weights[i] += self.eta *\
                                           hidden_error[i] *\
                                           self.hidden_bias[i]

        # compute weight adjustments
        for i in range(self.hidden_weights.shape[0]):
            for j in range(self.hidden_weights.shape[1]):
                self.hidden_weights[i][j] += self.eta\
                                             * output_error[j]\
                                             * self.hidden_vector[i]

        # calc input weights
        for i in range(self.input_weights.shape[0]):
            for j in range(self.input_weights.shape[1]):
                self.input_weights[i][j] += self.eta \
                                            * hidden_error[j] \
                                            * self.input_vector[i]

    """
    """
    def calc_hidden_output(self, hidden_index, input_weight_dot):
        netj = (self.hidden_bias[hidden_index] *
                self.hidden_bias_weights[hidden_index]) +\
                input_weight_dot[hidden_index]
        node_output = (1 / (1 + (np.exp(-1 * netj))))
        return node_output

    """
    """
    def calc_output(self, output_index, hidden_weight_dot):
        netj = (self.output_bias[output_index] *
                self.output_bias_weights[output_index]) + \
               hidden_weight_dot[output_index]
        node_output = (1 / (1 + (np.exp(-1 * netj))))
        return node_output

    """
    """
    def calc_tsse(self):
        total_sum_squares = 0

        for elem in self.training_sets:
            for i in range(len(elem[0])):
                for j in range(len(elem[0][i])):
                    self.input_vector[i * len(elem[0][i]) + j] = elem[0][i][j]
            for i in range(len(elem[1])):
                self.target_output_vector[i] = float(elem[1][i])

            out = self.feedforward()
            for i in range(out.shape[0]):
                total_sum_squares += (self.target_output_vector[i] - out[i])**2

        return .5 * total_sum_squares

    """
    """
    def test(self):
        for elem in self.testing_sets:
            for i in range(len(elem[0])):
                for j in range(len(elem[0][i])):
                    self.input_vector[i * len(elem[0][i]) + j] = elem[0][i][j]

            test_out = self.feedforward()
            self.testing_output.append((elem[0], test_out))

    """
    """
    def load_training_data(self, matrix):
        for elem in matrix:
            self.training_sets.append(elem)

    """
    """
    def load_testing_data(self, matrix):
        for elem in matrix:
            self.testing_sets.append(elem)

    """
    """
    def print_output(self, testing=False):
        if testing == False:
            for elem in self.output:
                print(elem)
            print("\n")
        else:
            for elem in self.testing_output:
                print(elem)
            print("\n")

    """
    """
    def write_output(self, file, testing=True):
        with open(file, "w") as f:
            if testing is True:
                for elem in self.testing_output:
                    in1 = elem[0][0]
                    in2 = elem[0][1]
                    out = elem[1]

                    output_string = str(in1) + ", " + str(in2) + ", " + str(out) + "\n"
                    f.write(output_string)
