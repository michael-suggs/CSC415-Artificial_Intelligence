import numpy as np
__author__ = "Michael Suggs // mjs3607@uncw.edu"


class FeedForwardNetwork:
    """ Creates a single hidden layer feedforward neural network that trains
        via backpropagation.

    This network takes inputs in the form of matrices and vectors. Input matrices
    are applied row by row to create a single input vector. Outputs are returned
    as a vector, with the vector index representing the output for the output
    node that shares said index.

    To ensure proper calculation, it is encouraged to normalise all input values
    in relation to one another.

    Attributes:
        training_sets: list which stores all training matrices with their
                        attached solution tuple
        testing_sets: list which stores all testing matrices, which do not
                        include solution tuples
        output: vector whose indices represent each output node for the network
        testing_output: vector whose indices represent each output node, but
                        is only applicable after testing has occurred

        input_vector:   n x 1 vector with each index representing an input node
        input_weights:  n x m matrix with index (n, m) representing the weight
                        on the path between input node n and hidden node m
        hidden_vector:  m x 1 vector with each index representing a hidden node
        hidden_weights: m x z matrix with index (m, z) representing the weight
                        on the path between hidden node m and the output node z
    """

    training_sets = []
    output = []
    testing_sets = []
    testing_output = []

    def __init__(self, num_inputs, num_hidden_nodes, hidden_depth, num_outputs, target_tsse, eta):
        """ Initialises all layers and weights for the neural network.

        Initialises all layers of the neural network. This initialises the
        input vector, the input to hidden weights, the hidden vector, the
        bias (with bias weights) for each hidden node, the hidden to output
        weights, the output vector, and a bias (with bias weight) for each
        output node.

        All node vectors are initialised to 0. All weights, including bias
        weights, are randomly initialised with values between -1 and 1. All
        biases are initialised to 1 - these can be changed to influence
        learning away from direct calculation.

        :param num_inputs: The number of elements in each input pattern
        :param num_hidden_nodes: Used for calculation - set as desired.
        :param hidden_depth: Currently only supports hidden_depth=1
        :param num_outputs: Number of desired categories
        :param target_tsse: Target maximal error (sum squared error)
        :param eta: Effects training rate - higher is faster.
        """

        self.input_vector = np.zeros([num_inputs, 1])
        self.input_weights = 2 * np.random.rand(num_inputs, num_hidden_nodes) - 1

        self.hidden_vector = np.zeros([num_hidden_nodes, hidden_depth])
        self.hidden_weights = 2 * np.random.rand(num_hidden_nodes, num_outputs) - 1
        self.hidden_bias = np.ones([num_hidden_nodes, hidden_depth])
        self.hidden_bias_weights = 2 * np.random.rand(num_hidden_nodes, hidden_depth) - 1

        self.target_output_vector = np.zeros([num_outputs, 1])
        self.output_bias = np.ones([num_outputs, 1])
        self.output_bias_weights = 2 * np.random.rand(num_outputs, 1) - 1

        self.target_tsse = target_tsse
        self.tsse = 0
        # self.target_tsse = self.rmse_to_tsse(target_rmse, num_outputs)
        self.eta = eta

    def rmse_to_tsse(self, target_rmse, num_outputs):
        """ Converts root mean squared error to total sum squared error.

        RMSE gives the allowed deviation from the target value in decimal form.
        For example, if the target output value if 0.9 and one considers 0.89 to
        be acceptable, a RMSE of 0.01 would be required. This results in a TSSE
        value of 0.002.

        Root mean squared error is given by 2 * TSSE over the number of patterns
        times the number of outputs. The square root of this calculation gives
        the RMSE calculation of the given network with respect to the provided
        TSSE value.

        :param target_rmse: Amount of allowed deviation from desired output
        :param num_outputs: Number of output nodes
        :return: Total Sum Squared Error for the given Root Mean Squared Error
        """

        tsse = ((target_rmse**2) * len(self.training_sets
                        * num_outputs))/2
        return tsse

    def train(self, epochs):
        """ Handles entire training loop - both feedforward and backpropagation.

        Works by matrix and vector multiplication working up from the initial
        input layer to the outputs. The errors are then pushed back along the
        same pathways, training the network to the input data sets. These
        features are expounded upon in the documentation for the feedforward
        and backpropagation methods, respectively.

        Each iteration causes a shuffle of the data, preventing the network
        from learning the patterns in a given order as well as stifling dreams.

        :param epochs: Minimum number of epochs - can go past if TSSE is too high
        :return:
        """

        current_epoch = 0

        while current_epoch < epochs or self.tsse > self.target_tsse:
            self.output.clear()
            np.random.shuffle(self.training_sets)

            for elem in self.training_sets:
                for i in range(len(elem[0])):
                    for j in range(len(elem[0][i])):
                        self.input_vector[i*len(elem[0][i])+j] = elem[0][i][j]

                # check if single element - isinstance or put comma at end inside tuple
                for i in range(len(elem[1])):
                    self.target_output_vector[i] = float(elem[1][i])

                out = self.feedforward()
                self.output.append((elem[0], out))
                self.backpropagate(out)

            if current_epoch % 250 == 0 and current_epoch != 0:
                self.tsse = self.calc_tsse()
                print(current_epoch, ": ", self.tsse)
                # print(self.output)

                # if current_epoch > epochs:
                #     print(self.input_weights)
                #     print(self.hidden_weights)

            current_epoch += 1

        print("Final Epoch: {}\n".format(current_epoch))

    def feedforward(self):
        """ Calculates output values by pushing the input data through the network.

        Output calculation is done through a series of matrix and vector
        multiplications. The neural signal for each hidden node is calculated
        using a sigmoid function. Steps are as follows:

            1. Input vector is multiplied with the input-to-hidden weight
                matrix - this gives the input to our hidden-layer nodes.

            2. Each hidden-layer node input is run through a sigmoid calculation
                to determine neuron activation. This vector becomes our
                hidden-layer node output, with each component corresponding
                to a different hidden node.

            3. This hidden-layer node activation signal vector is then
                multiplied through with the hidden-to-output weight matrix. This
                gives us the input to our output-layer nodes.

            4. As with the hidden-layer, each component of the resulting vector
                is run through a sigmoid calculation. The result of this
                calculation is the final output for that node along that path.

        :return:
        """

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

    def backpropagate(self, output_vector):
        """ How the network assesses its output and learns from it.

        Error signals are calculated beginning at the output-layer nodes. These
        errors tell the deviation between the desired and the actual outputs.

        :param output_vector:
        :return:
        """

        # Calculating output-layer error signals
        output_error = np.zeros((output_vector.shape[0], 1))
        for i in range(output_vector.shape[0]):
            output_error[i] = (self.target_output_vector[i] - output_vector[i])\
                              * output_vector[i] * (1 - output_vector[i])

        # Calculating and adjusting output-layer bias weights
        for i in range(self.output_bias_weights.shape[0]):
            self.output_bias_weights[i] += self.eta * output_error[i] * self.output_bias[i]

        # Calculating hidden-layer error signals
        output_hidden_dot = np.dot(self.hidden_weights, output_error)
        hidden_error = np.zeros((self.hidden_vector.shape[0], 1))
        for i in range(self.hidden_vector.shape[0]):
            hidden_error[i] = self.hidden_vector[i] \
                              * (1 - self.hidden_vector[i]) \
                              * (output_hidden_dot.sum(axis=1)[i])
                            # ^ Sum of each row of that dot product
                            # each row corresponds to all output nodes for a given hidden node

        # Calculating and adjusting hidden-layer bias weights
        for i in range(self.hidden_bias_weights.shape[0]):
            self.hidden_bias_weights[i] += self.eta *\
                                           hidden_error[i] *\
                                           self.hidden_bias[i]

        # Calculating and adjusting output to hidden-layer weights
        for i in range(self.hidden_weights.shape[0]):
            for j in range(self.hidden_weights.shape[1]):
                self.hidden_weights[i][j] += self.eta\
                                             * output_error[j]\
                                             * self.hidden_vector[i]

        # Calculating and adjusting hidden to input-layer weights
        for i in range(self.input_weights.shape[0]):
            for j in range(self.input_weights.shape[1]):
                self.input_weights[i][j] += self.eta \
                                            * hidden_error[j] \
                                            * self.input_vector[i]

    def calc_hidden_output(self, hidden_index, input_weight_dot):
        """ Sigmoid function for determining node output.

        :param hidden_index: Hidden node to calculate output signal for
        :param input_weight_dot: Pre-sigmoid node input
        :return:
        """

        netj = (self.hidden_bias[hidden_index] *
                self.hidden_bias_weights[hidden_index]) +\
                input_weight_dot[hidden_index]
        node_output = (1 / (1 + (np.exp(-1 * netj))))
        return node_output

    def calc_output(self, output_index, hidden_weight_dot):
        """ Sigmoid function for calculating network final output.

        :param output_index: Output node to calculate output signal for
        :param hidden_weight_dot: Hidden-layer weight dot matrix
        :return:
        """

        netj = (self.output_bias[output_index] *
                self.output_bias_weights[output_index]) + \
               hidden_weight_dot[output_index]
        node_output = (1 / (1 + (np.exp(-1 * netj))))
        return node_output

    def calc_tsse(self):
        """ Calculates total sum squared error for the network.

        This calculation is given by taking the sum for each pattern of the sum
        of the desired minus the actual final outputs squared for each output
        node. This is then multiplied by 1/2.

        :return: Total Sum Squared Error
        """

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

    def test(self):
        """ Loads testing data and runs through the networh in feedforward mode
            only. No learning is permitted in testing mode.

        :return:
        """

        for elem in self.testing_sets:
            for i in range(len(elem[0])):
                for j in range(len(elem[0][i])):
                    self.input_vector[i * len(elem[0][i]) + j] = elem[0][i][j]

            test_out = self.feedforward()
            self.testing_output.append((elem[0], test_out))

    def load_training_data(self, matrix):
        """ Loads training data.

        :param matrix: A matrix of patterns with attached solution tuple.
        :return:
        """
        self.training_sets.clear()
        for elem in matrix:
            self.training_sets.append(elem)

    def load_testing_data(self, matrix):
        """ Loads testing data.

        :param matrix: Matrix of patterns without an attached solution tuple.
        :return:
        """
        self.testing_sets.clear()
        for elem in matrix:
            self.testing_sets.append(elem)

    def print_output(self, testing=False):
        """ Prints pattern number, matrix, and calculated output.

        :param testing: If true, testing data is printed. Else, training data.
        :return:
        """

        if testing == False:
            for pat in range(len(self.output)):
                print("Pattern: {}".format(pat), end="")
                for row in self.output[pat][0]:
                    print(row)
                print('\nOutput:')
                for pat in self.output[pat][1]:
                    print(pat)
                print("\n")
        else:
            for pat in range(len(self.testing_output)):
                print("Pattern: {}\n".format(pat))
                for row in self.testing_output[pat][0]:
                    print(row)

                # TODO make print pretty
                # print('\nActual\t\tExpected\n')
                # for i in range(self.testing_output[pat][1]):
                #     print("{}\t{}\n".format(self.testing_output[i][1],
                #                             self.target_output_vector[i]))

                print('\nOutput:')
                for pat in self.testing_output[pat][1]:
                    print(pat)
                print("\n")
                print("Expected: ")
                for row in self.target_output_vector:
                    print(row)

    def write_output(self, file, testing=True):
        """ Writes output to a text file for easy manipulation.

        :param file: file name of desired file.
        :param testing: If true, testing data. Else, training data.
        :return:
        """

        with open(file, "w") as f:
            if testing is True:
                for elem in self.testing_output:
                    in1 = elem[0][0]
                    in2 = elem[0][1]
                    out = elem[1]

                    output_string = str(in1) + ", " + str(in2) + ", " + str(out) + "\n"
                    f.write(output_string)


if __name__ == '__main__':
    FFNN = FeedForwardNetwork(25, 25, 1, 6, .02, 1)

    training_data = [  # Cross
        (([.1, .1, .9, .1, .1],
          [.1, .1, .9, .1, .1],
          [.9, .9, .9, .9, .9],
          [.1, .1, .9, .1, .1],
          [.1, .1, .9, .1, .1]),
         (.9, .1, .1, .1, .1, .1)),

        # Dash
        (([.1, .1, .1, .1, .1],
          [.1, .1, .1, .1, .1],
          [.9, .9, .9, .9, .9],
          [.1, .1, .1, .1, .1],
          [.1, .1, .1, .1, .1]),
         (.1, .9, .1, .1, .1, .1)),

        # Backslash
        (([.9, .1, .1, .1, .1],
          [.1, .9, .1, .1, .1],
          [.1, .1, .9, .1, .1],
          [.1, .1, .1, .9, .1],
          [.1, .1, .1, .1, .9]),
         (.1, .1, .9, .1, .1, .1)),

        # Forward Slash
        (([.1, .1, .1, .1, .9],
          [.1, .1, .1, .9, .1],
          [.1, .1, .9, .1, .1],
          [.1, .9, .1, .1, .1],
          [.9, .1, .1, .1, .1]),
         (.1, .1, .1, .9, .1, .1)),

        # X
        (([.9, .1, .1, .1, .9],
          [.1, .9, .1, .9, .1],
          [.1, .1, .9, .1, .1],
          [.1, .9, .1, .9, .1],
          [.9, .1, .1, .1, .9]),
         (.1, .1, .1, .1, .9, .1)),

        # Vertical Line
        (([.1, .1, .9, .1, .1],
          [.1, .1, .9, .1, .1],
          [.1, .1, .9, .1, .1],
          [.1, .1, .9, .1, .1],
          [.1, .1, .9, .1, .1]),
         (.1, .1, .1, .1, .1, .9))]

    testing_data = [  # Cross
        (([.1, .1, .9, .1, .1],
          [.1, .1, .9, .1, .1],
          [.9, .9, .1, .9, .9],
          [.1, .1, .9, .1, .1],
          [.1, .1, .9, .1, .1]),
         (.9, .1, .1, .1, .1, .1)),

        # Dash
        (([.1, .1, .1, .1, .1],
          [.1, .1, .9, .1, .1],
          [.9, .9, .1, .9, .9],
          [.1, .1, .1, .1, .1],
          [.1, .1, .1, .1, .1]),
         (.1, .9, .1, .1, .1, .1)),

        # Backslash
        (([.9, .1, .1, .1, .1],
          [.1, .9, .1, .1, .1],
          [.1, .1, .1, .9, .1],
          [.1, .1, .1, .9, .1],
          [.1, .1, .1, .1, .9]),
         (.1, .1, .9, .1, .1, .1)),

        # Forward Slash
        (([.1, .1, .1, .1, .9],
          [.1, .9, .1, .9, .1],
          [.1, .1, .1, .1, .1],
          [.1, .9, .1, .1, .1],
          [.9, .1, .1, .1, .1]),
         (.1, .1, .1, .9, .1, .1)),

        # X
        (([.9, .1, .1, .1, .9],
          [.1, .9, .1, .9, .1],
          [.1, .1, .1, .1, .1],
          [.1, .9, .1, .9, .1],
          [.9, .1, .1, .1, .9]),
         (.1, .1, .1, .1, .9, .1)),

        # Vertical Line
        (([.1, .1, .9, .1, .1],
          [.1, .1, .9, .1, .1],
          [.1, .1, .1, .1, .1],
          [.1, .1, .9, .1, .1],
          [.1, .1, .9, .1, .1]),
         (.1, .1, .1, .1, .1, .9))]

    FFNN.load_training_data(training_data)
    for pattern in range(len(training_data)):
        print("Pattern {}:".format(pattern))
        for row in training_data[pattern][0]:
            print(row)
        print('\n')

    FFNN.train(100)
    # FFNN.print_output()

    FFNN.load_testing_data(testing_data)
    FFNN.test()
    FFNN.print_output(testing=True)
