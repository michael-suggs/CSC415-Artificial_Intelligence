import numpy as np
__author__ = "Michael Suggs // mjs3607@uncw.edu"


class FeedForwardNetwork:
    training_sets = []
    output = []
    testing_sets = []
    testing_output = []

    """
    """
    def __init__(self, num_inputs, num_hidden_nodes, hidden_depth, num_outputs, target_tsse, eta):
        self.input_vector = np.zeros([num_inputs, 1])
        self.input_weights = 2 * np.random.rand(num_inputs, num_hidden_nodes) - 1

        self.hidden_vector = np.zeros([num_hidden_nodes, hidden_depth])
        self.hidden_weights = 2 * np.random.rand(num_hidden_nodes, hidden_depth) - 1
        self.hidden_bias = np.ones([num_hidden_nodes, hidden_depth])
        self.hidden_bias_weights = 2 * np.random.rand(num_hidden_nodes, hidden_depth) - 1

        self.output_vector = np.zeros([num_outputs, 1])
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
                self.input_vector[0] = elem[0][0]
                self.input_vector[1] = elem[0][1]
                target_output = float(elem[1])

                out = self.feedforward()
                self.output.append(((self.input_vector[0].item(0), self.input_vector[1].item(0)), out.item(0)))
                self.backpropagate(out, target_output)

            if current_epoch % 500 == 0 and current_epoch != 0:
                self.tsse = self.calc_tsse()
                print(current_epoch, ": ", self.tsse)
                print(self.output)

                if current_epoch > epochs:
                    print(self.input_weights)
                    print(self.hidden_weights)

            current_epoch += 1

    """
    """
    def feedforward(self):
        input_transpose = self.input_vector.transpose()
        input_weight_dot = input_transpose.dot(self.input_weights).T

        # calculates the output of each hidden node
        for i in range(self.hidden_vector.shape[0]):
            self.hidden_vector[i] = self.calc_hidden_output(i, input_weight_dot)

        hidden_transpose = self.hidden_vector.transpose()
        hidden_weight_dot = hidden_transpose.dot(self.hidden_weights).T

        # calculates the final output of the network
        netj = (self.output_bias[0] * self.output_bias_weights[0]) + hidden_weight_dot
        final_out = (1 / (1 + (np.exp(-1 * netj))))

        return final_out

    """
    """
    def backpropagate(self, output, target_output):
        output_error = (target_output - output.item(0)) * output.item(0) * (1 - output.item(0))
        for i in range(self.output_bias_weights.shape[0]):
            self.output_bias_weights[i] += self.eta * output_error * self.output_bias[i]

        hidden_error = np.zeros((self.hidden_vector.shape[0], 1))
        for i in range(self.hidden_vector.shape[0]):
            hidden_error[i] = self.hidden_vector[i] \
                              * (1 - self.hidden_vector[i]) \
                              * (output_error * self.hidden_weights[i])

        for i in range(self.hidden_bias_weights.shape[0]):
            self.hidden_bias_weights[i] += self.eta *\
                                           hidden_error[i] *\
                                           self.hidden_bias[i]

        # compute weight adjustments
        for i in range(self.hidden_weights.shape[0]):
            self.hidden_weights[i] += self.eta * output_error * self.hidden_vector[i]

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
    def calc_tsse(self):
        total_sum_squares = 0

        for elem in self.training_sets:
            self.input_vector[0] = elem[0][0]
            self.input_vector[1] = elem[0][1]
            out = self.feedforward()
            total_sum_squares += (elem[1] - out)**2

        return .5 * total_sum_squares

    """
    """
    def test(self):
        for i in range(len(self.testing_sets)):
            self.input_vector[0] = self.testing_sets[i][0]
            self.input_vector[1] = self.testing_sets[i][1]
            test_out = self.feedforward()
            self.testing_output.append(((self.testing_sets[i]), test_out.item(0)))

    """
    """
    def load_training_data(self, file):
        with open(file) as f:
            for line in f:
                in1, in2, out = line.split(", ")
                input_tuple = (float(in1), float(in2))
                training_tuple = (input_tuple, float(out))
                self.training_sets.append(training_tuple)

    """
    """
    def load_testing_data(self, file):
        with open(file) as f:
            for line in f:
                in1, in2 = line.split(", ")
                input_tuple = (float(in1), float(in2))
                self.testing_sets.append(input_tuple)

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



if __name__ == '__main__':
    FFNN = FeedForwardNetwork(2, 2, 1, 1, .0002, .5)
    FFNN.load_training_data("XOR_testing_data.txt")
    FFNN.train(10000)
    FFNN.print_output()

    # FFNN.load_testing_data("reduced_test.txt")
    FFNN.load_testing_data("testing_data.txt")
    FFNN.test()
    # FFNN.print_output(testing=True)
    FFNN.write_output("testing_output.txt")
