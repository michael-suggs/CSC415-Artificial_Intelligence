from random import choice, randint, uniform
__author__ = "Michael Suggs // mjs3607@uncw.edu"


class MarkovChain:
    """

    Given an input set of data, intelligently determines grammar rules for said data
        based on locational letter frequency (relative probability that one letter
        or symbol will succeed another). After analysis of the language is complete,
        additional words based on said interpreted grammar rules may be generated.

    Input Types:
        List
        String

    """

    # 2D stochastic (probability) matrix
    stochastic_matrix = []

    def __init__(self, input_document):
        self.input_list = input_document
        self.generate_probabilities()

    def stringify_input(self):
        """
        If a list-type object is passed as the "input_document" parameter, this
            method is called to place all elements of said list into a
            space-delimited contiguous string, with initial and final spaces
            occurring at the beginning and terminus of said contiguous string.
        """
        if isinstance(self.input_list, list):
            markov_string = " "
            for elem in self.input_list:
                # Space separated list with leading and trailing space
                markov_string += elem + " "
            markov_string += " "

            return markov_string

        # Ensures a space is placed at both the front and end of the string
        elif isinstance(self.input_list, str):
            if self.input_list[:1] != " " and self.input_list[-1:] != " ":
                markov_string = " " + self.input_list + " "
            elif self.input_list[:1] != " ":
                markov_string = " " + self.input_list
            elif self.input_list[-1:] != " ":
                markov_string = self.input_list + " "

            return markov_string

    def generate_probabilities(self):
        """
        Generates, based on letter location and occurrance, the probability that
            a given letter will be proceeded by another letter in the language.
            Generates probabilities for all characters found in the language,
            including START and STOP symbols, represented by spaces.
        """

        markov_string = self.stringify_input()
        self.unique_chars = list(set(markov_string))
        self.unique_chars.sort()
        char_max_dict = {k:markov_string.count(k) for k in self.unique_chars}

        # matrix[i][j] tells likelihood of i being followed by j
        self.stochastic_matrix = [[0 for i in range(len(self.unique_chars))]
                                  for j in range(len(self.unique_chars))]

        for i in range(len(markov_string)):
            initial_char = markov_string[i:i+1]
            next_char = markov_string[i+1:i+2]

            if initial_char == ' ' and next_char == ' ':
                break

            # Adds 1 divided by the maximum occurrences of symbol initial_char
            # to the stochastic_matrix location representing initial_char
            # followed by next_char
            self.stochastic_matrix[self.unique_chars.index(initial_char)][
                self.unique_chars.index(next_char)] += (1/char_max_dict[initial_char])

    def generate_words(self, num_words):
        """

        :param num_words:
        :param cut_off_probability:
        :return:
        """
        generated_word_list = []

        for i in range(num_words):
            word = self.probability_choice(self.stochastic_matrix[self.unique_chars.index(' ')])
            generated_word_list.append(self.recursive_word_gen(word))

        return generated_word_list

    def recursive_word_gen(self, word):
        """

        :param word:
        :return:
        """
        # if self.stochastic_matrix[self.unique_chars.index(word[-1])][
        #         self.unique_chars.index(' ')] > 0 and (randint(0, 100000) % 4) == 0:
        if word[-1] == ' ':
            return word[0:len(word)-1]

        else:
            last_char = self.unique_chars.index(word[-1])
            next_char = self.probability_choice(self.stochastic_matrix[last_char])
            return self.recursive_word_gen(word + next_char)

    def probability_choice(self, weight_row):
        """

        :param weight_row:
        :return:
        """
        weight_sum = sum(weight_row)
        threshold = uniform(0, weight_sum)

        row_dict = dict(zip(self.unique_chars, weight_row))
        current = 0

        for letter, probability in row_dict.items():
            if probability == 0:
                continue
            elif current + probability >= threshold:
                return letter
            else:
                current += probability


    def display_stochastic_matrix(self):
        """
        Prints rows of the stochastic matrix, with symbols denoting each row/column
        """

        print("     ", self.unique_chars, "\r")
        for j in range(len(self.stochastic_matrix)):
            print([self.unique_chars[j]], end=" ")
            print(self.stochastic_matrix[j])


if __name__ == '__main__':
    input_language = ["spare", "spear", "pares", "peers", "reaps", "peaks", "speaker",
                      "keeper", "pester", "paste", "tapas", "pasta", "past", "straps",
                      "tears", "terse", "steer", "street", "stare", "rates", "streak",
                      "taste", "tapa", "peat", "eat", "ate", "tea", "seat"]

    markov = MarkovChain(input_language)
    markov.display_stochastic_matrix()
    print(markov.generate_words(100))
