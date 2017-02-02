import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from collections import Counter
from random import choice

__author__ = "Michael Suggs // mjs3607@uncw.edu"

# TODO Voronoi diagrams for k-NN

class ExhaustiveKNN2D:

    def __init__(self, filename):
        """ Initialises classifier by giving it an arbitrary number of
            pre-defined initial classes.

        :param args: Each arg is an array containing vectors representing each
                        point for that classification, where each arg array
                        represents a different classification.
        """
        # self.data = read_file(filename)
        self.data = read_file(filename)
        self.tree = KDTree([i for i in self.data.keys()])

    def kd_classify(self, point, k):
        # TODO can't use KDTree since it doesn't support updates
        neighbours = self.tree.query(point, k=k)
        # print("Neighbours = {}".format(neighbours))
        neighbours = [self.data[i] for i in neighbours]
        _class, count = Counter(neighbours).most_common(1)
        print("{} neighbours belonged to set {}\n".format(count, _class))
        print("Placing point {} in set {}\n".format(point, _class))
        self.data[point] = _class

    def classify(self, point, k):
        min_dist = []
        for i in range(k):
            _c = choice(list(self.data.keys()))
            min_dist.append((_c, self.distance(point, _c)))
        min_dist.sort()
        # print("Initial minimum distances: {}".format(min_dist))

        for p in self.data.values():
            dist = self.distance(point, p)
            for i in range(len(min_dist)):
                if dist < min_dist[i][1]:
                    min_dist[i][0] = p
                    min_dist[i][1] = dist

        min_dist = [self.data[i[0]] for i in min_dist]
        _class, count = Counter(min_dist).most_common(1)[0]
        print("{} neighbours belonged to set {}\n".format(count, _class))
        print("Placing point {} in set {}\n".format(point, _class))
        self.data[point] = _class

    def distance(self, p1, p2):
        return np.linalg.norm(np.asarray(p2) - np.asarray(p1))


def read_file(filename):
    dataset = {}
    with open(filename, 'r') as f:
        for line in f:
            line_list = tuple(map(float, line.strip().split(',')))
            dataset[line_list[:-1]] = int(line_list[-1])
    return dataset

if __name__ == '__main__':
    # TODO Needs training & testing since you have to have neighbour points to classify from
    knn = ExhaustiveKNN2D('seed_training.csv')
    testing_data = read_file('seed_testing.csv')

    for point in testing_data:
        knn.classify(point, 5)
