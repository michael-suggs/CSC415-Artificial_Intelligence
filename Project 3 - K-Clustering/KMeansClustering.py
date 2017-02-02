import numpy as np

__author__ = "Michael Suggs // mjs3607@uncw.edu"


class KMeansClustering:

    def __init__(self, k, data):
        """ Initialises classifier by giving it an arbitrary number of
         pre-defined initial classes.

        :param k: Total number of classification clusters to create
        :param data: Array containing all points represented as lists [[x1,y1],]
        """
        self.sets = [[np.random.rand(2, 1), []] for i in range(k)]
        self.data = [np.asarray(pt) for pt in data]
        self.centroids = self.generate_centroids(k)

        self.__classify_data()

    def generate_centroids(self, k):
        # Find centre
        # Find k points that are farthest from the centre
        for i in range(k):
            pass

    def __classify_data(self):
        for pt in self.data:
            closest = (100, len(self.sets)*2)
            for s in range(len(self.sets)):
                dist = np.abs(np.linalg.norm(pt - s[0]))
                if dist < closest[0]:
                    closest = (dist, s)

        # TODO instead, sort self.sets by mean distance and use binary search

    def calculate_means(self):
        pass

    def classify_point(self, point):
        pass


if __name__ == '__main__':
    kmeans = KMeansClustering(5)