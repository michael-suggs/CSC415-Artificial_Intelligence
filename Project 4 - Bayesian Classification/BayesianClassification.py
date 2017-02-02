import os
from sklearn import naive_bayes, datasets
from collections import Counter

__author__ = "Michael J. Suggs // mjs3607@uncw.edu"


def load_data(filename, training=True):
    if training is True:
        ans = []
        data = []
        with open(os.path.join('../Project 3 - K-Clustering/', filename)) as f:
            for line in f:
                line = [float(i) for i in line.split(',')]
                ans.append(int(line[-1]))
                # del line[-1]
                data.append(line)
        return {'ans': ans, 'data': data}

    else:
        data = []
        with open(os.path.join('../Project 3 - K-Clustering/', filename)) as f:
            for line in f:
                line = [float(i) for i in line.split(',')]
                # del line[-1]
                data.append(line)
        return data


if __name__ == '__main__':
    # print(datasets.load_iris())
    data = load_data('seed_training.csv')
    bayes = naive_bayes.MultinomialNB()
    bayes.fit(data['data'], data['ans'])

    # bayes.score(data['data'], data['ans'])

    test_data = load_data('seed_testing.csv', training=False)
    bayes_pred = bayes.predict(test_data)
    results = Counter([1 if (bayes_pred[i] == data['ans'][i]) else 0 for i in range(len(bayes_pred))])
    print("Correctly Classified: {}\nMisclassified: {}".format(results[1], results[0]))

