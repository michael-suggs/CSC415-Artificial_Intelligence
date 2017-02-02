from sklearn import svm
from sklearn.model_selection import LeaveOneOut
import time

__author__ = "Michael J. Suggs // mjs3607@uncw.edu"

def read_file(filename):
    data = []
    y_data = []
    with open(filename, "r") as f:
        for line in f:
            line_data = list(line.strip().split(','))
            line_data[0] = sum([ord(c) for c in line_data[0]])
            y = line_data[1]
            del line_data[1]
            data.append([float(i) for i in line_data])
            y_data.append(float(y))
    return data, y_data

if __name__ == '__main__':
    loo = LeaveOneOut()
    vm = svm.SVC()

    X, Y = read_file('ballex.csv')
    svmans = []

    start = time.clock()
    for train, test in loo.split(X):
        runstart = time.clock()
        x_data = [X[i] for i in train]
        y_data = [Y[i] for i in train]
        vm.fit(x_data, y_data)
        ans = vm.predict([X[i] for i in test])
        print("Run Time: {}".format(time.clock() - runstart))
        svmans.append(ans)

    print("Total Time: {}".format(time.clock() - start))
    for ans in svmans:
        print(ans)