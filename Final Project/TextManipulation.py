from random import shuffle

__author__ = "Michael J. Suggs // mjs3607@uncw.edu"

if __name__ == '__main__':
    new_file = []

    # # status is element 17 (in column 17 with names occupying column 0)
    # with open('parkinsons.data', 'r') as f:
    #     next(f)
    #     for line in f:
    #         l = line.strip().split(',')
    #         new_file.append(l[1:17] + l[18:] + list(l[17]) + list('\n'))
    #
    # with open('new_parkinsons.data', 'w') as f:
    #     for line in new_file:
    #         f.write(" ".join(str(num) for num in line))

    with open('park_train.data', 'r') as f:
        for line in f:
            new_file.append(line)

    for i in range(10):
        shuffle(new_file)

    with open('park_train.data', 'w') as f:
        for line in new_file:
            f.write(line)

    with open('park_test.data', 'r') as f:
        for line in f:
            new_file.append(line)

    for i in range(10):
        shuffle(new_file)

    with open('park_test.data', 'w') as f:
        for line in new_file:
            f.write(line)
