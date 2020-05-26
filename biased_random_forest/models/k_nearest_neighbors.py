# Note, kNN implementation  by
# Jason Brownlee in https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/

from math import sqrt


def dataset_minmax(dataset):
    """
    Find the min and max values for each column
    :param dataset:
    :return:
    """

    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax


def euclidean_distance(row1, row2):
    """
    Calculate the Euclidean distance between two vectors
    :param row1:
    :param row2:
    :return:
    """
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i]) ** 2
    return sqrt(distance)


def get_neighbors(test_row, train, num_neighbors):
    """
    Locate the most similar neighbors
    :param train:
    :param test_row:
    :param num_neighbors:
    :return:
    """
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()

    for i in range(num_neighbors):
        neighbors.append(distances[i][0])

    return neighbors
