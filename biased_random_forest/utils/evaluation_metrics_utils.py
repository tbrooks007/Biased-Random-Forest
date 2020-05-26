from random import randrange


def cross_validation_split(dataset, n_folds):
    """

    :param dataset:
    :param n_folds:
    :return:
    """

    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


def accuracy_metric(actual, predicted):
    """

    :param actual:
    :param predicted:
    :return:
    """

    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


def evaluate_algorithm(dataset, algorithm, total_forest_size, max_label_set, min_label_set, n_folds):
    """
    Evaluate model performance using k-cross fold validation.
    """

    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    trees = None

    for fold in folds:
        # split training set
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])

        # split test set
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
            trees = algorithm.fit(train_set, total_forest_size, max_label_set, min_label_set)

        predicted = [algorithm.bagging_predict(trees, row) for row in test_set]
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)

    return scores
