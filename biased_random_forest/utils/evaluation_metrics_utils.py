from random import randrange
import logging

logging.basicConfig(level=logging.INFO)


def cross_validation_split(dataset, n_folds):
    """
    Performs cross validation split of data set.
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
    Calculate accuracy metric.
    :param actual:
    :param predicted:
    :return: float, score
    """

    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


def calculate_classifier_eval_metrics(actual, predicted):
    """
    Calculate classification evaluation metrics.
    :param actual:
    :param predicted:
    :return:
    """

    TP = 0
    TN = 0
    FN = 0
    FP = 0

    for i in range(len(actual)):
        if actual[i] and actual[i] == predicted[i]:
            TP += 1
        elif not actual[i] and actual[i] == predicted[i]:
            TN += 1
        elif actual[i] and not predicted[i]:
            FN += 1
        elif not actual[i] and predicted[i]:
            FP += 1

    return TP, TN, FN, FP

def evaluate_algorithm(dataset, algorithm, total_forest_size, max_label_set, min_label_set, n_folds):
    """
    Evaluate model performance using k-cross fold validation.
    :param dataset:
    :param algorithm:
    :param total_forest_size:
    :param max_label_set:
    :param min_label_set:
    :param n_folds:
    :return: list of calculated eval metrics
    """

    trees = None
    fold_idx = 0
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    folds = cross_validation_split(dataset, n_folds)
    scores = list()

    for fold in folds:
        # split training set
        train_set = list(folds)
        train_set.pop(fold_idx)
        train_set = sum(train_set, [])

        # split test set
        test_set = list()

        logging.info("Now training and evaluating fold {} now...".format(fold_idx))
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None

        trees = algorithm.fit(train_set, total_forest_size, max_label_set, min_label_set)

        logging.info("Evaluating model for fold {} now (# of trees in forest: {})...".format(fold_idx, len(trees)))
        predicted = [algorithm.bagging_predict(trees, row) for row in test_set]
        actual = [row[-1] for row in fold]

        # calculate accuraccy
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)

        # calculate classifier evaluation metrics
        curr_tp, curr_tn, curr_fn, curr_fp = calculate_classifier_eval_metrics(actual, predicted)
        TP += curr_tp
        TN += curr_tn
        FN += curr_fn
        FP += curr_fp

        fold_idx += 1

    # calculate P/R
    precision = TP/(TP + FP)
    recall = TP/(TP + FN)

    return scores, precision, recall
