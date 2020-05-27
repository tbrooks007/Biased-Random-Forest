from random import randrange
from biased_random_forest.utils.preprocess import train_test_split
import logging

logging.basicConfig(level=logging.INFO)


def generate_validation_folds(dataset, k_folds):
    """
    Performs cross validation split of data set.
    :param dataset:
    :param k_folds:
    :return: list of folds for a given dataset
    """

    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / k_folds)
    for i in range(k_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


def calculate_accuracy(actual, predicted):
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


def _generate_fold(folds, fold_idx):

    train_set = None
    train_set = None

    for fold in folds:
        # split training set
        train_set = list(folds)
        train_set.pop(fold_idx)
        train_set = sum(train_set, [])

        # split test set
        test_set = list()

        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None

        yield train_set, test_set, fold


def _predict(X_test, algorithm, model, fold):
    """
    Predict labels for test set for a gien
    :param X_test:
    :param algorithm:
    :param model:
    :param fold:
    :return:
    """

    predicted = [algorithm.bagging_predict(model, row) for row in X_test]
    actual = [row[-1] for row in fold]

    return predicted, actual


def calculate_precision_recall(true_positives, false_positives, false_negatives):
    """
    Calculates precision and recall metrics
    :param true_positives:
    :param false_positives:
    :param false_negatives:
    :return:
    """

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    return precision, recall


def evaluate(df, algorithm, total_forest_size, max_label_set, min_label_set, n_folds):
    """
    Evaluate model performance using k-cross fold validation.
    :param df:
    :param algorithm:
    :param total_forest_size:
    :param max_label_set:
    :param min_label_set:
    :param n_folds:
    :return: list of calculated eval metrics
    """

    scores = list()
    fold_idx = 0
    true_positives = 0
    true_negatives = 0
    false_negatives = 0
    false_positives = 0

    # convert dataframe to numpy array for easy of use
    dataset = df.to_numpy()
    folds = generate_validation_folds(dataset, n_folds)

    logging.info("Now training and evaluating fold {} now...".format(fold_idx))
    for train_set, test_set, fold in _generate_fold(folds, fold_idx):

        # train model
        trees = algorithm.fit(train_set, total_forest_size, max_label_set, min_label_set)

        # run inference
        logging.info("Evaluating model for fold {} now (# of trees in forest: {})...".format(fold_idx, len(trees)))
        predicted, actual = _predict(test_set, algorithm, trees, fold)

        # calculate accuracy for current model
        accuracy = calculate_accuracy(actual, predicted)
        scores.append(accuracy)

        # calculate classifier evaluation metrics
        curr_tp, curr_tn, curr_fn, curr_fp = calculate_classifier_eval_metrics(actual, predicted)
        true_positives += curr_tp
        true_negatives += curr_tn
        false_negatives += curr_fn
        false_positives += curr_fp

        fold_idx = fold_idx + 1

    # calculate evaluation metrics
    precision, recall = calculate_precision_recall(true_positives, false_positives, false_negatives)

    # TODO: calculate PRC and ROC + flush to disk

    return scores, precision, recall
