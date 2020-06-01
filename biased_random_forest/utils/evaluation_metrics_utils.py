from random import randrange
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from biased_random_forest.utils.preprocess import split_data_from_target_columns
import numpy as np
import matplotlib.pylab as plt
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


def _generate_fold(folds):
    """
    Generate test and train set from folds
    :param folds:
    :return:
    """

    fold_idx = 0

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

        # yield train_set, test_set, fold_idx, fold
        yield train_set, test_set, fold
        fold_idx = fold_idx + 1


def predict(x_test, algorithm, model, fold):
    """
    Predict labels for test set for a given fold
    :param x_test:
    :param algorithm:
    :param model:
    :param fold:
    :return:
    """

    predicted = [algorithm.bagging_predict(model, row) for row in x_test]
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


def plot_roc_auc(plt, true_positive_rates, mean_false_positive_rate, title="ROC"):
    """
    Plot and save ROC AUC curves.
    :param plt:
    :param true_positive_rates:
    :param mean_false_positive_rate:
    :param title:
    :return: matplotlib plot object
    """

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black')
    mean_true_positive_rate = np.mean(true_positive_rates, axis=0)
    mean_auc = auc(mean_false_positive_rate, mean_true_positive_rate)
    plt.plot(mean_false_positive_rate, mean_true_positive_rate, color='blue', label=r'Mean ROC (AUC = %0.2f )' % (mean_auc), lw=2, alpha=1)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.text(0.32, 0.7, 'More accurate area', fontsize=12)
    plt.text(0.63, 0.4, 'Less accurate area', fontsize=12)

    return plt


def evaluate(df, algorithm, total_forest_size, max_label_set, min_label_set, k_folds, target_column):
    """
    Evaluate model performance using k-cross fold validation.
    :param df:
    :param algorithm:
    :param total_forest_size:
    :param max_label_set:
    :param min_label_set:
    :param k_folds:
    :return: list of calculated eval metrics
    """

    scores = list()
    true_positives = 0
    true_negatives = 0
    false_negatives = 0
    false_positives = 0

    true_positive_rates = list()
    auc_predictions = []
    mean_false_positive_rate = np.linspace(0, 1, 100)
    recall_array = np.linspace(0, 1, 100)

    # convert dataframe to numpy array for easy of use
    dataset = df.to_numpy()
    folds = generate_validation_folds(dataset, k_folds)
    # x_data, y_labels = split_data_from_target_columns(df, target_column)

    for roc_auc_plot_idx, (train_set, test_set, fold) in enumerate(_generate_fold(folds)):
        logging.info("")
        logging.info("Now evaluating fold {}...".format(roc_auc_plot_idx))

        # train model
        trees = algorithm.fit(train_set, total_forest_size, max_label_set, min_label_set)

        # run inference
        logging.info("")
        logging.info("Evaluating model for fold {} now (# of trees in forest: {})...".format(roc_auc_plot_idx, len(trees)))
        predicted, actual = predict(test_set, algorithm, trees, fold)

        # calculate accuracy for current model
        accuracy = calculate_accuracy(actual, predicted)
        scores.append(accuracy)

        # calculate classifier evaluation metrics
        curr_tp, curr_tn, curr_fn, curr_fp = calculate_classifier_eval_metrics(actual, predicted)
        true_positives += curr_tp
        true_negatives += curr_tn
        false_negatives += curr_fn
        false_positives += curr_fp

        # calculate ROC
        false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predicted, pos_label=1)
        true_positive_rates.append(np.interp(mean_false_positive_rate, false_positive_rate, true_positive_rate))

        # calculate ROC AUC
        plt.figure(1)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        auc_predictions.append(roc_auc)
        plt.plot(false_positive_rate, true_positive_rate, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (roc_auc_plot_idx, roc_auc))

        # calculate PRC
        plt.figure(2)
        precision_fold, recall_fold, threshold = precision_recall_curve(actual, predicted)
        precision_fold, recall_fold, threshold = precision_fold[::-1], recall_fold[::-1], threshold[::-1]  # reverse order of results
        #threshold = np.insert(threshold, 0, 1.0)
        precision_array = np.interp(recall_array, recall_fold, precision_fold)
        pr_auc = auc(recall_array, precision_array)

        label_fold = 'Fold %d AUC=%.4f' % (roc_auc_plot_idx + 1, pr_auc)
        plt.plot(recall_fold, precision_fold, alpha=0.3, label=label_fold)

        # calculate evaluation metrics
    precision, recall = calculate_precision_recall(true_positives, false_positives, false_negatives)

    # plot ROC AUC
    plt.figure(1)
    title = 'ROC: {}-cross fold validation'.format(k_folds)
    plot_roc_auc(plt, true_positive_rates, mean_false_positive_rate, title)

    # plot precision + recall curve
    plt.figure(2)
    plt.legend(loc='lower right', fontsize='small')

    return scores, precision, recall, plt
