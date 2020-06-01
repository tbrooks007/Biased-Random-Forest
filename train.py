from random import seed
from biased_random_forest.utils.preprocess import replace_zero_values, scale_numeric_features, train_test_split, get_min_max_label_sets
from biased_random_forest.utils.evaluation_metrics_utils import evaluate, plot_roc_auc
from biased_random_forest.models.biased_random_forest import BiasedRandomForest
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import os.path
import logging

logging.basicConfig(level=logging.INFO)


def load_csv_as_dataframe(path):
    """
    Load csv as pandas dataframe.
    :param path: csv file path
    :return: dataframe or None
    """

    df = None

    if path:
     df = pd.read_csv(path)

    return df


def preprocess_data(df, columns, target_column, array_agg_function='median'):
    """
    Preprocess data in preparation for training or inference.
    :param df: pandas dataframe
    :param columns: columns with zero values to be replaced
    :param array_agg_function: numpy array aggregating function (i.e mean, median)
    :param target_column: string, name of column with target variable value
    :return: pandas dataframe
    """

    # replace all columns with zero values with their given aggregating value
    replace_zero_values(df, columns, array_agg_function)

    # scale features
    normalized_df = scale_numeric_features(df, target_column)

    return normalized_df


def cross_fold_validation(df, total_forest_size, k_nearest_neighbors, critical_area_ratio, target_column, k_folds=10):
    """
    Evaluate model performance using k-cross fold validation.
    :param df:
    :param total_forest_size:
    :param k_nearest_neighbors:
    :param critical_area_ratio:
    :param target_column:
    :param k_folds:
    """

    # get max and min label sets
    min_label_set, max_label_set = get_min_max_label_sets(df)

    # evaluate model using k-cross fold validation
    braf = BiasedRandomForest(k=k_nearest_neighbors, p=critical_area_ratio)
    scores, precision, recall, plot_obj, = evaluate(df, braf, total_forest_size, max_label_set, min_label_set,
                                         k_folds, target_column)

    logging.info('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
    logging.info('Test Precision: %s' % precision)
    logging.info('Test Recall: %s' % recall)

    # save k-cross fold ROC AUC plot to disk
    abs_path = os.path.abspath(os.path.dirname(__file__))
    roc_output_path = os.path.join(abs_path, "eval_output/k_fold_cross_validation_roc_auc.png")

    plot_obj.figure(1)
    roc_auc_figure = plot_obj.gcf()
    plot_obj.draw()
    roc_auc_figure.savefig(roc_output_path)

    # save PRC
    abs_path = os.path.abspath(os.path.dirname(__file__))
    pr_auc_path = os.path.join(abs_path, "eval_output/prc_auc.png")

    plot_obj.figure(2)
    pr_auc_figure = plot_obj.gcf()
    plot_obj.draw()
    pr_auc_figure.savefig(pr_auc_path)

    plot_obj.show()


def plot_and_save_roc_test(x_test, algorithm, trees):
    """
    Calculate, plot and save ROC AUC for test dataset.
    :param x_test:
    :param algorithm:
    :param trees:
    """

    true_positive_rates = list()
    auc_predictions = []
    mean_false_positive_rate = np.linspace(0, 1, 100)
    recall_array = np.linspace(0, 1, 100)

    # run predictions
    target_values = [row[-1] for row in x_test]
    predicted = [algorithm.bagging_predict(trees, row) for row in x_test]
    actual = [value for value in target_values]

    # calculate ROC
    false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predicted, pos_label=1)
    true_positive_rates.append(np.interp(mean_false_positive_rate, false_positive_rate, true_positive_rate))

    # calculate and plot ROC AUC
    plt.figure(3)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    auc_predictions.append(roc_auc)
    plt.plot(false_positive_rate, true_positive_rate, lw=2, alpha=0.3, label='(AUC = %0.2f)' % roc_auc)

    title = 'ROC: Test Data Set'
    roc_auc_plot_obj = plot_roc_auc(plt, true_positive_rates, mean_false_positive_rate, title)
    abs_path = os.path.abspath(os.path.dirname(__file__))
    roc_output_path = os.path.join(abs_path, "eval_output/roc_auc_test.png")

    roc_figure = roc_auc_plot_obj.gcf()
    roc_auc_plot_obj.draw()
    roc_figure.savefig(roc_output_path)

    # calculate and plot PRC AUC
    plt.figure(4)
    precision_fold, recall_fold, threshold = precision_recall_curve(actual, predicted)
    precision_fold, recall_fold, threshold = precision_fold[::-1], recall_fold[::-1], threshold[::-1]  # reverse order of results
    precision_array = np.interp(recall_array, recall_fold, precision_fold)
    pr_auc = auc(recall_array, precision_array)

    auc_label = 'AUC=%.4f' % (pr_auc)
    plt.legend(loc='lower right')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.plot(recall_fold, precision_fold, alpha=0.3, label=auc_label)

    abs_path = os.path.abspath(os.path.dirname(__file__))
    pr_output_path = os.path.join(abs_path, "eval_output/pr_auc_test.png")

    prc_figure = plt.gcf()
    plt.draw()
    prc_figure.savefig(pr_output_path)

    plt.show()

def train_model(df, total_forest_size, k_nearest_neighbors, critical_area_ratio, target_column, k_folds=10):
    """
    Train BRAF model.
    :param df:
    :param total_forest_size:
    :param k_nearest_neighbors:
    :param critical_area_ratio:
    :param target_column:
    :param k_folds:
    """

    if not df.empty:
        # get max and min label sets
        min_label_set, max_label_set = get_min_max_label_sets(df)

        # convert dataframe to numpy array for easy of use
        dataset = df.to_numpy()

        # split dataset
        x_train, x_test = train_test_split(dataset)

        # train model
        braf = BiasedRandomForest(k=k_nearest_neighbors, p=critical_area_ratio)
        trees = braf.fit(x_train, total_forest_size, max_label_set, min_label_set)

        # evaluate the model w/ k-cross fold
        cross_fold_validation(df, total_forest_size, k_nearest_neighbors, critical_area_ratio, target_column, k_folds)

        # evaluate the model w/ test dataset
        plot_and_save_roc_test(x_test, braf, trees)


if __name__ == "__main__":
    # set random seed
    seed(1)

    # Load pima data set
    abs_path = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(abs_path, "data/diabetes.csv")
    pima_dataset_df = load_csv_as_dataframe(path)

    # Note, these columns were identified during quick EDA analysis
    columns_with_zero_values = ['BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

    # Preprocess data for training and model evaluation
    pima_target_column = 'Outcome'
    processed_pima_df = preprocess_data(pima_dataset_df, columns_with_zero_values, pima_target_column)

    # TODO: taking in params from cmd
    # set user defined hyperparameters
    forest_size = 100
    k = 10
    p = 0.5
    k_folds = 10
    train_model(processed_pima_df, forest_size, k, p, pima_target_column, k_folds)
