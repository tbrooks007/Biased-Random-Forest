from random import seed
from biased_random_forest.utils.preprocess import replace_zero_values, scale_numeric_features, train_test_split
from biased_random_forest.utils.evaluation_metrics_utils import evaluate_algorithm, accuracy_metric
from biased_random_forest.models.biased_random_forest import BiasedRandomForest
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


def train_model(df, total_forest_size, k_nearest_neighbors, critical_area_ratio, num_folds=10):
    """
    Train BRAF model.
    :param df:
    :param total_forest_size:
    :param k_nearest_neighbors:
    :param critical_area_ratio:
    :param num_folds:
    """

    if not df.empty:
        # convert dataframe to numpy array for easy of use
        dataset = processed_pima_df.to_numpy()

        # split dataset
        seed(1)
        X_train, x_test = train_test_split(dataset)

        # get max and min label sets
        max_label_set = set()
        max_label_set.add(int(df['Outcome'].value_counts().argmax()))

        min_label_set = set()
        min_label_set.add(int(df['Outcome'].value_counts().argmin()))

        # train BRAF
        braf = BiasedRandomForest(k=k_nearest_neighbors, p=critical_area_ratio)

        scores, precision, recall = evaluate_algorithm(X_train, braf, total_forest_size, max_label_set, min_label_set, num_folds)
        logging.info('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
        logging.info('Test Precision: %s' % precision)
        logging.info('Test Recall: %s' % recall)

if __name__ == "__main__":

    # Load pima data set
    abs_path = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(abs_path, "data/diabetes.csv")
    pima_dataset_df = load_csv_as_dataframe(path)

    # Note, these columns were identified during quick EDA analysis
    columns_with_zero_values = ['BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

    # Preprocess data for training and model evaluation
    pima_target_column = 'Outcome'
    processed_pima_df = preprocess_data(pima_dataset_df, columns_with_zero_values, pima_target_column)

    # set user defined hyperparameters
    forest_size = 100
    k = 10
    p = 0.5
    num_folds = 10
    train_model(processed_pima_df, forest_size, k, p, num_folds)
