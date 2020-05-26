from math import sqrt
from biased_random_forest.utils.preprocess import replace_zero_values, scale_numeric_features
from biased_random_forest.models.biased_random_forest import BiasedRandomForest
import pandas as pd
import os.path
import numpy


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


if __name__ == "__main__":

    # TODO: read training data, preprocess data, split dataset to train / test sets, train model, then eval by
    # running predictions for both train and test set (calculate metrics train and test:
    #   precision, recall, AUPRC and AUROC for 10-fold cross-validation.)

    # Load pima data set
    abs_path = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(abs_path, "data/diabetes.csv")
    pima_dataset_df = load_csv_as_dataframe(path)

    # Note, these columns were identified during quick EDA analysis
    columns_with_zero_values = ['BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

    # Preprocess data for training and model evaluation
    pima_target_column = 'Outcome'
    processed_pima_df = preprocess_data(pima_dataset_df, columns_with_zero_values, pima_target_column)

    print(processed_pima_df.head(5))

    # train RF
    dataset = processed_pima_df.to_numpy()
    num_features = int(sqrt(len(dataset[0]) - 1))

    # TODO: split train/test datasets and train model on "train"

    braf = BiasedRandomForest(num_features)
    model = braf.fit(dataset)
