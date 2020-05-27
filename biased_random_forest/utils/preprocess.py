from random import randrange


def train_test_split(dataset, split=0.75):
    """
    Functions splits dataset into training and test datasets.
    :param dataset:
    :param split:
    :return: train, test datasets
    """

    train = list()
    train_size = split * len(dataset)
    dataset_copy = list(dataset)

    while len(train) < train_size:
        index = randrange(len(dataset_copy))
        train.append(dataset_copy.pop(index))

    return train, dataset_copy


def get_min_max_label_sets(df):
    """
    Get min and max label sets.
    :param df: dataframe
    :return: min and max label sets
    """

    # get max and min label sets
    max_label_set = set()
    max_label_set.add(int(df['Outcome'].value_counts().argmax()))

    min_label_set = set()
    min_label_set.add(int(df['Outcome'].value_counts().argmin()))

    return min_label_set, max_label_set

def replace_zero_values(df, columns, agg_function='median'):
    """
    Replace zero values in specified columns with aggregated values.
    :param df: pandas dataframe
    :param columns: columns with zero values to be replaced
    :param agg_function: string
    :return: pandas dataframe
    """

    for column in columns:
        temp = df[df[column] != 0][column]

        if agg_function == 'mean':
            df[column].replace(0, temp.mean())
        else:
            df[column].replace(0, temp.median())


def scale_numeric_features(df, target_column):
    """
    Performs basic column-wise feature scaling for all columns except the target variable column.
    :param df: pandas dataframe
    :param target_column: column value to omit in scaling
    :return: pandas dataframe
    """

    result = df.copy()

    for feature_name in df.columns:
        if feature_name != target_column:
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()

            denominator = max_value - min_value

            if denominator > 0:
                result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)

    return result
