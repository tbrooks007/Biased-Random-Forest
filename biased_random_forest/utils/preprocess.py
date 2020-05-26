
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
