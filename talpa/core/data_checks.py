import pandas as pd
import logging

def is_null(df):
    '''
    Checks if any column in a dataframe is null.
    :param df: Dataframe of shape (n_samples, n_features)
            The input samples.
    :return:
    '''
    if df.isnull().values.any():
        logging.info("Missing values for each column ", df.isnull().sum())
        return False
    # No Null values.
    return True

def is_numeric(df):
    '''Checks if all columns in a dataframe are numeric.
    :param df: Dataframe of shape (n_samples, n_features)
            The input samples.
    :return:
    '''

    for dtype in df.dtypes:
        if dtype not in ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']:
            return False
    # All numeric.
    return True


def check_data(df):
    '''
    This function will check the validity of data
    :param df: Dataframe of shape (n_samples, n_features)
            The input samples.
    :return:
    '''
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    assert isinstance(df, pd.DataFrame), 'Data has to be a pandas.Dataframe'
    assert not df.empty, 'Data has to be a non empty pandas.Dataframe'
    assert is_null(df), 'Dataframe has null values'
    assert is_numeric(X), 'Data has to be numeric'

