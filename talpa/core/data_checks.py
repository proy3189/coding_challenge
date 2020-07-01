import pandas as pd


def is_null(df):
    """
        Checks if any column in a dataframe is null.
        Parameters
        ----------
        df
        Returns
        -------
        """

    if df.isnull().values.any():
        print("Missing values for each column ", df.isnull().sum())
        return False
    # No Null values.
    return True


def is_numeric(df):
    """
    Checks if all columns in a dataframe are numeric.
    Parameters
    ----------
    df

    Returns
    -------
    """
    for dtype in df.dtypes:
        if dtype not in ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']:
            return False
    # All numeric.
    return True


def _check_data(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    assert isinstance(df, pd.DataFrame), 'Data has to be a pandas.Dataframe'
    assert not df.empty, 'Data has to be a non empty pandas.Dataframe'
    assert is_null(df), 'Dataframe has null values'
    print("Count of null values in each column :", df.isnull().sum().sum())
    assert is_numeric(X), 'Data has to be numeric'

