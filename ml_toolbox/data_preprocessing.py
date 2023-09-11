import pandas as pd
from sklearn.impute import SimpleImputer


# Data Preparation

def handle_missing_values(data, num_imputation_type='mean', categorical_imputation_type='mode',
                          numerical_imputation_value=None, categorical_imputation_value=None):
    """
    Handle missing values in a Pandas DataFrame.

    Parameters:
        - data (DataFrame): The input dataset.
        - num_imputation_type (str): Imputation strategy for missing values in numerical columns.
        - categorical_imputation_type (str): Imputation strategy for missing values in categorical columns.
        - numerical_imputation_value: The value to use for numerical imputation when 'num_imputation_type' is int or float.
        - categorical_imputation_value: The value to use for categorical imputation when 'categorical_imputation_type' is 'str'.

    Returns:
        DataFrame: The original DataFrame with missing values handled according to the specified strategy.
    """
    numerical_cols = data.select_dtypes(include=['number']).columns
    imputer = SimpleImputer(strategy=num_imputation_type, fill_value=numerical_imputation_value)
    data[numerical_cols] = imputer.fit_transform(data[numerical_cols])
    
    categorical_cols = data.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        imputer = SimpleImputer(strategy=categorical_imputation_type, fill_value=categorical_imputation_value)
        data[categorical_cols] = imputer.fit_transform(data[categorical_cols])
    return data


def handle_data_types(df, ignore_features):
    """
    Handle data types in a Pandas DataFrame.

    Parameters:
        - df (DataFrame): The input dataset.
        - ignore_features (list of str): List of column names to be ignored.
    Returns:
        DataFrame: The DataFrame with updated data types based on the specified parameters.
    """
    if ignore_features:
        df = df.drop(columns=ignore_features)

    for column in df.columns:
        if df[column].dtype == 'object':
            try:
                df[column] = pd.to_numeric(df[column], errors='raise', downcast='integer')
                print(f"Converted '{column}' column to 'int' type.")
            except ValueError:
                try:
                    df[column] = pd.to_numeric(df[column], errors='raise', downcast='float')
                    print(f"Converted '{column}' column to 'float' type.")
                except ValueError:
                    print(f"Unable to convert '{column}' column to numeric type.")
    return df
