from . import data_preprocessing as dp
import pandas as pd
from pycaret.classification import ClassificationExperiment
from pycaret.regression import RegressionExperiment
from pycaret.datasets import get_data
class autoML :


    def __init__(self, df, target):
        self.df = df
        self.target = target

        column = self.df[self.target].dtype
        if pd.api.types.is_numeric_dtype(column):
            self.s = RegressionExperiment()
            self.task = 'R'
        elif column == 'object':
            self.s = ClassificationExperiment()
            self.task = 'C'
        

    def handle_missing_and_types(self,
                                 num_imputation_type,
                                 categorical_imputation_type,
                                 numerical_imputation_value,
                                 categorical_imputation_value,
                                 ignore_features):
        
        self.df = dp.handle_missing_values(self.df, num_imputation_type, categorical_imputation_type,
                                   numerical_imputation_value, categorical_imputation_value)
        self.df = dp.handle_data_types(self.df, ignore_features)

    

    