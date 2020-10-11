import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import MinMaxScaler
from category_encoders import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
import imblearn
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss


def fit_model(model, df, target, numeric_columns, categorical_columns, param_grid):
    x_train, x_test, y_train, y_test = train_test_split(df, target, test_size=0.10, random_state=42)

    imputer = IterativeImputer(max_iter=30, random_state=42)
    scaler = MinMaxScaler()

    frequent = SimpleImputer(missing_values=np.NaN, strategy='most_frequent')
    onehot = OneHotEncoder()
    
#     over_samp = SMOTE(sampling_strategy={0: count_class_0})
#     under_samp = NearMiss(sampling_strategy={1: count_class_1})

    preprocess = make_column_transformer(
        (make_pipeline(imputer, scaler), numeric_columns),
    (make_pipeline(frequent, onehot), categorical_columns)
        #,(make_pipeline(smt), numeric_columns+categorical_columns)
    )

    pipe = make_pipeline(preprocess,GridSearchCV(model, param_grid=param_grid, verbose=10))
    
    return pipe.fit(x_train, y_train)