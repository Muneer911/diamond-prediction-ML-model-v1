import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
from sklearn.metrics import mean_squared_error

def create_preprocessing_pipeline():
    categorical_features = ['cut', 'color', 'clarity']
    numerical_features = ['carat', 'depth', 'table', 'x', 'y', 'z']

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown="ignore")),
        ('imputer',SimpleImputer(strategy='most-frequent'))
    ])

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy="mean")),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )


    return preprocessor

preprocessor = create_preprocessing_pipeline
joblib.dump(preprocessor, "../../model/preprocessor.pkl")
print("Preprocessor saved to ../../model/preprocessor.pkl")