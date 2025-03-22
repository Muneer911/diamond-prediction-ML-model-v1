import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

def create_preprocessing_pipeline():
    # Define numerical and categorical features
    numerical_features = ['carat', 'depth', 'table', 'x', 'y', 'z']
    categorical_features = ['cut', 'color', 'clarity']

    # Define transformations for numerical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy="mean")),  # Handle missing values
        ('scaler', StandardScaler())                 # Scale numerical features
    ])

    # Define transformations for categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy="most_frequent")),  # Handle missing values
        ('onehot', OneHotEncoder(handle_unknown="ignore"))     # One-hot encode categorical features
    ])

    # Combine transformations into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    return preprocessor

if __name__ == "__main__":
    # Load your training dataset
    file_path = "../data/diamond_cleaned.csv"  
    df = pd.read_csv(file_path)

    # Separate features and target
    X = df.drop(columns=['price'])  # Drop the 'price' column from the features
    y = df['price']  # Assign the 'price' column as the target variable

    # Create and fit the preprocessor
    preprocessor = create_preprocessing_pipeline()

    # Fit and transform the data
    X_preprocessed = preprocessor.fit_transform(X)

    # Get the feature names for one-hot encoded columns
    onehot_encoded_columns = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(['cut', 'color', 'clarity'])

    # Combine numerical and one-hot encoded categorical feature names
    processed_columns = ['carat', 'depth', 'table', 'x', 'y', 'z'] + list(onehot_encoded_columns)

    # Convert the preprocessed data into a DataFrame
    X_preprocessed = pd.DataFrame(X_preprocessed, columns=processed_columns)

    print(X_preprocessed)  # Preprocessed data

    # Save the fitted preprocessor
    joblib.dump(preprocessor, "../../model/preprocessor.joblib")
    print("Preprocessor saved to ../../model/preprocessor.joblib")