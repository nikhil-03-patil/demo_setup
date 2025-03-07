import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

def train_model():
    # Load the dataset
    dataset_path = os.path.join('data', 'salary_data.csv')
    df = pd.read_csv(dataset_path)
    
    # Display basic info about the dataset
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nSample data:")
    print(df.head())
    
    # Separate features and target
    X = df.drop('salary', axis=1)
    y = df['salary']
    
    # Create preprocessor for categorical features
    categorical_features = ['education', 'role']
    categorical_transformer = OneHotEncoder(drop='first')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    
    # Create and train the model pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions on test data
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("\nModel Evaluation Metrics:")
    print(f"Mean Absolute Error: ${mae:.2f}")
    print(f"Root Mean Squared Error: ${rmse:.2f}")
    print(f"R² Score: {r2:.4f}")
    
    # Save the model
    with open('salary_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save the column information
    feature_names = X.columns.tolist()
    with open('features.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    
    return model, feature_names

if __name__ == "__main__":
    train_model()