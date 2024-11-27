import numpy as np
import pandas as pd
import os
import sys
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from src.car_price_prediction.logger import logging
from src.car_price_prediction.exception import CustomException
from src.car_price_prediction.utils import save_object


@dataclass
class DataTransformationConfig:
    """
    Configuration class to store paths related to data transformation.
    
    Attributes:
        preprocessor_obj_file_path (str): Path where the preprocessor object will be saved.
    """
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    """
    A class to handle the transformation of the dataset.
    
    This class is responsible for:
    1. Building a pipeline for data transformation including scaling, encoding, and imputation.
    2. Transforming the dataset for model training and testing.
    
    Attributes:
        data_transformation_config (DataTransformationConfig): Config object for storing transformation-related paths.
    """
    
    def __init__(self):
        """
        Initializes the DataTransformation class by setting up the transformation configuration.
        """
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_object(self):
        """
        Creates a pipeline for data transformation which includes:
        - Imputation of missing values for numerical and categorical columns.
        - Scaling of numerical and categorical columns.
        - Ordinal encoding for categorical features.
        - Principal Component Analysis (PCA) for dimensionality reduction.

        Returns:
            Pipeline: A scikit-learn pipeline that performs data transformation.
        
        Raises:
            CustomException: If any error occurs during the creation of the transformation pipeline.
        """
        
        try:
            logging.info("Creating Data Transformation Pipeline.")
            
            # Define the columns for numerical and categorical features
            numerical_columns = ['Year', 'Selling_Price', 'Kms_Driven', 'Owner']
            categorical_columns = ['Fuel_Type', 'Seller_Type', 'Transmission']
            total_columns = numerical_columns + categorical_columns
            
            # Custom transformer to drop specified columns from the dataframe
            class DropColumns(BaseEstimator, TransformerMixin):
                def __init__(self, columns_to_drop):
                    self.columns_to_drop = columns_to_drop

                def fit(self, X, y=None):
                    return self

                def transform(self, X):
                    return X.drop(columns=self.columns_to_drop, axis=1)
                
            # Define transformation steps for numerical columns
            num_col_transformation = Pipeline(
                steps=[
                    ('impute', SimpleImputer(strategy='median')),  # Impute missing values with the median
                    ('scaler', StandardScaler())  # Scale numerical values
                ]
            )

            # Define transformation steps for categorical columns
            cat_column_transformation = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values with the most frequent value
                    ('encoder', OrdinalEncoder()),  # Encode categorical features using ordinal encoding
                    ('scale', StandardScaler())  # Scale the encoded values
                ]
            )

            # Use ColumnTransformer to apply different transformations on numerical and categorical columns
            preprocessor = ColumnTransformer(
                transformers=[
                    ('numerical_column_transformation', num_col_transformation, numerical_columns),
                    ('categorical_column_transformation', cat_column_transformation, categorical_columns)
                ],
                remainder='passthrough'  # Keep the other columns unchanged
            )

            # Create a pipeline that includes preprocessing and PCA for dimensionality reduction
            pipe = Pipeline(
                steps=[
                    ('preprocessing', preprocessor),
                    ('pca', PCA(n_components=5))  # Reduce data dimensions to 5 principal components
                ]
            )
            
            logging.info("Creation of data transformation pipeline successful.")
            
            return pipe
        
        except Exception as e:
            logging.info("Data Transformation pipeline creation failed.")
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        """
        Initiates the transformation of the training and testing data by:
        - Reading the data from CSV files.
        - Applying the transformation pipeline to preprocess the data.
        - Saving the transformed data and preprocessing object for future use.

        Args:
            train_path (str): Path to the training data CSV file.
            test_path (str): Path to the testing data CSV file.

        Returns:
            tuple: A tuple containing the following:
                - Transformed training data array.
                - Transformed testing data array.
                - Path where the preprocessor object is saved.
        
        Raises:
            CustomException: If any error occurs during data transformation.
        """
        
        try:
            logging.info("Data Transformation Started.")
            
            # Define the columns for numerical and categorical features
            numerical_columns = ['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven', 'Owner']
            categorical_columns = ['Fuel_Type', 'Seller_Type', 'Transmission']
            total_columns = numerical_columns + categorical_columns
            
            # Read the training and testing datasets from the provided paths
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            # Select only the relevant columns from the datasets
            train_df = train_df[total_columns]
            test_df = test_df[total_columns]
            
            logging.info("Getting data from artifacts successful.")
            
            # Get the data transformation pipeline
            preprocessing_obj = self.get_data_transformer_object()
            
            # Specify the target column name
            target_column_name = "Present_Price"
            
            # Separate features (X) and target (y) for both training and testing datasets
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            # Apply transformations on both training and testing features
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            logging.info("Data Preprocessing Successful.")
            
            # Combine the transformed features with the target variables for both train and test sets
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            # Save the preprocessor object for future use
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            logging.info("Saving preprocessor object successful.")
            logging.info("Data Transformation Successful.")
            
            return (
                train_arr,  # Transformed training data
                test_arr,   # Transformed testing data
                self.data_transformation_config.preprocessor_obj_file_path  # Path to the preprocessor object
            )
            
        except Exception as e:
            logging.info("Data Transformation failed.")
            raise CustomException(e, sys)
