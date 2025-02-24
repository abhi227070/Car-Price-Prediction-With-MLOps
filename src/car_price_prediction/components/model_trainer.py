import os
import sys
from dataclasses import dataclass
from src.car_price_prediction.exception import CustomException
from src.car_price_prediction.logger import logging
from src.car_price_prediction.utils import save_object, evaluate_models

# Importing various machine learning regression models from sklearn
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import r2_score

import numpy as np
import pandas as pd

# Defining configuration for saving trained model
@dataclass
class ModelTrainerConfig:
    # Path to save the trained model
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    model_report_file = os.path.join('artifacts', 'model_report.csv')
    
class ModelTrainer:
    """
    ModelTrainer class is responsible for training multiple machine learning models on the provided dataset.
    It performs hyperparameter tuning, model evaluation, and saves the best performing model.
    """
    
    def __init__(self):
        """
        Initializes the ModelTrainer class and sets the configuration for saving the trained model.
        """
        self.model_trainer_config = ModelTrainerConfig()
        
    def initialize_model_trainer(self, train_arr, test_arr):
        """
        Trains multiple machine learning regression models and selects the best one based on R2 score.
        
        Args:
            train_arr (np.array): Training data array, where the last column is the target variable.
            test_arr (np.array): Test data array, where the last column is the target variable.
        
        Returns:
            None. The best model is saved to the disk and its details are logged.
        
        Raises:
            CustomException: If an error occurs during model training.
        """
        
        try:
            logging.info("Model Training Started.")
            
            # Splitting the train and test data into features and target columns
            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],  # Features for training
                train_arr[:, -1],   # Target for training
                test_arr[:, :-1],   # Features for testing
                test_arr[:, -1]     # Target for testing
            )
            
            # Defining a dictionary of models to be used for training
            models = {
                "Linear Regression": LinearRegression(),
                "Support Vector Machine": SVR(),
                "K Nearest Neighbor": KNeighborsRegressor(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boost": GradientBoostingRegressor(),
                "Ada Boost": AdaBoostRegressor()
            }
            
            # Defining the hyperparameters for grid search
            params = {
                "Linear Regression": {},
                "Support Vector Machine": {'kernel': ['linear', 'poly', 'rbf', 'sigmoid']},
                "K Nearest Neighbor": {'n_neighbors': [3, 5, 7, 9]},
                "Ridge": {'alpha': [0.5, 1, 1.5, 2]},
                "Lasso": {'alpha': [0.5, 1, 1.5, 2]},
                "Decision Tree": {'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']},
                "Random Forest": {'n_estimators': [8, 16, 32, 64, 128, 256]},
                "Gradient Boost": {'learning_rate': [0.1, 0.01, 0.05, 0.001],
                                   'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                                   'n_estimators': [8, 16, 32, 64, 128, 256]},
                "Ada Boost": {'n_estimators': [8, 16, 32, 64, 128, 256],
                              'learning_rate': [0.1, 0.01, 0.5, 0.001]}
            }
            
            # Evaluating the models using the evaluate_models function
            model_report = evaluate_models(x_train, y_train, x_test, y_test, models, params)
            
            # Selecting the best model based on R2 score
            best_model_score = max(model_report['Test Score'])
            best_model_index = model_report['Test Score'].index(best_model_score)
            best_model_name = model_report['Model Name'][best_model_index]
            best_model = models[best_model_name]
            best_params = model_report['Model Params'][best_model_index]
            
            best_model_with_params = best_model.set_params(**best_params)
            
            # Saving the best model to disk using the save_object function
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model_with_params
            )
            
            # Saving model report
            model_report_df = pd.DataFrame(model_report)
            model_report_df.to_csv(self.model_trainer_config.model_report_file, index=False, header=True)
            
            # Making predictions on the test data using the best model
            y_pred = best_model_with_params.predict(x_test)
            r2_value = r2_score(y_test, y_pred)  # Evaluating the R2 score on the test set
            
            # Logging the results
            logging.info(f"Best Model Score: {best_model_score}.")
            logging.info(f"Best Model: {best_model_name}.")
            logging.info(f"Test Score (R2): {r2_value}.")
            logging.info("Model Training Completed.")
            
        except Exception as e:
            # If an error occurs, log the failure and raise a custom exception
            logging.error("Model Training Failed.")
            raise CustomException(e, sys)
