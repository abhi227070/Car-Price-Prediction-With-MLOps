import os
import sys
from src.car_price_prediction.logger import logging
from src.car_price_prediction.exception import CustomException
from dotenv import load_dotenv
import pandas as pd
import pymysql
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import dill

# Load environment variables from a .env file
load_dotenv()

# Retrieve database connection details from environment variables
host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
db = os.getenv("db")

def read_sql_data():
    """
    Reads data from a SQL database and returns it as a pandas DataFrame.

    This function connects to a MySQL database using credentials from environment variables 
    and executes a SQL query to fetch all records from the 'car_dataset' table.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the SQL data.

    Raises:
        CustomException: If any error occurs during the SQL data reading process.
    """
    
    logging.info("Reading SQL data started.")
    
    try:
        # Establish a connection to the MySQL database
        mydb = pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db
        )
        
        # Execute SQL query to fetch all data from the 'car_dataset' table
        df = pd.read_sql_query("select * from car_dataset", mydb)
        
        logging.info("Reading data from SQL successful.")
        
        return df
    except Exception as e:
        # Log error if any exception occurs while reading data
        logging.info("Exception occurred in Reading data from SQL")
        raise CustomException(e, sys)

def save_object(file_path, obj):
    """
    Saves a Python object to a file using pickle.

    Args:
        file_path (str): The path where the object will be saved.
        obj (object): The Python object to be saved.

    Raises:
        CustomException: If any error occurs during the saving process.
    """
    try:
        # Extract the directory path from the file path
        dir_path = os.path.dirname(file_path)

        # Create directory if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)

        # Save the object to the file using pickle
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        # Log error if any exception occurs while saving the object
        raise CustomException(e, sys)

def evaluate_models(x_train, y_train, x_test, y_test, models, params):
    """
    Evaluates multiple models using GridSearchCV for hyperparameter tuning and calculates R-squared scores.

    Args:
        x_train (pd.DataFrame): The feature matrix for training data.
        y_train (pd.Series): The target variable for training data.
        x_test (pd.DataFrame): The feature matrix for testing data.
        y_test (pd.Series): The target variable for testing data.
        models (dict): A dictionary of model names and their corresponding machine learning models.
        params (dict): A dictionary of model hyperparameters to be tuned using GridSearchCV.

    Returns:
        dict: A dictionary with model names as keys and R-squared scores on the test data as values.

    Raises:
        CustomException: If any error occurs during the model evaluation process.
    """
    
    try:
        logging.info("Report Making Started.")
        
        # Initialize an empty dictionary to store model evaluation results
        report = {}
        
        # Iterate through the models and their corresponding hyperparameters
        for i in range(len(list(models))):
            
            # Get the model and its hyperparameters
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]
            
            # Perform grid search for hyperparameter tuning
            gs = GridSearchCV(model, param)
            gs.fit(x_train, y_train)
            
            # Set the best hyperparameters found by grid search
            model.set_params(**gs.best_params_)
            model.fit(x_train, y_train)
            
            # Make predictions on both training and test data
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)
            
            # Calculate R-squared scores for both training and test sets
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            
            # Store the test R-squared score in the report dictionary
            report[list(models.keys())[i]] = test_model_score
            
        logging.info("Report made successfully.")
        
        return report
    
    except Exception as e:
        # Log error if any exception occurs during the model evaluation process
        logging.info("Report making failed.")
        raise CustomException(e, sys)

def load_object(file_path):
    """
    Loads a Python object from a file using dill.

    Args:
        file_path (str): The path of the file to load the object from.

    Returns:
        object: The loaded Python object.

    Raises:
        CustomException: If any error occurs during the loading process.
    """
    
    try:
        # Open the file and load the object using dill
        with open(file_path, "rb") as file:
            return dill.load(file)
        
    except Exception as e:
        # Log error if any exception occurs while loading the object
        raise CustomException(e, sys)
