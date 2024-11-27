import os
import sys
from src.car_price_prediction.logger import logging
from src.car_price_prediction.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from src.car_price_prediction.utils import read_sql_data
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    """
    A dataclass to store paths for the data ingestion process.
    
    Attributes:
        train_data_path (str): Path where the training data will be saved.
        test_data_path (str): Path where the test data will be saved.
        raw_data_path (str): Path where the raw data will be saved.
    """
    
    train_data_path = os.path.join('artifacts', 'train.csv')
    test_data_path = os.path.join('artifacts', 'test.csv')
    raw_data_path = os.path.join('artifacts', 'raw.csv')
    
class DataIngestion:
    """
    A class for managing the data ingestion process.
    
    This class is responsible for reading the raw dataset, splitting it into
    training and testing datasets, and saving them to specified paths.

    Attributes:
        ingestion_config (DataIngestionConfig): Configuration object containing paths for storing the data.
    """
    
    def __init__(self):
        """
        Initializes the DataIngestion class by setting up the data ingestion configuration.

        Attributes:
            ingestion_config (DataIngestionConfig): Initializes the configuration with default paths.
        """
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        """
        Initiates the data ingestion process, which involves:
        1. Reading the raw data from a CSV file.
        2. Splitting the data into training and testing sets.
        3. Saving the raw, training, and testing datasets to the specified paths.

        Returns:
            tuple: A tuple containing the paths to the training and testing datasets.

        Raises:
            CustomException: If any error occurs during the data ingestion process.
        """
        
        try:
            # Log the start of the data ingestion process
            logging.info("Data Ingestion Started.")
            
            # Read the raw data from the CSV file
            df = read_sql_data()
            
            # Ensure that the directory for raw data exists
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            
            # Split the data into training and testing sets (80% train, 20% test)
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            
            # Log successful data splitting
            logging.info("Train test split data successful.")
            
            # Save the raw, training, and testing datasets to their respective paths
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            # Log the success of the data ingestion process
            logging.info('Data Ingestion Successful.')
            
            # Return the paths to the saved train and test data
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            # Log error and raise a custom exception if data ingestion fails
            logging.info("Data Ingestion Failed")
            raise CustomException(e, sys)
