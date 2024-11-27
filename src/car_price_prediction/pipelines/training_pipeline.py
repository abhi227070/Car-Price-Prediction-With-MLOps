from src.car_price_prediction.logger import logging
from src.car_price_prediction.exception import CustomException
import sys
from src.car_price_prediction.components.data_ingestion import DataIngestion
from src.car_price_prediction.components.data_transformation import DataTransformation
from src.car_price_prediction.components.model_trainer import ModelTrainer

class TrainingPipeline:
    
    def __init__(self):
        
        """
        Initializes the PredictionPipeline class.
        No parameters are required for initialization.
        """
        
        pass
    
    def start_pipeline(self):
        
        try:
            
            # Logging the start of the training pipeline
            logging.info("Training Pipeline Started")
            
            # Data Ingestion: This step loads the raw data from the source (e.g., a file or database).
            # The DataIngestion class handles this process.
            data = DataIngestion()
            train_path, test_path = data.initiate_data_ingestion()
            
            # Data Transformation: This step processes the ingested data to prepare it for modeling.
            # The DataTransformation class performs necessary cleaning, feature engineering, etc.
            data_transformation = DataTransformation()
            train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_path, test_path)
            
            # Model Training: In this step, the transformed data is used to train the model.
            # The ModelTrainer class is responsible for initializing, training, and saving the best model.
            model_trainer = ModelTrainer()
            model_trainer.initialize_model_trainer(train_arr, test_arr)
            
            # Logging the successful completion of the pipeline
            logging.info("Training Pipeline Completed Successfully.")
            
        except Exception as e:
            # Logging any exceptions that occur during the pipeline execution
            logging.info("Custom Exception Occurred.")
            raise CustomException(e, sys)  # Raising a custom exception to handle errors more effectively
