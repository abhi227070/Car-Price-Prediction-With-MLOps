import pandas as pd
import sys
from src.car_price_prediction.exception import CustomException
from src.car_price_prediction.utils import load_object


class PredictionPipeline:
    """
    PredictionPipeline handles the process of loading a trained model and preprocessor,
    applying the preprocessor to the input data, and making predictions using the model.
    """
    
    def __init__(self):
        """
        Initializes the PredictionPipeline class.
        No parameters are required for initialization.
        
        It initialize the model and preprocesser object to do the prediction later on.
        """
        
        try:
            # Load the trained model and preprocessor from the saved file paths.
            model_path = "artifacts\\model.pkl"  # Path to the trained model.
            preprocessor_path = "artifacts\\preprocessor.pkl"  # Path to the preprocessor.
            
            self.model = load_object(model_path)  # Load the model.
            self.preprocessor = load_object(preprocessor_path)  # Load the preprocessor.
            
        except Exception as e:
            # If any exception occurs, raise a custom exception with the error details.
            raise CustomException(e, sys)
    
    def predict(self, feature):
        """
        Predicts the car price based on input features by using the trained model and preprocessor.

        Args:
            feature (pd.DataFrame): The input features for prediction.

        Returns:
            float: The predicted car price.

        Raises:
            CustomException: If there is an error during the prediction process.
        """
        try:
                        
            # Transform the input data using the preprocessor.
            scaled_data = self.preprocessor.transform(feature)
            
            # Use the model to predict the output (car price).
            predicted_data = self.model.predict(scaled_data)
            
            return predicted_data[0]  # Return the first predicted value.
        
        except Exception as e:
            # If any exception occurs, raise a custom exception with the error details.
            raise CustomException(e, sys)


class CustomeData:
    """
    CustomeData is used to encapsulate the car features into an object and convert them into
    a DataFrame format to be used for prediction.
    """
    
    def __init__(self, Year, Selling_Price, Kms_Driven, Fuel_Type, Seller_Type, Transmission, Owner):
        """
        Initializes the CustomeData object with car details.

        Args:
            Year (int): The year of the car.
            Selling_Price (float): The selling price of the car.
            Kms_Driven (float): The number of kilometers driven.
            Fuel_Type (str): The type of fuel used (e.g., 'Petrol', 'Diesel').
            Seller_Type (str): The type of seller (e.g., 'Dealer', 'Individual').
            Transmission (str): The type of transmission (e.g., 'Manual', 'Automatic').
            Owner (int): The number of previous owners.
        """
        
        self.Year = Year
        self.Selling_Price = Selling_Price
        self.Kms_Driven = Kms_Driven
        self.Fuel_Type = Fuel_Type
        self.Seller_Type = Seller_Type
        self.Transmission = Transmission
        self.Owner = Owner
        
    def get_data_in_dataframe_format(self):
        """
        Converts the input data into a pandas DataFrame format.

        Returns:
            pd.DataFrame: A DataFrame containing the input features.

        Raises:
            CustomException: If there is an error during the data conversion process.
        """
        try:
            # Convert the input data into a dictionary format.
            custom_data_dict = {
                "Year": [self.Year],
                "Selling_Price": [self.Selling_Price],
                "Kms_Driven": [self.Kms_Driven],
                "Fuel_Type": [self.Fuel_Type],
                "Seller_Type": [self.Seller_Type],
                "Transmission": [self.Transmission],
                "Owner": [self.Owner]
            }
            
            # Convert the dictionary into a DataFrame.
            df = pd.DataFrame(custom_data_dict)
            
            return df
        
        except Exception as e:
            # If any exception occurs, raise a custom exception with the error details.
            raise CustomException(e, sys)
