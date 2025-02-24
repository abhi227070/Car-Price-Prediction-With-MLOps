import gradio as gr  
from src.car_price_prediction.pipelines.prediction_pipeline import CustomeData, PredictionPipeline

# Function to perform the prediction based on input data
def predictor(Year, Selling_Price, Kms_Driven, Fuel_Type, Seller_Type, Transmission, Owner):
    """
    This function processes the input features, transforms the data, and makes a prediction
    using a trained model pipeline.

    Args:
        Year (int): The year of the car.
        Selling_Price (float): The selling price of the car.
        Kms_Driven (float): The kilometers driven by the car.
        Fuel_Type (str): The fuel type of the car (e.g., 'Petrol', 'Diesel').
        Seller_Type (str): The type of seller (e.g., 'Dealer', 'Individual').
        Transmission (str): The type of transmission (e.g., 'Manual', 'Automatic').
        Owner (int): The number of previous owners.

    Returns:
        float: The predicted price of the car.
    """
    
    # Creating an instance of CustomeData with the input features
    data = CustomeData(
        Year, Selling_Price, Kms_Driven, Fuel_Type, Seller_Type, Transmission, Owner
    )
    
    # Transform the input data into the required format for prediction
    transformed_data = data.get_data_in_dataframe_format()
    
    # Creating an instance of PredictionPipeline
    pipe = PredictionPipeline()
    
    # Making a prediction using the transformed data
    result = pipe.predict(transformed_data)
    
    # Returning the prediction result
    return result

# Creating a Gradio interface for the prediction function
iface = gr.Interface(
    fn = predictor,  # The function to call for prediction
    inputs= [
        # Defining the input components for the Gradio interface:
        gr.Number(label="Year", maximum=2050, minimum=2000, step=1, value=2000),  # Year of the car (numeric input)
        gr.Number(label="Selling Price"),  # Selling price (numeric input)
        gr.Number(label="Kms Driven"),  # Kilometers driven (numeric input)
        gr.Dropdown(choices=['Petrol','Diesel','CNG'], label="Fuel Type"),  # Fuel type dropdown
        gr.Dropdown(choices=['Dealer', 'Individual'], label="Seller Type"),  # Seller type dropdown
        gr.Dropdown(choices=['Manual', 'Automatic'], label="Transmission"),  # Transmission type dropdown
        gr.Dropdown(choices=[0, 1, 2, 3], label="Owner")  # Owner dropdown (previous owners)
    ],
    
    outputs= gr.Number()  # Output is a numeric value (predicted price)
)

# Launch the Gradio interface with debugging enabled
iface.launch(debug=True)