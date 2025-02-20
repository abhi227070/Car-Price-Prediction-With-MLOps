import streamlit as st
from src.car_price_prediction.pipelines.prediction_pipeline import CustomeData, PredictionPipeline

st.title("Car Price Prediction")

Year = st.number_input(label="Year", min_value=2000, max_value=2050, step=1)      
Selling_Price = st.number_input(label="Selling Price")
Kms_Driven = st.number_input(label="Kms Driven")
Fuel_Type = st.selectbox(label="Fuel Type", options=['Petrol','Diesel','CNG'])
Seller_Type = st.selectbox(label="Seller Type", options=['Dealer', 'Individual'])
Transmission = st.selectbox(label="Transmission", options=['Manual', 'Automatic'])
Owner = st.selectbox(label="Owner", options=[0,1,2,3])

if st.button(label = "Predict"):
    
    data = CustomeData(
        Year, Selling_Price, Kms_Driven, Fuel_Type, Seller_Type, Transmission, Owner
    )
    
    transformed_data = data.get_data_in_dataframe_format()
    
    pipe = PredictionPipeline()
    
    result = pipe.predict(transformed_data)
    
    st.success(result)
