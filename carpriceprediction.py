import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model pipeline
model_pipeline = joblib.load('C:/Users/PMLS/Downloads/Car price prediction/Model/car_price_prediction_pipeline.pkl')

# Function to make predictions
def predict_price(year, km_driven, fuel, seller_type, transmission, owner, brand):
    try:
        # Input data for prediction
        data = pd.DataFrame({
            'year': [year],
            'km_driven': [km_driven],
            'fuel': [fuel],
            'seller_type': [seller_type],
            'transmission': [transmission],
            'owner': [owner],
            'brand': [brand]
        })

        # Make prediction using the model pipeline
        predicted_price = model_pipeline.predict(data)[0]
        return predicted_price
    except Exception as e:
        st.error(f"Error occurred during prediction: {e}")
        return None

# Streamlit app layout
st.set_page_config(page_title='Car Price Prediction', layout='wide', initial_sidebar_state='expanded')

st.title('Car Price Prediction App')
st.markdown('## Predict the selling price of a car based on its features.')

# Improved styling
st.markdown(
    """
    <style>
    .main {
        background-color: #ffffff;
    }
    .sidebar .sidebar-content {
        background-color: #f5f5f5;
    }
    .block-container {
        padding: 1rem 2rem;
        border-radius: 10px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        background-color: #000000;
        margin-top: 10px;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .stSuccess {
        background-color: #d4edda;
        color: #155724;
        border-radius: 5px;
        padding: 10px;
    }
    .stTable {
        background-color: #f8f9fa;
        color: #333;
        border-radius: 5px;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Input fields in sidebar
st.sidebar.header('Input Car Features')
year = st.sidebar.number_input('Year', min_value=1990, max_value=2024, value=2015)
km_driven = st.sidebar.number_input('Kilometers Driven', min_value=0, max_value=500000, value=50000)
fuel = st.sidebar.selectbox('Fuel Type', ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'])
seller_type = st.sidebar.selectbox('Seller Type', ['Individual', 'Dealer', 'Trustmark Dealer'])
transmission = st.sidebar.selectbox('Transmission', ['Manual', 'Automatic'])
owner = st.sidebar.selectbox('Owner Type', ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'])
brand = st.sidebar.selectbox('Car Brand', ['Maruti', 'Hyundai', 'Honda', 'Toyota', 'Ford', 'Chevrolet', 'Tata', 'Mahindra', 'Volkswagen', 'Renault', 'Nissan', 'Skoda', 'Mercedes-Benz', 'BMW', 'Audi', 'Datsun', 'Jeep', 'Jaguar', 'Mitsubishi', 'Land', 'Volvo'])

# Predict button
if st.sidebar.button('Predict'):
    predicted_price = predict_price(year, km_driven, fuel, seller_type, transmission, owner, brand)
    if predicted_price is not None:
        st.write('### Prediction Result')
        st.success(f'Predicted Selling Price: ${predicted_price:.2f}')  # Display in dollars

# Additional features: show input summary
st.write('## Input Summary')
input_summary = {
    'Year': year,
    'Kilometers Driven': km_driven,
    'Fuel Type': fuel,
    'Seller Type': seller_type,
    'Transmission': transmission,
    'Owner Type': owner,
    'Car Brand': brand
}
st.table(pd.DataFrame(input_summary, index=[0]))
