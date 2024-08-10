import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Load the trained model
model = load_model('C:/Users/PMLS/Downloads/Car price prediction/Model/car_price_prediction_model.h5')

# Load the fitted scaler
scaler = joblib.load('C:/Users/PMLS/Downloads/Car price prediction/Model/scaler.joblib')

# Initialize and fit label encoders
fuel_encoder = LabelEncoder()
seller_type_encoder = LabelEncoder()
transmission_encoder = LabelEncoder()
owner_encoder = LabelEncoder()
brand_encoder = LabelEncoder()

# List of car brands used for fitting the label encoder
car_brands = ['Maruti', 'Hyundai', 'Honda', 'Toyota', 'Ford', 'Chevrolet', 'Tata', 'Mahindra', 'Volkswagen', 'Renault', 'Nissan', 'Skoda', 'Mercedes-Benz', 'BMW', 'Audi', 'Datsun', 'Jeep', 'Jaguar', 'Mitsubishi', 'Land', 'Volvo']

# Fit the brand encoder
brand_encoder.fit(car_brands)

# Dictionary to set base prices for each car brand
brand_prices = {
    'Maruti': 20000,
    'Hyundai': 25000,
    'Honda': 30000,
    'Toyota': 35000,
    'Ford': 28000,
    'Chevrolet': 27000,
    'Tata': 22000,
    'Mahindra': 24000,
    'Volkswagen': 32000,
    'Renault': 26000,
    'Nissan': 29000,
    'Skoda': 31000,
    'Mercedes-Benz': 40000,
    'BMW': 45000,
    'Audi': 42000,
    'Datsun': 23000,
    'Jeep': 33000,
    'Jaguar': 47000,
    'Mitsubishi': 34000,
    'Land': 38000,
    'Volvo': 36000
}

# Function to make predictions
def predict_price(year, km_driven, fuel, seller_type, transmission, owner, brand):
    # Input data for prediction
    data = np.array([[year, km_driven, fuel, seller_type, transmission, owner, brand]])
    
    # Scale the data using the fitted scaler
    scaled_data = scaler.transform(data)

    # Make prediction
    price_prediction = model.predict(scaled_data)
    above_median = (price_prediction > 0.5).astype(int)

    # Assuming model outputs a scaled price, reverse scaling for actual price
    predicted_price = price_prediction[0][0] * 1000  # Assuming the model output is a normalized value
    
    # Add the base price for the car brand
    predicted_price += brand_prices[brand_encoder.inverse_transform([brand])[0]]
    
    return above_median, predicted_price

# Streamlit app layout
st.set_page_config(page_title='Car Price Prediction', layout='wide', initial_sidebar_state='expanded')

st.title('Car Price Prediction App')
st.markdown('## Predict whether the selling price of a car is above or below the median price.')

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
brand = st.sidebar.selectbox('Car Brand', car_brands)  # Add car brand input

# Encoding categorical features
fuel_mapping = {'Petrol': 0, 'Diesel': 1, 'CNG': 2, 'LPG': 3, 'Electric': 4}
seller_type_mapping = {'Individual': 0, 'Dealer': 1, 'Trustmark Dealer': 2}
transmission_mapping = {'Manual': 0, 'Automatic': 1}
owner_mapping = {'First Owner': 0, 'Second Owner': 1, 'Third Owner': 2, 'Fourth & Above Owner': 3, 'Test Drive Car': 4}

fuel = fuel_mapping[fuel]
seller_type = seller_type_mapping[seller_type]
transmission = transmission_mapping[transmission]
owner = owner_mapping[owner]

# Transform car brand input
brand = brand_encoder.transform([brand])[0]

# Predict button
if st.sidebar.button('Predict'):
    above_median, predicted_price = predict_price(year, km_driven, fuel, seller_type, transmission, owner, brand)
    st.write('### Prediction Result')
    if above_median == 1:
        st.success('The selling price is above the median price.', icon='✅')
    else:
        st.success('The selling price is below the median price.', icon='❌')
    st.write(f'### Predicted Selling Price: ${predicted_price:.2f}')  # Display in dollars

# Additional features: show input summary
st.write('## Input Summary')
input_summary = {
    'Year': year,
    'Kilometers Driven': km_driven,
    'Fuel Type': list(fuel_mapping.keys())[list(fuel_mapping.values()).index(fuel)],
    'Seller Type': list(seller_type_mapping.keys())[list(seller_type_mapping.values()).index(seller_type)],
    'Transmission': list(transmission_mapping.keys())[list(transmission_mapping.values()).index(transmission)],
    'Owner Type': list(owner_mapping.keys())[list(owner_mapping.values()).index(owner)],
    'Car Brand': brand_encoder.inverse_transform([brand])[0]
}
st.table(pd.DataFrame(input_summary, index=[0]))
