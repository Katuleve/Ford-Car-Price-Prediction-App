import numpy as np
import joblib
import streamlit as st

model = joblib.load('xgb_model.pkl')

scaler = joblib.load('scaler.pkl')

def car_price_prediction(input_data):
    input_changed= np.array(input_data).reshape(1,-1)
    std_input = scaler.transform(input_changed)
    prediction = model.predict(std_input)
    return 'Estimated Car Price: '+ str(prediction)

def main():
    st.title('FORD Car Price Prediction')

    year = st.text_input('Year')
    transmission = st.text_input('Transmission')
    mileage = st.text_input('Mileage')
    fuel_type = st.text_input('Fuel Type')
    tax = st.text_input('Tax')
    mpg = st.text_input('MPG')
    enginesize = st.text_input('Engine Size')

    pred_price : ''

    if st.button('Check Estimated Price'):
        pred_price = car_price_prediction([year, transmission, mileage,fuel_type, tax, mpg, enginesize])

        st.success(pred_price)

if __name__ == '__main__':
    main()