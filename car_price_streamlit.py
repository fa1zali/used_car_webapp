import numpy as np
import pickle
import streamlit as st

# loading the model
loaded_model = pickle.load(open('saved_model/car_price_ls_model.sav', 'rb'))

# create function for prediction

def car_price_prediction(input_data):
    
    input_data_arr= np.asarray(input_data)
    
    input_data_reshaped = input_data_arr.reshape(1, -1)
    
    # standardize input data
    
    price = loaded_model.predict(input_data_reshaped)
    
    return price[0]

def main():
    
    # tite for the app
    st.set_page_config(page_title="Used Car Price Prediction", page_icon="🚗", layout="centered")
    st.title("🚗 Used Car Price Prediction")
    
    # getting input data from user
    form = st.form(key="annotation")

    with form:
        cols = st.columns((1, 1))
        Year = cols[0].number_input("Year:", min_value=0,  value=0, step=1)
        Present_Price = cols[1].number_input("Present Price:", min_value=0.0,  value=0.0)
        Kms_Driven = cols[0].number_input("Km's Driven:", min_value=0,  value=0, step=1)
        Fuel_Type = cols[1].selectbox('Fuel Type:',('Petrol', 'Diesel', 'CNG'))
        Seller_Type = cols[0].selectbox('Seller Type:',('Dealer', 'Individual'))
        Transmission = cols[1].selectbox('Transmission:',('Manual', 'Automatic'))
        Owner = cols[0].selectbox('Owner:',(0, 1, 3))
        submitted = st.form_submit_button(label="Check Price")

    # code for prediction
    price = ""
    
    # create a button
    if submitted:

        fuel_dct = {"Petrol":2, "Diesel":1, "CNG":0}
        seller_dct = {"Individual":1, "Dealer":0}
        transmission_dct = {"Manual":1, "Automatic":0}

        Fuel_Type = fuel_dct.get(Fuel_Type)
        Seller_Type = seller_dct.get(Seller_Type)
        Transmission = transmission_dct.get(Transmission)

        price = car_price_prediction([Year, Present_Price, Kms_Driven, 
                                         Fuel_Type, Seller_Type, Transmission, Owner])
        st.success(price)

if __name__ == "__main__":
    main()