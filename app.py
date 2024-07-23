
import streamlit as st
import pandas as pd
import numpy as np
#import pickle
#import os
#from src.utils import load_object
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline

st.title('Restaurant Revenue Prediction')
name=st.text_input('Enter restaurant name')
location=st.selectbox('Select Location',options=['Rural','Downtown','Urban'])	
rating=st.text_input('Enter Rating')
cuisine=st.selectbox('Select Cuisine',options=['Japanese','Mexican','Indian','Italian','French','American'])	
seating=st.text_input("Enter the Seating Capacity")
price=st.text_input("Enter Average Meal Price")
marketing=st.text_input("Enter Marketing Budget")
followers=st.text_input("Enter the Social Media Followers")
experience=st.text_input('Enter the Experience of Chefs')
reviews=st.text_input('Enter the Number of Reviews')
review_length=st.text_input('Enter the Average Review Length')
ambience_score=st.text_input('Enter Ambience Score')
service_score=st.text_input('Enter Service Quality Score')
parking=st.selectbox("Parking Availability",options=['Yes', 'No'])
weekend_reservation=st.text_input('Enter Weekend Reservations')
weekday_reservation=st.text_input('Enter Weekday Reservations')
button=st.button('Generate the Revenue')

def predict():
    #name=str(name)
    rating=float(rating)
    seating=int(seating)
    price=float(price)
    marketing=int(marketing)
    followers=int(followers)
    experience=int(experience)
    reviews=int(reviews)
    review_length=float(review_length)
    ambience_score=float(ambience_score)
    service_score=float(service_score)
    weekend_reservation=int(weekend_reservation)
    weekday_reservation=int(weekday_reservation)
    
    data=CustomData(
        #name=name,
        Location=location,
        Rating=rating,
        Cuisine=cuisine,
        Seating_Capacity=seating,
        price=price,
        marketing=marketing,
        followers=followers,
        experience=experience,
        reviews=reviews,
        review_length=review_length,
        ambience_score=ambience_score,
        service_score=service_score,
        parking=parking,
        weekend_reservation=weekend_reservation,
        weekday_reservation=weekday_reservation
    )
    
    final_new_data=data.get_data_as_dataframe()
    predict_pipeline=PredictPipeline()
    pred=predict_pipeline.predict(final_new_data)
    results=round(pred[0],2)
    return results

if button==True:
    #name=str(name)
    rating=float(rating)
    seating=int(seating)
    price=float(price)
    marketing=int(marketing)
    followers=int(followers)
    experience=int(experience)
    reviews=int(reviews)
    review_length=float(review_length)
    ambience_score=float(ambience_score)
    service_score=float(service_score)
    weekend_reservation=int(weekend_reservation)
    weekday_reservation=int(weekday_reservation)
    
    data=CustomData(
        #name=name,
        Location=location,
        Rating=rating,
        Cuisine=cuisine,
        Seating_Capacity=seating,
        price=price,
        marketing=marketing,
        followers=followers,
        experience=experience,
        reviews=reviews,
        review_length=review_length,
        ambience_score=ambience_score,
        service_score=service_score,
        parking=parking,
        weekend_reservation=weekend_reservation,
        weekday_reservation=weekday_reservation

    )
    
    
    final_new_data=data.get_data_as_dataframe()
    predict_pipeline=PredictPipeline()
    pred=predict_pipeline.predict(final_new_data)
    results=round(pred[0],2)
    
    st.text_area('Annual Revenue will be:',results)