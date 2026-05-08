import streamlit as st
from model.predict import predict_gold

st.title("MineralIQ - Gold Probability AI")

st.write("Enter land features to estimate gold availability")

lat = st.number_input("Latitude")
lon = st.number_input("Longitude")
elevation = st.number_input("Elevation")
slope = st.number_input("Slope")
distance = st.number_input("Distance to known gold deposits")

if st.button("Predict Gold"):
    result, prob = predict_gold(lat, lon, elevation, slope, distance)

    st.write("### Result")
    st.write("Gold Found:", "YES" if result == 1 else "NO")
    st.write("Probability:", round(prob * 100, 2), "%")