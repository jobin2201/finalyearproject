import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
import json
import uuid
import numpy as np
import os

# Load the dataset


@st.cache_data
def load_data():
    return pd.read_csv("carprice.csv")

# Preprocess the data and train the model


def train_model(data):
    data = pd.get_dummies(
        data, columns=["Car Make", "Car Model", "Part"], drop_first=True)
    X = data.drop("Price", axis=1)
    y = data["Price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, "price_prediction_model.pkl")
    return model

# Load the model or train if not already saved


def load_or_train_model():
    try:
        model = joblib.load("price_prediction_model.pkl")
    except:
        data = load_data()
        model = train_model(data)
    return model


# Load data and model
data = load_data()
model = load_or_train_model()

# Streamlit app title
st.title("Car Part Price Prediction")

# Initialize session state variables for item list, total cost, and user ID
if "items_list" not in st.session_state:
    st.session_state["items_list"] = []
if "total_cost" not in st.session_state:
    st.session_state["total_cost"] = 0.0
if "user_id" not in st.session_state:
    st.session_state["user_id"] = str(
        uuid.uuid4())  # Unique ID for this session

# Path to the single JSON file for all bills
bills_file = "bills.json"

# Ensure dropdowns are blank initially and reset after each item addition
st.header("Enter Car Details")
car_make = st.selectbox("Select Car Make", options=[
                        ""] + list(data["Car Make"].unique()))
car_model = st.selectbox("Select Car Model", options=[
                         ""] + list(data[data["Car Make"] == car_make]["Car Model"].unique()) if car_make else [""])
year = st.selectbox("Select Year", options=[
                    ""] + sorted(data["Year"].unique()))
part = st.selectbox("Select Part", options=[""] + list(data["Part"].unique()))

# Prepare input data for prediction
if car_make and car_model and year and part:
    input_data = pd.DataFrame([[car_make, car_model, year, part]], columns=[
                              "Car Make", "Car Model", "Year", "Part"])
    input_data = pd.get_dummies(
        input_data, columns=["Car Make", "Car Model", "Part"], drop_first=True)
    input_data = input_data.reindex(
        columns=model.feature_names_in_, fill_value=0)

    # Predict and add item to cart
    if st.button("Add Item to Cart"):
        prediction = model.predict(input_data)[0]
        item_details = {
            "Car Make": car_make,
            "Car Model": car_model,
            "Year": int(year),
            "Part": part,
            "Price": float(prediction)
        }

        # Check for duplicate items
        if item_details not in st.session_state["items_list"]:
            st.session_state["items_list"].append(item_details)
            st.session_state["total_cost"] += prediction
            st.write(f"Added {part} for {car_make} {
                     car_model} ({year}) at ₹{prediction:.2f}")
        else:
            st.warning("This item is already in the cart.")

        st.experimental_rerun()

# Display dynamic summary of all items added
st.subheader("Bill Summary")
if st.session_state["items_list"]:
    for i, item in enumerate(st.session_state["items_list"], 1):
        col1, col2 = st.columns([0.9, 0.1])
        with col1:
            st.write(f"{i}. {item['Car Make']} {item['Car Model']} ({
                     item['Year']}) - {item['Part']}: ₹{item['Price']:.2f}")
        with col2:
            if st.button("❌", key=f"remove_{i}"):
                st.session_state["total_cost"] -= item["Price"]
                st.session_state["items_list"].remove(item)
                st.experimental_rerun()
    st.write(f"**Total Cost: ₹{st.session_state['total_cost']:.2f}**")

# Finish and generate bill
if st.button("Finish and Generate Bill"):
    date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    bill = {
        "User ID": st.session_state["user_id"],
        "Date and Time": date_time,
        "Items": st.session_state["items_list"],
        "Total Cost": st.session_state["total_cost"]
    }

    # Load or create the bills JSON file
    if os.path.exists(bills_file):
        with open(bills_file, "r") as f:
            bills_data = json.load(f)
    else:
        bills_data = []

    # Append the new bill and save to the file
    bills_data.append(bill)
    with open(bills_file, "w") as f:
        json.dump(bills_data, f, indent=4)

    # Display final bill summary
    st.subheader("Final Bill")
    st.write(f"**User ID**: {st.session_state['user_id']}")
    st.write(f"**Date and Time**: {date_time}")
    for i, item in enumerate(bill["Items"], 1):
        st.write(f"{i}. {item['Car Make']} {item['Car Model']} ({
                 item['Year']}) - {item['Part']}: ₹{item['Price']:.2f}")
    st.write(f"**Total Cost: ₹{bill['Total Cost']:.2f}**")

    # Clear session state for a new transaction
    st.session_state["items_list"] = []
    st.session_state["total_cost"] = 0.0
    st.success("Bill saved to bills.json")
