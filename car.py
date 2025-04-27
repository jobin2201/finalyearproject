from tensorflow.keras.layers import DepthwiseConv2D
import streamlit as st
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
from keras.models import load_model
import os

# Custom function to remove 'groups' from the configuration


def custom_depthwise_conv2d(*args, **kwargs):
    kwargs.pop('groups', None)  # Remove 'groups' if it exists
    return DepthwiseConv2D(*args, **kwargs)


# Load the models with the custom object for DepthwiseConv2D
model_vehicle = tf.keras.models.load_model('keras_modelup.h5', custom_objects={
                                           'DepthwiseConv2D': custom_depthwise_conv2d})
model_damage = load_model('vgg16_car_damage_model.h5', compile=False, custom_objects={
    'DepthwiseConv2D': custom_depthwise_conv2d})
model_location = load_model('keras_model_loc.h5', compile=False, custom_objects={
    'DepthwiseConv2D': custom_depthwise_conv2d})
model_damage_level = load_model('keras_model_level.h5', compile=False, custom_objects={
    'DepthwiseConv2D': custom_depthwise_conv2d})

# Load the labels
class_names_vehicle = open("labelsup.txt", "r").readlines()
class_names_location = open("labelsloc.txt", "r").readlines()
class_names_damage_level = open("labels_level.txt", "r").readlines()

results = []

# Function to predict vehicle type


def import_and_predict_vehicle(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size)
    image = image.convert('RGB')  # Ensure image has 3 channels (RGB)
    image = np.asarray(image)
    image = (image.astype(np.float32) / 127.5) - 1
    img_reshape = image[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

# Function to predict damage


def import_and_predict_damage(image_data, model):
    size = (150, 150)
    image = ImageOps.fit(image_data, size)
    image = image.convert('RGB')  # Ensure image has 3 channels (RGB)
    image = np.asarray(image)
    image = (image.astype(np.float32) / 255.0)
    img_reshape = image[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

# Function to predict location


def import_and_predict_location(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size)
    image = image.convert('RGB')  # Ensure image has 3 channels (RGB)
    image = np.asarray(image)
    image = (image.astype(np.float32) / 127.5) - 1
    img_reshape = image[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

# Function to predict damage level


def import_and_predict_damage_level(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size)
    image = image.convert('RGB')  # Ensure image has 3 channels (RGB)
    image = np.asarray(image)
    image = (image.astype(np.float32) / 127.5) - 1
    img_reshape = image[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction


# Streamlit UI
st.write(""" 
         # Automated Car Damage Detection 
         """)
st.write("A Web App Utilizing Advanced Image Classification for Instant Damage Prediction")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])
if file is None:
    st.text("Please upload an image file for analysis")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)

    # Predict if it's a vehicle
    prediction_vehicle = import_and_predict_vehicle(image, model_vehicle)
    if np.argmax(prediction_vehicle) == 0:
        st.write("Vehicle!")
        st.text("Vehicle Probability: {:.2f}".format(prediction_vehicle[0][0]))

        # Check if the vehicle is damaged
        prediction_damage = import_and_predict_damage(image, model_damage)
        damage_prediction = "Damaged" if np.argmax(
            prediction_damage) == 0 else "Not Damaged"
        damage_probability = prediction_damage[0][0] if np.argmax(
            prediction_damage) == 0 else prediction_damage[0][1]
        st.write("Damage:", damage_prediction)
        st.text("Probability: {:.2f}".format(damage_probability))

        # Predict vehicle location
        prediction_location = import_and_predict_location(
            image, model_location)
        location_index = np.argmax(prediction_location)
        location_name = class_names_location[location_index].strip()
        location_probability = prediction_location[0][location_index]
        st.write("Location:", location_name)
        st.text("Location Probability: {:.2f}".format(location_probability))

        # Only predict damage level if the vehicle is damaged
        if damage_prediction == "Damaged":
            prediction_damage_level = import_and_predict_damage_level(
                image, model_damage_level)
            damage_level_index = np.argmax(prediction_damage_level)
            damage_level_name = class_names_damage_level[damage_level_index].strip(
            )
            damage_level_probability = prediction_damage_level[0][damage_level_index]
            st.write("Damage Level:", damage_level_name)
            st.text("Damage Level Probability: {:.2f}".format(
                damage_level_probability))

        results.append([file.name, "Vehicle", prediction_vehicle[0][0],
                        damage_prediction, damage_probability, location_name, location_probability,
                        damage_level_name if damage_prediction == "Damaged" else "N/A", damage_level_probability if damage_prediction == "Damaged" else "N/A"])
    else:
        st.write("Not a Vehicle.")
        results.append([file.name, "Not a Vehicle", "",
                       "", "", "", "", "N/A", "N/A"])

    # Create or load the Excel file
    excel_file = "results.xlsx"
    if os.path.isfile(excel_file):
        df = pd.read_excel(excel_file)
    else:
        df = pd.DataFrame(columns=["Image Name", "Vehicle or not", "Probability of Vehicle",
                                   "Damaged or not", "Probability", "Location", "Location Probability",
                                   "Damage Level", "Damage Level Probability"])

    # Append the new results to the DataFrame and save it to the Excel file
    new_df = pd.DataFrame(results, columns=["Image Name", "Vehicle or not", "Probability of Vehicle",
                                            "Damaged or not", "Probability", "Location", "Location Probability",
                                            "Damage Level", "Damage Level Probability"])
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_excel(excel_file, index=False)

    # Display the DataFrame in the app
    st.write("Current Results:")
    st.dataframe(df)

    # Provide a download button for the Excel file
    with open(excel_file, "rb") as f:
        st.download_button(
            label="Download Results",
            data=f,
            file_name=excel_file,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
