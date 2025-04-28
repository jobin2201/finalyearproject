import sys
import tempfile
from fpdf import FPDF
import streamlit as st
from seven_try2 import main_drowsiness_detection
import pandas as pd
import os
import uuid
import requests
import openai
import json
from PIL import Image
from tensorflow.keras.models import load_model
import numpy as np
from datetime import datetime
import cost
# Ensure users.csv exists
if not os.path.exists("users.csv"):
    with open("users.csv", "w") as f:
        f.write("ID,Name,Phone,Username,Password\n")  # CSV header

# Initialize session state variables
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "car_damage_page" not in st.session_state:
    st.session_state.car_damage_page = False
if 'reasons_page' not in st.session_state:
    st.session_state.reasons_page = False
if 'drowsiness_detection_page' not in st.session_state:
    st.session_state.drowsiness_detection_page = False
if 'other_reasons_page' not in st.session_state:
    st.session_state.other_reasons_page = False
if 'demo_checking_page' not in st.session_state:
    st.session_state.demo_checking_page = False
if 'automatic_checking_page' not in st.session_state:
    st.session_state.automatic_checking_page = False
if 'result_page' not in st.session_state:
    st.session_state.result_page = False
if 'cost_page' not in st.session_state:
    st.session_state.cost_page = False
if 'cost_result_page' not in st.session_state:
    st.session_state.cost_result_page = False
if 'bill_generated' not in st.session_state:
    st.session_state.bill_generated = False
if 'scenario_generation_page' not in st.session_state:
    st.session_state.scenario_generation_page = False
if 'story_page' not in st.session_state:  # New page initialization
    st.session_state.story_page = False
if 'damage_confirmed' not in st.session_state:
    st.session_state.damage_confirmed = False

from car import (
    import_and_predict_vehicle,
    import_and_predict_damage,
    import_and_predict_location,
    import_and_predict_damage_level,
    model_vehicle,
    model_damage,
    model_location,
    model_damage_level,
    class_names_location,
    class_names_damage_level
)


# Function to load users from CSV
def load_users():
    return pd.read_csv("users.csv")

# Function to register a new user


def register_user(name, username, password):
    users = load_users()
    if username in users['Username'].values:
        st.warning("Username already exists. Please choose another one.")
    else:
        unique_id = str(uuid.uuid4())
        with open("users.csv", "a") as f:
            # Phone is omitted
            f.write(f"{unique_id},{name},,{username},{password}\n")
        st.success("You have successfully registered. Please log in.")

# Function to log in a user


def login_user(username, password):
    users = load_users()
    user = users[(users['Username'] == username) &
                 (users['Password'] == password)]
    if not user.empty:
        return user.iloc[0]  # Return user details
    return None


def create_pdf(user_details, damage_status, drowsiness_status, bill_details, story):
    from fpdf import FPDF
    import os

    def sanitize_text(text):
        # Replace problematic characters
        text = text.replace("\u201d", '"').replace(
            "\u201c", '"')  # Replace quotes
        text = text.replace("\u2018", "'").replace(
            "\u2019", "'")  # Replace single quotes
        return text

    # Sanitize story text
    story = sanitize_text(story)

    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, 'Scenario Report', border=0, ln=1, align='C')
            self.ln(5)

        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', align='C')

    # Create instance of FPDF
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Add a page
    pdf.add_page()

    # Title
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Scenario Report', ln=True, align='C')
    pdf.ln(10)

    # User Details
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'User Details', ln=True)
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f"Name: {user_details['Name']}", ln=True)
    pdf.cell(0, 10, f"User ID: {user_details['ID']}", ln=True)
    pdf.ln(5)

    # Car Damage Status
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Car Damage Status', ln=True)
    pdf.set_font('Arial', '', 12)
    pdf.cell(
        0, 10, f"Damaged: {damage_status.get('Damaged', 'Undamaged')}", ln=True)
    pdf.cell(0, 10, f"Location: {damage_status['Location']}", ln=True)
    pdf.cell(0, 10, f"Damage Level: {damage_status['Damage Level']}", ln=True)
    pdf.ln(5)

    # Drowsiness Detection
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Driver Drowsiness', ln=True)
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f"Eye Status: {drowsiness_status['Eye Status']}", ln=True)
    pdf.cell(
        0, 10, f"Mouth Status: {drowsiness_status['Mouth Status']}", ln=True)
    pdf.cell(
        0, 10, f"Final Status: {drowsiness_status['Final Status']}", ln=True)
    pdf.ln(5)

    # Bill Details
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Bill Details', ln=True)
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f"User ID: {bill_details['User ID']}", ln=True)
    pdf.cell(0, 10, f"Date and Time: {bill_details['Date and Time']}", ln=True)
    pdf.cell(0, 10, f"Total Cost: {bill_details['Total Cost']:.2f}", ln=True)
    pdf.ln(5)

    # Detailed Items in Bill
    if 'Items' in bill_details:
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Items in Bill:', ln=True)
        pdf.set_font('Arial', '', 12)
        for item in bill_details['Items']:
            pdf.cell(
                0, 10, f"- {item['Part']}({item['Car Make']} {item['Car Model']} - {item['Year']}): ${item['Price']: .2f}", ln=True)
        pdf.ln(5)

    # Story
    def safe_text(text):
        return text.replace('—', '-').encode('latin-1', 'ignore').decode('latin-1')

    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Generated Story', ln=True)
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 10, safe_text(story))
    pdf.ln(5)

    # Save the PDF
    pdf_file_path = os.path.join(os.getcwd(), "Scenario_Report.pdf")
    pdf.output(pdf_file_path)

    return pdf_file_path



def load_drowsiness_results():
    if os.path.exists('drowsiness_results.json'):
        with open('drowsiness_results.json') as f:
            return json.load(f)
    else:
        st.error("Drowsiness result file not found.")
        return {}


drowsiness_results_file = "drowsiness_results.json"
users_file = "users.csv"
assessment_results_file = "assessment_results.xlsx"
# Path to the bills JSON file
bills_file = "bills.json"

# Function to retrieve and display the last entry from the JSON file

# Function to get the GPT-4 response


def get_gpt4_response(prompt):
    try:
        # Replace with your actual OpenAI API key
        openai.api_key = ''
        # Request GPT-4 model for response
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        # Return the generated content
        return response['choices'][0]['message']['content'].strip()

    except Exception as e:
        return f"Error generating response: {str(e)}"


def show_last_bill():
    # Load the data from the JSON file
    try:
        with open(bills_file, "r") as f:
            bills_data = json.load(f)

        # Get the last entry if the file is not empty
        if bills_data:
            last_bill = bills_data[-1]  # Last entered data
            st.subheader("Last Bill Details")

            # Save last bill details with keyword "result1"
            result1 = last_bill

            # Display the details of the last bill
            st.write("**User ID:**", result1["User ID"])
            st.write("**Date and Time:**", result1["Date and Time"])
            st.write("**Items:**")
            for item in result1["Items"]:
                st.write(
                    f"- {item['Car Make']} {item['Car Model']}({item['Year']}) - {item['Part']}: ₹{item['Price']: .2f}")
            st.write("**Total Cost:**", f"₹{result1['Total Cost']:.2f}")
        else:
            st.warning("No bill records found.")
    except FileNotFoundError:
        st.error("The bills.json file does not exist.")


def get_last_entry_from_json(file_path):
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
            return data[-1] if data else None
    except FileNotFoundError:
        return None


def get_last_entry_from_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        return df.iloc[-1] if not df.empty else None
    except FileNotFoundError:
        return None


def get_last_entry_from_excel(file_path):
    try:
        df = pd.read_excel(file_path)
        return df.iloc[-1] if not df.empty else None
    except FileNotFoundError:
        return None


# Streamlit app
if not st.session_state.logged_in:
    st.title("WELCOME TO THE SMART SAFETY ASSESSMENT")
    st.subheader("To begin, please log in or register")

    # Option to select between login and register
    option = st.selectbox("Select an option", ["Login", "Register"])

    if option == "Login":
        username = st.text_input("Username", key="login_username")
        password = st.text_input(
            "Password", type="password", key="login_password")
        if st.button("Login", key="login_button"):
            if username.lower() == "admin" and password == "admin":
                st.session_state.logged_in = True
                st.session_state.username = "Admin"
                st.session_state.car_damage_page = True
                st.success("You have successfully logged in as Admin!")
                st.experimental_rerun()
            else:
                user = login_user(username, password)
                if user is not None:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.name = user['Name']
                    st.session_state.car_damage_page = True
                    st.success("You have successfully logged in!")
                    st.experimental_rerun()
                else:
                    st.error("Invalid username or password.")

    elif option == "Register":
        name = st.text_input("Name", key="register_name")
        username = st.text_input("Username", key="register_username")
        password = st.text_input(
            "Password", type="password", key="register_password")
        if st.button("Register", key="register_button"):
            if not name or not username or not password:
                st.warning("All fields are mandatory!")
            else:
                register_user(name, username, password)
                st.experimental_rerun()

else:
    # Initialize session state variables if they don't exist
    if 'car_damage_page' not in st.session_state:
        st.session_state.car_damage_page = False
    if 'reasons_page' not in st.session_state:
        st.session_state.reasons_page = False
    if 'drowsiness_detection_page' not in st.session_state:
        st.session_state.drowsiness_detection_page = False
    if 'other_reasons_page' not in st.session_state:
        st.session_state.other_reasons_page = False
    if 'demo_checking_page' not in st.session_state:
        st.session_state.demo_checking_page = False
    if 'automatic_checking_page' not in st.session_state:
        st.session_state.automatic_checking_page = False
    if 'result_page' not in st.session_state:
        st.session_state.result_page = False
    if 'cost_page' not in st.session_state:
        st.session_state.cost_page = False
    if 'cost_result_page' not in st.session_state:
        st.session_state.cost_result_page = False
    if 'bill_generated' not in st.session_state:
        st.session_state.bill_generated = False
    if 'scenario_generation_page' not in st.session_state:
        st.session_state.scenario_generation_page = False
    if 'story_page' not in st.session_state:  # New page initialization
        st.session_state.story_page = False
    if 'damage_confirmed' not in st.session_state:
        st.session_state.damage_confirmed = False

    
    # ✅ Confirmation page before going to car damage or drowsiness page
    if not st.session_state.damage_confirmed:
        st.title("CAR DAMAGE CONFIRMATION")
        st.subheader("Do you have any damage in the car?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes, there is damage"):
                st.session_state.damage_confirmed = True
                st.session_state.car_damage_page = True
                st.session_state.drowsiness_detection_page = False
                st.experimental_rerun()
        with col2:
            if st.button("No, Check Drowsiness"):
                st.session_state.damage_confirmed = True
                st.session_state.car_damage_page = False
                st.session_state.drowsiness_detection_page = True
                st.experimental_rerun()

    # Car Damage Assessment Page
    elif st.session_state.car_damage_page:
        st.title("Car Damage Assessment")
        st.write(
            f"Welcome to the Car Damage Assessment module, {st.session_state.username}!")

        # File uploader for car image
        file = st.file_uploader(
            "Please upload an image file", type=["jpg", "png"])
        if file is None:
            st.text("Please upload an image file for analysis.")
        else:
            image = Image.open(file)
            st.image(image, use_column_width=True)

            # Predict if it's a vehicle
            prediction_vehicle = import_and_predict_vehicle(
                image, model_vehicle)
            if np.argmax(prediction_vehicle) == 0:
                st.write("Detected: Vehicle!")
                st.text(f"Vehicle Probability: {prediction_vehicle[0][0]:.2f}")

                # Check if the vehicle is damaged
                prediction_damage = import_and_predict_damage(
                    image, model_damage)
                damage_prediction = "Damaged" if np.argmax(
                    prediction_damage) == 0 else "Not Damaged"
                damage_probability = prediction_damage[0][0] if np.argmax(
                    prediction_damage) == 0 else prediction_damage[0][1]
                st.write(f"Damage: {damage_prediction}")
                st.text(f"Probability: {damage_probability:.2f}")

                # Predict vehicle location
                prediction_location = import_and_predict_location(
                    image, model_location)
                location_index = np.argmax(prediction_location)
                location_name = class_names_location[location_index].strip()
                location_probability = prediction_location[0][location_index]
                st.write(f"Location: {location_name}")
                st.text(f"Location Probability: {location_probability:.2f}")

                # Only predict damage level if the vehicle is damaged
                if damage_prediction == "Damaged":
                    prediction_damage_level = import_and_predict_damage_level(
                        image, model_damage_level)
                    damage_level_index = np.argmax(prediction_damage_level)
                    damage_level_name = class_names_damage_level[damage_level_index].strip(
                    )
                    damage_level_probability = prediction_damage_level[0][damage_level_index]
                    st.write(f"Damage Level: {damage_level_name}")
                    st.text(
                        f"Damage Level Probability: {damage_level_probability: .2f}")
                else:
                    damage_level_name, damage_level_probability = "N/A", "N/A"

                # Prepare result for the current image
                assessment_result = [file.name, "Vehicle", prediction_vehicle[0][0],
                                     damage_prediction, damage_probability, location_name, location_probability,
                                     damage_level_name, damage_level_probability]
            else:
                st.write("Detected: Not a Vehicle.")
                assessment_result = [file.name, "Not a Vehicle",
                                     "", "", "", "", "", "N/A", "N/A"]

            # Create or load the Excel file
            excel_file = "assessment_results.xlsx"
            if os.path.isfile(excel_file):
                df = pd.read_excel(excel_file)
            else:
                df = pd.DataFrame(columns=["Image Name", "Vehicle or not", "Probability of Vehicle",
                                           "Damaged or not", "Probability", "Location", "Location Probability",
                                           "Damage Level", "Damage Level Probability"])

            # Append new result to the DataFrame
            new_df = pd.DataFrame([assessment_result], columns=df.columns)
            df = pd.concat([df, new_df], ignore_index=True)

            # Save the updated DataFrame back to the Excel file
            with pd.ExcelWriter(excel_file, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
                df.to_excel(writer, index=False)

            # Display the DataFrame in the app
            st.write("Current Assessment Results:")
            st.dataframe(df)

            # Provide a download button for the Excel file
            with open(excel_file, "rb") as f:
                st.download_button(
                    label="Download Assessment Results",
                    data=f,
                    file_name=excel_file,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            if st.button("Go to Reasons"):
                st.session_state.car_damage_page = False
                st.session_state.reasons_page = True
                st.experimental_rerun()

    # Reasons Page
    elif st.session_state.reasons_page:
        st.title("REASONS")
        if st.button("Go to Drowsiness Detection"):
            st.session_state.reasons_page = False
            st.session_state.drowsiness_detection_page = True
            st.experimental_rerun()
        if st.button("Go to Other Reasons"):
            st.session_state.reasons_page = False
            st.session_state.other_reasons_page = True
            st.experimental_rerun()
        if st.button("Back to Car Damage Assessment"):
            st.session_state.reasons_page = False
            st.session_state.car_damage_page = True
            st.experimental_rerun()

    # Drowsiness Detection Page
    elif st.session_state.drowsiness_detection_page:
        st.title("Welcome to Drowsiness Detection")

        # Button for automatic checking
        if st.button("Automatic Checking"):
            st.session_state.drowsiness_detection_page = False
            st.session_state.automatic_checking_page = True
            st.experimental_rerun()
            # Implement the functionality for Automatic Checking

        # Button for demo checking
        if st.button("Demo Checking"):
            st.session_state.drowsiness_detection_page = False
            st.session_state.demo_checking_page = True
            st.experimental_rerun()

        # Button to navigate back to reasons
        if st.button("Back to Reasons"):
            st.session_state.drowsiness_detection_page = False
            st.session_state.reasons_page = True
            st.experimental_rerun()

    # Other Reasons Page
    elif st.session_state.other_reasons_page:
        st.title("Other Reasons")
        # Your other reasons page content here
        if st.button("Back to Reasons"):
            st.session_state.other_reasons_page = False
            st.session_state.reasons_page = True
            st.experimental_rerun()

    # Demo Checking Page
    elif st.session_state.demo_checking_page:
        st.title("Demo Checking")
        # Call the main_drowsiness_detection function for drowsiness detection
        main_drowsiness_detection()
        # After the detection, add a button to view the results
        if st.button("Proceed to Results"):
            st.session_state.result_page = True
            st.session_state.demo_checking_page = False  # Reset demo checking page
            st.experimental_rerun()

        # Back to Drowsiness Detection Page
        if st.button("Back to Drowsiness Detection"):
            st.session_state.demo_checking_page = False
            st.session_state.drowsiness_detection_page = True
            st.experimental_rerun()

    # Automatic Checking Page
    elif st.session_state.automatic_checking_page:
        st.title("Automatic Checking")
        main_drowsiness_detection()

        # Back to Drowsiness Detection Page
        if st.button("Back to Drowsiness Detection"):
            st.session_state.automatic_checking_page = False
            st.session_state.drowsiness_detection_page = True
            st.experimental_rerun()

    # Results Page
    elif st.session_state.result_page:
        # Load the drowsiness results
        results = load_drowsiness_results()

        if results:
            # Get the last result from the list
            last_result = results[-1]

            # Display the last result
            st.title("RESULT FROM DROWSINESS")
            st.write("**Image Name:**", last_result.get("Image Name", "N/A"))
            st.write("**Eye Status:**", last_result.get("Eye Status", "N/A"))
            st.write("**Mouth Status:**",
                     last_result.get("Mouth Status", "N/A"))
            st.write("**Final Status:**",
                     last_result.get("Final Status", "N/A"))
            st.write("**Drowsy Time:**", last_result.get("Drowsy Time", "N/A"))
            st.write("**Yawning Time:**",
                     last_result.get("Yawning Time", "N/A"))

        # Button to proceed to the next step
        if st.button("Proceed to Next Step"):
            st.session_state.cost_page = True
            st.session_state.result_page = False  # Reset result page
            st.experimental_rerun()

        # Button to navigate back to the Demo Checking page
        if st.button("Back to Demo Checking"):
            st.session_state.result_page = False
            st.session_state.demo_checking_page = True
            st.experimental_rerun()

    # Cost Prediction Page
    elif st.session_state.cost_page:
        st.title("Car Part Price Prediction")

        # Initialize session state variables
        if "items_list" not in st.session_state:
            st.session_state["items_list"] = []
        if "total_cost" not in st.session_state:
            st.session_state["total_cost"] = 0.0
        if "user_id" not in st.session_state:
            st.session_state["user_id"] = str(
                uuid.uuid4())  # Unique ID for this session
        if "bill_generated" not in st.session_state:
            # Track if the bill is generated
            st.session_state["bill_generated"] = False

        # Load data and model for cost prediction
        data3 = cost.load_data()
        model3 = cost.load_or_train_model()

        # Streamlit UI for Car Details input
        st.header("Enter Car Details")
        car_make = st.selectbox("Select Car Make", options=[
                                ""] + list(data3["Car Make"].unique()))
        car_model = st.selectbox("Select Car Model", options=[
            ""] + list(data3[data3["Car Make"] == car_make]["Car Model"].unique()) if car_make else [""])
        year = st.selectbox("Select Year", options=[
                            ""] + sorted(data3["Year"].unique()))
        part = st.selectbox("Select Part", options=[
                            ""] + list(data3["Part"].unique()))

        # Predict and add item to cart
        if car_make and car_model and year and part:
            input_data = pd.DataFrame([[car_make, car_model, year, part]], columns=[
                "Car Make", "Car Model", "Year", "Part"])
            input_data = pd.get_dummies(
                input_data, columns=["Car Make", "Car Model", "Part"], drop_first=True)

            # Ensure all columns are in input_data, even if they are missing
            input_data = input_data.reindex(
                columns=model3.feature_names_in_, fill_value=0)

            if st.button("Add Item to Cart"):
                prediction = model3.predict(input_data)[0]
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
                    st.write(
                        f"Added {part} for {car_make} {car_model}({year}) at ₹{prediction: .2f}")
                else:
                    st.warning("This item is already in the cart.")
                st.experimental_rerun()

        # Display bill summary
        st.subheader("Bill Summary")
        if st.session_state["items_list"]:
            for i, item in enumerate(st.session_state["items_list"], 1):
                col1, col2 = st.columns([0.9, 0.1])
                with col1:
                    st.write(
                        f"{i}. {item['Car Make']} {item['Car Model']}({item['Year']}) - {item['Part']}: ₹{item['Price']: .2f}")
                with col2:
                    if st.button("❌", key=f"remove_{i}"):
                        st.session_state["total_cost"] -= item["Price"]
                        st.session_state["items_list"].remove(item)
                        st.experimental_rerun()
            st.write(f"**Total Cost: ₹{st.session_state['total_cost']:.2f}**")

        # Finish and generate bill
        if st.button("Finish and Generate Bill"):
            # Prepare the bill data
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

            # Append the new bill to the existing data
            bills_data.append(bill)

            # Save the updated bills data back to the file
            with open(bills_file, "w") as f:
                json.dump(bills_data, f, indent=4)

            # Clear session state for a new transaction
            st.session_state["items_list"] = []
            st.session_state["total_cost"] = 0.0
            st.session_state.bill_generated = True  # Set to True after saving the bill
            st.success("Bill saved to bills.json")

        # Show the "Proceed to Cost Result Page" button only after the bill is generated
        if st.session_state.bill_generated:
            if st.button("Proceed to Cost Result Page"):
                st.session_state.cost_result_page = True
                st.session_state.cost_page = False
                st.experimental_rerun()

    # Cost Result Page
    elif st.session_state.cost_result_page:
        st.title("COST RESULT PAGE")
        show_last_bill()
        # Add a button to navigate to the Scenario Generation Page
        if st.button("Proceed to Scenario Generation"):
            st.session_state.cost_result_page = False
            st.session_state.scenario_generation_page = True
            st.experimental_rerun()

    # Scenario Generation Page
    elif st.session_state.scenario_generation_page:
        st.title("SCENARIO GENERATION")

        # Retrieve data from files
        last_bill = get_last_entry_from_json(bills_file)
        last_drowsiness_result = get_last_entry_from_json(
            drowsiness_results_file)
        last_user = get_last_entry_from_csv(users_file)
        last_assessment = get_last_entry_from_excel(assessment_results_file)

        # Convert Series to dict for JSON serialization
        if isinstance(last_assessment, pd.Series):
            last_assessment = last_assessment.to_dict()
        if isinstance(last_bill, pd.Series):
            last_bill = last_bill.to_dict()
        if isinstance(last_drowsiness_result, pd.Series):
            last_drowsiness_result = last_drowsiness_result.to_dict()

        # Display the last entries
        st.subheader("Last Entries from Files")
        st.write("### Last Bill Details:")
        st.json(last_bill)

        st.write("### Last Drowsiness Result:")
        st.json(last_drowsiness_result)

        if last_user is not None:
            st.write("### Last User Details:")
            st.write(
                f"**Name:** {last_user['Name']}, **ID:** {last_user['ID']}")

        st.write("### Last Assessment Result:")
        st.write(last_assessment)

        # Use GPT-4 API to narrate a story
        if st.button("Generate Story"):
            # Save the retrieved data in session state for use on the story page
            st.session_state.last_bill = last_bill
            st.session_state.last_drowsiness_result = last_drowsiness_result
            st.session_state.last_user = last_user
            st.session_state.last_assessment = last_assessment
            # Construct the story prompt
            story_prompt = f"""
                Analyze the following data:

                1. **Last Bill:** {json.dumps(last_bill, indent=2)}
                2. **Last Drowsiness Result:** {json.dumps(last_drowsiness_result, indent=2)}
                3. **Last User (Name, ID):** {last_user['Name']}, {last_user['ID']}
                4. **Last Assessment Result:** {json.dumps(last_assessment, indent=2)}

                Narrate a story based on the above details:

                - What happened during the event (for drowsiness detection, mention if the driver was drowsy or awake)?
                - How did the drowsiness or damage situation unfold during the journey or assessment?
                - Who was involved in the situation (driver, vehicle, etc.)?
                - How can such incidents be prevented in the future?

                For drowsiness detection, please mention whether the driver remained drowsy after multiple alerts or was successfully awakened.
                For damage detection, describe the extent and location of the damage (e.g., side, front, or rear of the car).

                Provide a coherent narrative that answers these questions.
                """

            # Get GPT-4 response
            story = get_gpt4_response(story_prompt)

            # Store the story in session state
            st.session_state.generated_story = story

            # Navigate to the story page
            st.session_state.scenario_generation_page = False
            st.session_state.story_page = True
            st.experimental_rerun()

    # New Story Page for displaying the generated story
    elif st.session_state.story_page:
        st.title("Generated Story")

        # Retrieve and display the generated story
        story = st.session_state.get('generated_story', 'No story generated.')
        st.subheader("Generated Story")
        st.write(story)

        # Button to generate the PDF
        if st.button("Generate PDF"):
            # Retrieve dynamic data from session state
            last_user = st.session_state.get(
                'last_user', {"Name": "Unknown", "ID": "Unknown"})
            last_assessment = st.session_state.get('last_assessment', {})
            last_drowsiness_result = st.session_state.get(
                'last_drowsiness_result', {})
            last_bill = st.session_state.get('last_bill', {})

            # Prepare data for the PDF
            user_details = {
                "Name": last_user.get("Name", "Unknown"),
                "ID": last_user.get("ID", "Unknown")
            }

            damage_status = {
                "Damaged": last_assessment.get("Damaged or not", "Unknown"),
                "Location": last_assessment.get("Location", "Unknown"),
                "Damage Level": last_assessment.get("Damage Level", "Unknown")
            }

            drowsiness_status = {
                "Eye Status": last_drowsiness_result.get("Eye Status", "Unknown"),
                "Mouth Status": last_drowsiness_result.get("Mouth Status", "Unknown"),
                "Final Status": last_drowsiness_result.get("Final Status", "Unknown")
            }

            bill_details = {
                "User ID": last_bill.get("User ID", "Unknown"),
                "Date and Time": last_bill.get("Date and Time", "Unknown"),
                "Items": last_bill.get("Items", []),
                "Total Cost": last_bill.get("Total Cost", 0)
            }

            # Generate PDF
            pdf_path = create_pdf(user_details, damage_status,drowsiness_status, bill_details, story)

            # Provide PDF download link
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="Download PDF",
                    data=f,
                    file_name="Scenario_Report.pdf",
                    mime="application/pdf"
                )

        # Button to go back to the scenario generation page
        if st.button("Back to Scenario Generation"):
            st.session_state.scenario_generation_page = True
            st.session_state.story_page = False
            st.experimental_rerun()
