import streamlit as st
import pickle
import pandas as pd
import numpy as np

# --- Load your data for choices
@st.cache_data
def load_data():
    df = pd.read_csv("salary_prediction_data.csv")
    return df

data = load_data()

# --- Unique choices from the dataset (use exact column names)
education_list = sorted(data['Education'].dropna().unique())
location_list = sorted(data['Location'].dropna().unique())
jobtitle_list = sorted(data['Job_Title'].dropna().unique())
gender_list = sorted(data['Gender'].dropna().unique())

# --- Load model and columns (make sure the files exist and are trained with the right columns!)
with open('salary_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('model_columns.pkl', 'rb') as f:
    model_columns = pickle.load(f)
st.title("💼 Salary Prediction App")
st.write("Enter your information and get an estimated salary below! 🚀")

# --- Form for user input
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        experience = st.number_input("Years of Experience:", int(data["Experience"].min()), int(data["Experience"].max()), int(data["Experience"].mean()))
        age = st.number_input("Age:", int(data["Age"].min()), int(data["Age"].max()), int(data["Age"].mean()))
        education = st.selectbox("Education Level:", education_list)
    with col2:
        location = st.selectbox("Location:", location_list)
        job_title = st.selectbox("Job Title:", jobtitle_list)
        gender = st.selectbox("Gender:", gender_list)
    submitted = st.form_submit_button("Predict Salary")

# --- Prepare input
input_dict = {
    'Education': education,
    'Experience': experience,
    'Location': location,
    'Job_Title': job_title,   # column name as in your CSV
    'Age': age,
    'Gender': gender
}
input_df = pd.DataFrame([input_dict])
input_encoded = pd.get_dummies(input_df)
for col in model_columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[model_columns]

# --- Predict and display
if submitted:
    prediction = model.predict(input_encoded)
    st.success(f"💰 Estimated Salary: {prediction[0]:,.2f}")
    st.balloons()

# --- Batch prediction: Upload CSV
st.markdown("---")
st.subheader("Batch Prediction: Upload your CSV")
upload = st.file_uploader("Upload your CSV", type=['csv'])
if upload is not None:
    batch_df = pd.read_csv(upload)
    batch_in = pd.get_dummies(batch_df)
    for col in model_columns:
        if col not in batch_in.columns:
            batch_in[col] = 0
    batch_in = batch_in[model_columns]
    batch_pred = model.predict(batch_in)
    batch_df['Predicted Salary'] = batch_pred
    st.dataframe(batch_df)
    csv = batch_df.to_csv(index=False).encode()
    st.download_button("Download Predictions", csv, "batch_predictions.csv", "text/csv")

st.markdown("---")
st.markdown("<p style='font-size:16px;text-align:center;'>Made with ❤ using Streamlit</p>", unsafe_allow_html=True)