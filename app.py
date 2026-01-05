import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler,LabelEncoder

def load_model():
    with open("linear_regression_model.pkl","rb") as file:
        model,scaler,le = pickle.load(file)
    return model,scaler,le

def preprocessing_input_data(data,scaler,le):
    data["Extracurricular Activities"] = le.transform([data["Extracurricular Activities"]])
    df = pd.DataFrame([data])
    df_transformed = scaler.transform(df)
    return df_transformed

def predict_data(data):
    model,scaler,le = load_model()
    processed_data = preprocessing_input_data(data,scaler,le) 
    prediction = model.predict(processed_data)
    return prediction

def main():
    st.title("Student Performance Prediction")
    st.write("Enter your data toget prediction for your performance")
    hours_studied = st.number_input("Hours Studied",min_value = 1,max_value = 10,value = 5)
    previous_score = st.number_input("previous score",min_value = 1,max_value = 100,value = 70)
    extra = st.selectbox("Extra Curricular Activity",["Yes","No"])
    sleeping_hours = st.number_input("Sleeping Hous",min_value = 1,max_value = 10,value = 5)
    paper_solved = st.number_input("No Of QUestion Paper SOlved",min_value = 1,max_value = 10,value = 5)
    if st.button("Predict Your Score"):
        user_data = {
            "Hours Studied":hours_studied,
            "Previous Scores":previous_score,
            "Extracurricular Activities":extra,
            "Sleep Hours":sleeping_hours,
            "Sample Question Papers Practiced":paper_solved
        }
        prediction = predict_data(user_data)
        st.success(f"Your Prediction Result is {prediction}")
        
if __name__=="__main__":
    main()

    
    
       