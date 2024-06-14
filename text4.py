# # !pip install streamlit transformers
# !pip install streamlit transformers

import streamlit as st
from transformers import pipeline

# Load your pre-trained models
gender_classifier = pipeline('text-classification', model='priyabrat/gender', truncation=True)
age_classifier = pipeline('text-classification', model='priyabrat/AGE_predict_model', truncation=True)

# Define a function to predict gender
def predict_gender(text):
    data = gender_classifier(text)
    label = [d['label'] for d in data if 'label' in d]
    score = [d['score'] for d in data if 'score' in d]
    score1 = score[0] * 100
    if label[0] == "LABEL_0":
        label1 = "female"
    else:
        label1 = "male"
    score2 = 100 - score1
    if label1 == "male":
        label2 = "female"
    else:
        label2 = "male"
    return label1, round(score1), label2, round(score2)

# Define a function to predict age
def predict_age(text):
    data = age_classifier(text)
    label = [d['label'] for d in data if 'label' in d]
    score = [d['score'] for d in data if 'score' in d]
    return label[0], round(score[0] * 100)

# Streamlit app
st.title("Target Audience Analyzer for Blog Writers")

# Subtitle and explanation
st.header("Understand Your Audience")
st.write("Enter the blog content below to analyze the age group and gender that your writing is most likely to appeal to.")

# Text area for user input
text = st.text_area("Paste your blog content here:")

if st.button('Analyze Audience'):
    # Predict gender
    gender_prediction = predict_gender(text)
    st.write(f"Predicted Gender Preference: {gender_prediction[0]} with a confidence of {gender_prediction[1]}%")
    
    # Predict age
    age_prediction = predict_age(text)
    st.write(f"Predicted Age Group Preference: {age_prediction[0]} with a confidence of {age_prediction[1]}%")
