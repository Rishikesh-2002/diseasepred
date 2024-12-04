import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
from streamlit_option_menu import option_menu
import pickle
from PIL import Image
import numpy as np
import plotly.figure_factory as ff

from DiseaseModel import DiseaseModel
from helper import prepare_symptoms_array

# sidebar
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction', [
        'Disease Prediction'],
        icons=['', 'activity', 'bar-chart-fill'],
        default_index=0)

# multiple disease prediction
if selected == 'Disease Prediction':
    # Create disease class and load ML model
    disease_model = DiseaseModel()
    disease_model.load_xgboost('xgboost_model.json')

    # Title
    st.write('# Disease Prediction using Machine Learning')

    symptoms = st.multiselect('What are your symptoms?', options=disease_model.all_symptoms)

    X = prepare_symptoms_array(symptoms)

    # Trigger XGBoost model
    if st.button('Predict'):
        # Run the model with the python script
        prediction, prob = disease_model.predict(X)
        st.write(f'## Disease: {prediction} with {prob * 100:.2f}% probability')

        tab1, tab2 = st.tabs(["Description", "Precautions"])

        with tab1:
            st.write(disease_model.describe_predicted_disease())

        with tab2:
            precautions = disease_model.predicted_disease_precautions()
            for i in range(4):
                st.write(f'{i+1}. {precautions[i]}')