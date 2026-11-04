"""Streamlit dashboard"""

from joblib import load
import os

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer

from constants import RANDOM_STATE
from format_data import format_columns,normalize_data

MODEL_PATH = os.path.join("model", "model.pkl")

@st.cache_resource
def load_model():
    with open(MODEL_PATH, 'rb') as file:
        model = load(file)
    return model

model = load_model()

if "loading" not in st.session_state:
    st.session_state.loading = False
if "probs" not in st.session_state:
    st.session_state.probs = None


def disable_button():
    st.session_state.loading = True

def predict(
    fa:float,
    adl:float,
    mmse:float,
    mc:bool,
    bp:bool
):
    column_trans = ColumnTransformer([
        ("cols", StandardScaler(),[
            'MMSE', 'FunctionalAssessment', 'ADL'
        ])
    ],
    remainder="passthrough",
    verbose_feature_names_out=False)
    column_trans.set_output(transform="pandas")

    df = pd.DataFrame({
        "FunctionalAssessment":fa,
        "ADL":adl,
        "MemoryComplaints":float(mc),
        "MMSE":mmse,
        "BehavioralProblems":float(bp)
    }, index=[0])
    df = column_trans.fit_transform(df)
    probs = model.predict_proba(df[model.feature_names_in_])
    st.session_state.probs = probs[0]

# ----- DASHBOARD WIDGETS ---------------------------------------

st.title("Stacking model for Alzheimer's prediction")

st.write(f"""
The model is based on Stacking with Random Forest descriminators and Logistic regression as the meta model.
All models using default parameters from Scikit-Learn and seed={RANDOM_STATE}.
The model was trained with the data from https://www.kaggle.com/competitions/alzheimers-disease-risk-prediction-eu-business/overview
and oversampled via SMOTE.
""")


with st.container(border=True):
    st.header("Submit parameters for the model")

    cols = st.columns(4, vertical_alignment="bottom")

    functional_assessment = cols[0].number_input("Functional Assessment", min_value=0.0, max_value=10.0, step=0.01)
    adl = cols[1].number_input("Activities of Daily Living score (ADL)", min_value=0.0, max_value=10.0, step=0.01)
    mmse = cols[2].number_input("Mini-Mental State Exam score", min_value=0.0, max_value=30.0, step=0.01)

    with cols[3].container() :
        memory_complaints = st.checkbox("Memory Complaints?")
        behavioral_problems = st.checkbox("Behavioral Problems?")
    
    st.caption("These parameters were selected using SHAP.")

    st.button(
        "Submit", 
        width="stretch", 
        disabled=st.session_state.loading,
        on_click=disable_button
        )

    if st.session_state.loading:
        with st.spinner("Making predictions...", width="stretch"):
            predict(
                functional_assessment,
                adl,
                mmse,
                memory_complaints,
                behavioral_problems
            )
            st.session_state.loading = False
            st.rerun()
    
if not st.session_state.loading and st.session_state.probs is not None:
    with st.container(border=True):
        st.header("Model prediction - Probabilities")

        pred_cols = st.columns(2)

        for i,c in enumerate(model.classes_):
            class_name = "Has Alzheimer" if c == 1 else "Hasn't Alzheimer"
            pred_cols[i].metric(
                class_name, 
                np.round(st.session_state.probs[i],decimals=2), 
                border=True)
        st.session_state.probs = None