"""Streamlit dashboard"""

from joblib import load, dump
import os
import io

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer

from constants import RANDOM_STATE
from format_data import format_columns,normalize_data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR,"model", "model.pkl")
TRAIN_X_PATH = os.path.join(BASE_DIR,"model", "x_train.npy")
TRAIN_Y_PATH = os.path.join(BASE_DIR,"model", "y_train.npy")
CLASSES = {
    0:"Doesn't Have Alzheimer",
    1:"Has Alzheimer",
}

@st.cache_resource
def load_model():
    with open(MODEL_PATH, 'rb') as file:
        model = load(file)
    return model

def init_params():
    print("[i] Initializing Parameters...")

    keys_params = {
        'loading':False,
        "loading_update": False,
        "probs":None,
        "predicted_class":None,
        "params":None,
        'fa':0.0,
        'adl':0.0,
        'mmse':0.0,
        'mc':False,
        'bp':False,
        'has_voted':False,
        'modified_model':False
    }

    for k,default_v in keys_params.items():
        if not k in st.session_state:
            st.session_state[k] = default_v

# ----- init data --------------
init_params()
model = load_model()

# ----- auxiliary functions ----

def start_loading():
    st.session_state.loading = True

def start_loading_update():
    st.session_state.loading_update = True

def predict():
    print("[i] Starting predictions...")

    column_trans = ColumnTransformer([
        ("cols", StandardScaler(),[
            'MMSE', 'FunctionalAssessment', 'ADL'
        ])
    ],
    remainder="passthrough",
    verbose_feature_names_out=False)
    column_trans.set_output(transform="pandas")

    df = pd.DataFrame({
        "FunctionalAssessment":st.session_state.fa,
        "ADL":st.session_state.adl,
        "MemoryComplaints":float(st.session_state.mc),
        "MMSE":st.session_state.mmse,
        "BehavioralProblems":float(st.session_state.bp)
    }, index=[0])
    df = column_trans.fit_transform(df)
    probs = np.round(model.predict_proba(df[model.feature_names_in_])[0], decimals=2)
            
    pred_class = None
    higher_prob = 0.5

    for i,c in enumerate(model.classes_):
        if probs[i] > higher_prob:
            higher_prob = probs[i]
            pred_class = c

    st.session_state.probs = probs
    st.session_state.predicted_class = pred_class
    st.session_state.params = df.to_numpy()
    st.session_state.loading = False
    st.session_state.has_voted = False
    st.rerun()

def update_model():
    if st.session_state.pred_change == 'Yes':
        st.session_state.has_voted = True
        st.session_state.loading_update = False
        st.rerun()
        return

    x_train = np.load(TRAIN_X_PATH)
    y_train = np.load(TRAIN_Y_PATH)

    X = np.concatenate([x_train, st.session_state.params])
    y = np.concatenate([y_train, [float((st.session_state.predicted_class + 1)%2)]])

    # It would be better to incrementally train it, but our Stacking model has no
    # partial_fit, so we need to do this way
    model.fit(X,y)

    st.session_state.modified_model =True
    st.session_state.probs = None
    st.session_state.predicted_class = None
    st.session_state.params = None
    st.session_state.loading_update = False
    st.session_state.has_voted = True

    st.rerun()

# ----- DASHBOARD WIDGETS ---------------------------------------

st.title("Stacking model for Alzheimer's prediction")

st.write(f"""
The model is based on Stacking with Random Forest descriminators and Logistic regression as the meta model.
All models using default parameters from Scikit-Learn and seed={RANDOM_STATE}.
The model was trained with the data from https://www.kaggle.com/competitions/alzheimers-disease-risk-prediction-eu-business/overview
and oversampled via SMOTE.
""")


with st.form("parameters_form", border=True):
    st.header("Submit parameters for the model")

    cols = st.columns(4, vertical_alignment="bottom")

    cols[0].number_input(
        "Functional Assessment", 
        min_value=0.0, 
        max_value=10.0, 
        step=0.01,
        key="fa",
        persist_state="session")

    cols[1].number_input(
        "Activities of Daily Living score (ADL)", 
        min_value=0.0, 
        max_value=10.0, 
        step=0.01,
        key="adl",
        persist_state="session")

    cols[2].number_input(
        "Mini-Mental State Exam score", 
        min_value=0.0, 
        max_value=30.0, 
        step=0.01,
        key="mmse",
        persist_state="session")

    with cols[3].container() :
        st.checkbox(
            "Memory Complaints?",
            key="mc",
            persist_state="session"
        )
        st.checkbox(
            "Behavioral Problems?",
            key="bp",
            persist_state="session"
        )
    
    st.caption("These parameters were selected using SHAP.")

    st.form_submit_button(
        "Submit", 
        width="stretch", 
        disabled=st.session_state.loading,
        on_click=start_loading,
    )

    if st.session_state.loading:
        with st.spinner("Making predictions...", width="stretch"):
            predict()
    
if not st.session_state.loading and st.session_state.probs is not None:
    with st.container(border=True):
        st.header("Model prediction - Probabilities")

        pred_cols = st.columns(2)


        for i,c in enumerate(model.classes_):
            pred_cols[i].metric(
                CLASSES[c], 
                f'~{st.session_state.probs[i]}', 
                border=True)
            
        st.caption("*Rounded to two decimal places.")

        if st.session_state.predicted_class is None:
            st.write("The model couldn't predict the class correctly!")
        else:
            st.write(f"The model decided that it '{CLASSES[st.session_state.predicted_class]}'")

        if not st.session_state.has_voted:
            with st.form("update_form", border=True):
                st.header("Update Predictions")

                st.radio(
                    "Does the model made the correct prediction for the given parameters?", 
                    ["Yes", "No"], 
                    width="stretch", 
                    horizontal=True, 
                    key="pred_change",
                    persist_state='session')

                st.form_submit_button(
                    width='stretch', 
                    on_click=start_loading_update,
                    disabled=st.session_state.loading_update)
        
                if st.session_state.loading_update:
                    with st.spinner("Updating Model...", width="stretch"):
                        update_model()
        else:
            with st.container(border=True):
                st.header("Thanks for your contribution!")
                st.write("You vote is essential for the developmente of this model.")



if st.session_state.modified_model:
    with st.container(border=True, horizontal=True, vertical_alignment='bottom'):
        st.write("Download updated model: ")

        buffer = io.BytesIO()
        dump(model, buffer, protocol=5)

        st.download_button(
            label="Download model",
            data=buffer,
            file_name="new_model.pkl",
            mime="application/octet-stream"
        )
