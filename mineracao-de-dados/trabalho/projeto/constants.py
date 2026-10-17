"""
Code used across multiple notebooks.
"""

import os

RANDOM_STATE = 42
DATASET_HANDLE = 'alzheimers-disease-risk-prediction-eu-business'
PLOTS_PATH = lambda img_file : os.path.join(".", "plots", img_file)