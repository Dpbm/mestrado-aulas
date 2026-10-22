"""
Code used across multiple notebooks.
"""

import os
from pathlib import Path

RANDOM_STATE = 42
TEST_SPLIT = 0.2

DATASET_HANDLE = 'alzheimers-disease-risk-prediction-eu-business'

# just to ensure that the dataset/ folder will always exist
os.makedirs("dataset", exist_ok=True)

current_file = Path(__file__).resolve().parent
DATASET_TRAIN_FILE = os.path.join(current_file, "dataset", "train.csv")
DATASET_EVAL_FILE = os.path.join(current_file, "dataset", "comp.csv")

PLOTS_PATH = lambda img_file : os.path.join(".", "plots", img_file)


TARGET = "Diagnosis"

RAW_COLUMNS = [
    'Age', 'Gender', 'EducationLevel', 'BMI',
    'Smoking', 'AlcoholConsumption', 'PhysicalActivity', 'DietQuality',
    'SleepQuality', 'FamilyHistoryAlzheimers', 'CardiovascularDisease',
    'Diabetes', 'Depression', 'HeadInjury', 'Hypertension', 'SystolicBP',
    'DiastolicBP', 'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL',
    'CholesterolTriglycerides', 'MMSE', 'FunctionalAssessment',
    'MemoryComplaints', 'BehavioralProblems', 'ADL', 'Confusion',
    'Disorientation', 'PersonalityChanges', 'DifficultyCompletingTasks',
    'Forgetfulness',
]
OURS_COLUMNS = ["BMI_cat", "T/HDL", "health_points", "secondary_indicator"]
CAIDE_COLUMNS = [
    'CAIDE', 'BMI', 'Smoking', 'AlcoholConsumption', 'DietQuality',
    'SleepQuality', 'FamilyHistoryAlzheimers', 'CardiovascularDisease',
    'Diabetes', 'Depression', 'HeadInjury', 'Hypertension', 
    'DiastolicBP', 'CholesterolLDL', 'CholesterolHDL',
    'CholesterolTriglycerides', 'MMSE', 'FunctionalAssessment',
    'MemoryComplaints', 'BehavioralProblems', 'ADL', 'Confusion',
    'Disorientation', 'PersonalityChanges', 'DifficultyCompletingTasks',
    'Forgetfulness',
]