import pandas as pd
import numpy as np
import random
import joblib
from sklearn.model_selection import train_test_split
from sklearn.calibration import IsotonicRegression
from xgboost import XGBClassifier

np.random.seed(42)
n = 10000

hospital_ids = [f"HOSP_{i}" for i in range(1, 101)]

data = []

for i in range(n):
    claim_amount = np.random.randint(2000, 300000)
    los = np.random.randint(0, 15)
    provider_exp = np.random.randint(1, 30)
    denials = np.random.randint(0, 20)
    
    readmission = np.random.choice([0,1], p=[0.85, 0.15])
    repeat_proc = np.random.choice([0,1], p=[0.9, 0.1])
    
    fraud = (
        (claim_amount > 150000 and los <= 1) or
        (denials > 10) or
        (repeat_proc == 1 and readmission == 1)
    )
    
    data.append([
        f"CLM_{i}",
        random.choice(hospital_ids),
        claim_amount, los, provider_exp, denials,
        readmission, repeat_proc, int(fraud)
    ])

df = pd.DataFrame(data, columns=[
    "claim_id","hospital_id","claim_amount","length_of_stay",
    "provider_experience","historical_denials",
    "readmission_flag","repeat_procedure_flag","fraud_flag"
])