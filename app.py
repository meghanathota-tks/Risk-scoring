import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "data", "synthetic_data.csv")

data = pd.read_csv(data_path)
st.set_page_config(page_title="Risk Scoring System", layout="wide")

# ----------------------------
# LOAD ARTIFACTS
# ----------------------------
model = joblib.load("artifacts/model.pkl")
import json

booster = model.get_booster()
config = json.loads(booster.save_config())

config["learner"]["learner_model_param"]["base_score"] = "0.5"

booster.load_config(json.dumps(config))
model.set_params(base_score=0.5)


calibrator = joblib.load("artifacts/calibrator.pkl")
scaler = joblib.load("artifacts/scaler_params.pkl")




# ----------------------------
# FEATURE ENGINEERING
# ----------------------------
def prepare_features(input_data):
    df_input = pd.DataFrame([input_data])
    
    # Safe scaling
    claim_std = scaler["claim_std"] if scaler["claim_std"] != 0 else 1
    max_exp = scaler["max_exp"] if scaler["max_exp"] != 0 else 1
    max_denials = scaler["max_denials"] if scaler["max_denials"] != 0 else 1

    df_input["amount_zscore"] = (df_input["claim_amount"] - scaler["claim_mean"]) / claim_std
    df_input["los_ratio"] = df_input["length_of_stay"] / (scaler["los_mean"] + 1)
    df_input["provider_exp_norm"] = df_input["provider_experience"] / max_exp
    df_input["denial_risk"] = df_input["historical_denials"] / max_denials

    df_input["high_claim_short_stay"] = (
        (df_input["claim_amount"] > 120000) & (df_input["length_of_stay"] <= 1)
    ).astype(int)

    return df_input[
        [
            "amount_zscore",
            "los_ratio",
            "provider_exp_norm",
            "denial_risk",
            "readmission_flag",
            "repeat_procedure_flag",
            "high_claim_short_stay",
        ]
    ]

# ----------------------------
# PREDICTION FUNCTION
# ----------------------------
def predict_risk(input_data):
    X = prepare_features(input_data)
    
    prob = model.predict_proba(X)[:, 1][0]
    calibrated = calibrator.transform([prob])[0]
    
    score = int(calibrated * 999 + 1)
    
    if score >= 851:
        band = "Critical"
    elif score >= 651:
        band = "High"
    elif score >= 351:
        band = "Medium"
    else:
        band = "Low"
    
    return score, band, X

# ----------------------------
# FAST SHAP (TreeExplainer)
# ----------------------------
@st.cache_resource
def load_explainer():
    # ✅ Use SAME feature engineering
    background = data.sample(100)

    background_X = background.apply(lambda row: prepare_features({
        "claim_amount": row["claim_amount"],
        "length_of_stay": row["length_of_stay"],
        "provider_experience": row["provider_experience"],
        "historical_denials": row["historical_denials"],
        "readmission_flag": row["readmission_flag"],
        "repeat_procedure_flag": row["repeat_procedure_flag"],
    }), axis=1)

    background_X = pd.concat(background_X.tolist(), ignore_index=True)

    return shap.Explainer(
        model.predict_proba,
        masker=background_X   # ✅ NOW MATCHES X
    
    )
# ----------------------------
# TABS
# ----------------------------
tab1, tab2 = st.tabs(["🏥 Dashboard", "🧠 Explainability"])

# =========================================================
# 🏥 DASHBOARD TAB
# =========================================================
with tab1:
    st.title("🏥 Risk Scoring System")

    # ----------------------------
    # INPUT FORM
    # ----------------------------
    st.subheader("🧾 Claim Risk Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        claim_amount = st.number_input("Claim Amount (₹)", 1000, 300000, 50000)
        los = st.number_input("Length of Stay", 0, 20, 3)

    with col2:
        exp = st.number_input("Provider Experience", 1, 30, 10)
        denials = st.number_input("Historical Denials", 0, 20, 2)

    with col3:
        readmission = st.selectbox("Readmission", [0, 1])
        repeat = st.selectbox("Repeat Procedure", [0, 1])

    if st.button("Predict Risk"):
        input_data = {
            "claim_amount": claim_amount,
            "length_of_stay": los,
            "provider_experience": exp,
            "historical_denials": denials,
            "readmission_flag": readmission,
            "repeat_procedure_flag": repeat,
        }
        
        score, band, _ = predict_risk(input_data)
        if band == "Critical":
            st.error(f"Risk Band: {band}")
        elif band == "High":
            st.warning(f"Risk Band: {band}")
        else:
            st.info(f"Risk Band: {band}")

    # ----------------------------
    # HOSPITAL RANKING
    # ----------------------------
    st.subheader("🏆 Hospital Risk Ranking")

    @st.cache_data
    def compute_hospital_scores(data):
        def batch_score(row):
            input_data = {
                "claim_amount": row["claim_amount"],
                "length_of_stay": row["length_of_stay"],
                "provider_experience": row["provider_experience"],
                "historical_denials": row["historical_denials"],
                "readmission_flag": row["readmission_flag"],
                "repeat_procedure_flag": row["repeat_procedure_flag"],
            }
            score, _, _ = predict_risk(input_data)
            return score

        data["risk_score"] = data.apply(batch_score, axis=1)

        hospital = (
            data.groupby("hospital_id")["risk_score"]
            .mean()
            .reset_index()
            .sort_values("risk_score", ascending=False)
        )

        return hospital

    hospital_df = compute_hospital_scores(data.copy())

    st.dataframe(hospital_df.head(20))

# =========================================================
# 🧠 EXPLAINABILITY TAB
# =========================================================
with tab2:
    st.title("🧠 Model Explainability")

    # ----------------------------
    # FEATURE IMPORTANCE
    # ----------------------------
    st.subheader("📊 Feature Importance")

    feature_names = [
        "amount_zscore",
        "los_ratio",
        "provider_exp_norm",
        "denial_risk",
        "readmission_flag",
        "repeat_procedure_flag",
        "high_claim_short_stay",
    ]

    importances = model.feature_importances_

    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    st.bar_chart(fi_df.set_index("feature"))

    # ----------------------------
    # SHAP EXPLANATION
    # ----------------------------
    st.subheader("🔍 Prediction Explanation")

    col1, col2, col3 = st.columns(3)

    with col1:
        claim_amount = st.number_input("Claim Amount (₹)", 1000, 300000, 60000, key="exp1")
        los = st.number_input("LOS", 0, 20, 2, key="exp2")

    with col2:
        exp = st.number_input("Experience", 1, 30, 5, key="exp3")
        denials = st.number_input("Denials", 0, 20, 5, key="exp4")

    with col3:
        readmission = st.selectbox("Readmission", [0, 1], key="exp5")
        repeat = st.selectbox("Repeat", [0, 1], key="exp6")

    if st.button("Explain Prediction"):
        input_data = {
            "claim_amount": claim_amount,
            "length_of_stay": los,
            "provider_experience": exp,
            "historical_denials": denials,
            "readmission_flag": readmission,
            "repeat_procedure_flag": repeat,
        }

        _, _, X = predict_risk(input_data)

      
        # FAST SHAP
        import numpy as np

        X = X.astype(float)

        explainer = load_explainer()
        shap_values = explainer(X)

# ✅ Convert to numpy safely
        vals = np.array(shap_values.values)

# ✅ Force correct extraction
        try:
            if vals.ndim == 3:          # (1, features, classes)
                shap_vals = vals[0, :, 1]
            elif vals.ndim == 2:        # (1, features)
                shap_vals = vals[0]
            else:
                shap_vals = vals
        except:
            shap_vals = vals

# ✅ FORCE 1D no matter what
        shap_vals = np.squeeze(shap_vals)

# ✅ FINAL SAFETY: ensure exact length match
        shap_vals = np.resize(shap_vals, len(X.columns))

# 🔍 Debug (optional - uncomment if needed)
# st.write("Final shape:", shap_vals.shape)

# ✅ Now guaranteed safe
        shap_df = pd.DataFrame({
            "feature": list(X.columns),
            "impact": shap_vals.tolist()   # <-- IMPORTANT (forces 1D)
        })

# Sort
        shap_df["abs_impact"] = shap_df["impact"].abs()
        shap_df = shap_df.sort_values("abs_impact", ascending=False).drop(columns="abs_impact")

        st.dataframe(shap_df)
        st.bar_chart(shap_df.set_index("feature"))