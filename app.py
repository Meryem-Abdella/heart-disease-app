import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Heart Disease Risk Screening", page_icon="🫀")


@st.cache_resource
def load_model():
    return joblib.load("heart_model.joblib")


bundle = load_model()
model = bundle["model"]
threshold = bundle["threshold"]

st.title("Heart Disease Risk Screening")
st.caption(
    "Educational project trained on the UCI Heart Disease (Cleveland) data. "
    "This is not a medical diagnosis tool."
)

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (years)", 20, 90, 55)
    sex_label = st.selectbox("Sex", ["Female", "Male"])
    cp_label = st.selectbox(
        "Chest pain type",
        ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"],
    )
    trestbps = st.number_input("Resting blood pressure (mm Hg)", 80, 220, 130)
    chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 240)
    fbs_label = st.selectbox("Fasting blood sugar > 120 mg/dl", ["No", "Yes"])
    restecg_label = st.selectbox(
        "Resting ECG",
        ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"],
    )

with col2:
    thalach = st.number_input("Maximum heart rate achieved", 60, 220, 150)
    exang_label = st.selectbox("Exercise-induced angina", ["No", "Yes"])
    oldpeak = st.number_input("ST depression (oldpeak)", 0.0, 7.0, 1.0, step=0.1)
    slope_label = st.selectbox(
        "Slope of peak exercise ST segment", ["Upsloping", "Flat", "Downsloping"]
    )
    ca = st.selectbox("Major vessels colored by fluoroscopy (0-3)", [0, 1, 2, 3])
    thal_label = st.selectbox(
        "Thalassemia", ["Normal", "Fixed defect", "Reversible defect"]
    )

# Map the readable labels back to the numeric codes used in the dataset
sex = {"Female": 0, "Male": 1}[sex_label]
cp = {"Typical angina": 1, "Atypical angina": 2,
      "Non-anginal pain": 3, "Asymptomatic": 4}[cp_label]
fbs = {"No": 0, "Yes": 1}[fbs_label]
restecg = {"Normal": 0, "ST-T wave abnormality": 1,
           "Left ventricular hypertrophy": 2}[restecg_label]
exang = {"No": 0, "Yes": 1}[exang_label]
slope = {"Upsloping": 1, "Flat": 2, "Downsloping": 3}[slope_label]
thal = {"Normal": 3, "Fixed defect": 6, "Reversible defect": 7}[thal_label]

if st.button("Predict", type="primary"):
    row = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
        # same engineered feature used during training
        "max_hr_pct": thalach / (220 - age),
    }])

    prob = model.predict_proba(row)[0, 1]
    st.metric("Estimated probability of heart disease", f"{prob:.1%}")

    if prob >= threshold:
        st.error(f"Higher risk (probability is above the {threshold:.2f} cutoff). "
                 "Further medical evaluation would be advised.")
    else:
        st.success(f"Lower risk (probability is below the {threshold:.2f} cutoff).")

    st.caption(
        "The cutoff is set low on purpose to catch as many true cases as possible, "
        "so some healthy people will be flagged."
    )
