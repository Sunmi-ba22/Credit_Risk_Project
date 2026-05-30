import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="",
    layout="wide"
)

@st.cache_resource
def load_artifacts():
    model    = joblib.load("gradient_boosting_model")
    scaler   = joblib.load("scaler")
    features = joblib.load("feature_columns")
    return model, scaler, features

model, scaler, feature_columns = load_artifacts()
THRESHOLD = 0.35

def make_gauge(probability):
    pct = probability * 100
    color = "#2ecc71" if pct < 35 else ("#f39c12" if pct < 60 else "#e74c3c")
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number+delta",
        value = pct,
        delta = {"reference": 35, "suffix": "%",
                 "increasing": {"color": "#e74c3c"},
                 "decreasing": {"color": "#2ecc71"}},
        number = {"suffix": "%", "font": {"size": 36}},
        title  = {"text": "Default Probability", "font": {"size": 18}},
        gauge  = {
            "axis":      {"range": [0, 100], "tickwidth": 1, "tickcolor": "darkblue"},
            "bar":       {"color": color, "thickness": 0.25},
            "bgcolor":   "white",
            "borderwidth": 2,
            "bordercolor": "gray",
            "steps": [
                {"range": [0,  35], "color": "#d5f5e3"},
                {"range": [35, 60], "color": "#fdebd0"},
                {"range": [60,100], "color": "#fadbd8"},
            ],
            "threshold": {"line": {"color": "red", "width": 4},
                          "thickness": 0.8, "value": 35},
        },
    ))
    fig.update_layout(height=280, margin=dict(t=40, b=10, l=20, r=20),
                      paper_bgcolor="rgba(0,0,0,0)",
                      font={"color": "#2c3e50", "family": "Arial"})
    return fig

st.markdown(
    "<div style='background:linear-gradient(135deg,#1a237e,#283593);"
    "padding:20px 30px;border-radius:12px;margin-bottom:24px;'>"
    "<h1 style='color:white;margin:0;font-size:2rem;'> Credit Risk Predictor</h1>"
    "<p style='color:#90caf9;margin:6px 0 0;font-size:1rem;'>"
    "Gradient Boosting &middot; Threshold = 0.35 &middot; AUC-ROC = 0.945"
    "</p></div>",
    unsafe_allow_html=True
)

st.sidebar.header(" Borrower Details")
st.sidebar.markdown("Enter the loan application details below.")

with st.sidebar:
    age              = st.number_input("Age",                    18, 80, 35)
    income           = st.number_input("Annual Income (NGN)",    10_000, 10_000_000, 500_000, step=10_000)
    loan_amount      = st.number_input("Loan Amount (NGN)",      10_000, 10_000_000, 300_000, step=10_000)
    loan_term        = st.selectbox("Loan Term (months)",        [12, 24, 36, 48, 60])
    interest_rate    = st.slider("Interest Rate (%)",            1.0, 30.0, 12.0, 0.5)
    employment_years = st.number_input("Years Employed",         0, 40, 5)
    num_credit_lines = st.number_input("Number of Credit Lines", 0, 20, 3)
    credit_score     = st.number_input("Credit Score",           300, 850, 650)
    existing_debt    = st.number_input("Existing Debt (NGN)",    0, 5_000_000, 100_000, step=10_000)
    num_dependents   = st.number_input("Number of Dependants",   0, 10, 1)
    predict_btn      = st.button("🔍 Assess Credit Risk",
                                 use_container_width=True, type="primary")

def build_features():
    lti  = loan_amount / income if income > 0 else 0
    dti  = existing_debt / income if income > 0 else 0
    emi  = (loan_amount * (interest_rate/100/12) *
            (1 + interest_rate/100/12)**loan_term /
            ((1 + interest_rate/100/12)**loan_term - 1)) if interest_rate > 0 else loan_amount / loan_term
    emir = emi / income if income > 0 else 0

    raw = {
        "Age": age, "Annual_Income": income, "Loan_Amount": loan_amount,
        "Loan_Term_Months": loan_term, "Interest_Rate": interest_rate,
        "Employment_Years": employment_years, "Num_Credit_Lines": num_credit_lines,
        "Credit_Score": credit_score, "Existing_Debt": existing_debt,
        "Num_Dependents": num_dependents,
        "Loan_To_Income_Ratio": round(lti, 4),
        "Debt_To_Income_Ratio": round(dti, 4),
        "Monthly_EMI":          round(emi, 2),
        "EMI_To_Income_Ratio":  round(emir, 4),
    }
    df_input = pd.DataFrame([raw])
    for col in feature_columns:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[feature_columns]
    return df_input, raw

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader(" Risk Assessment")
    if predict_btn:
        df_input, raw_features = build_features()
        df_scaled   = scaler.transform(df_input)
        probability = model.predict_proba(df_scaled)[0, 1]
        prediction  = int(probability >= THRESHOLD)

        st.plotly_chart(make_gauge(probability), use_container_width=True)

        if prediction == 1:
            st.error(
                f"###  HIGH DEFAULT RISK\n"
                f"**Default Probability: {probability*100:.1f}%**  \n"
                f"This application exceeds the risk threshold of 35%.  \n"
                f"Recommend: Manual review or decline."
            )
        else:
            st.success(
                f"###  LOW DEFAULT RISK\n"
                f"**Default Probability: {probability*100:.1f}%**  \n"
                f"This application is below the risk threshold of 35%.  \n"
                f"Recommend: Proceed with standard due diligence."
            )

        st.markdown("#### Computed Risk Ratios")
        m1, m2, m3 = st.columns(3)
        lti  = raw_features["Loan_To_Income_Ratio"]
        dti  = raw_features["Debt_To_Income_Ratio"]
        emir = raw_features["EMI_To_Income_Ratio"]
        m1.metric("Loan-to-Income", f"{lti:.2f}",
                  delta="High" if lti > 5 else "Normal", delta_color="inverse")
        m2.metric("Debt-to-Income", f"{dti:.2f}",
                  delta="High" if dti > 0.4 else "Normal", delta_color="inverse")
        m3.metric("EMI-to-Income",  f"{emir:.2%}",
                  delta="High" if emir > 0.3 else "Normal", delta_color="inverse")
    else:
        st.info(" Fill in the borrower details on the left and click **Assess Credit Risk**.")

with col_right:
    st.subheader(" Model Information")
    st.markdown(
        "| Item | Detail |\n"
        "|------|--------|\n"
        "| **Algorithm** | Gradient Boosting Classifier |\n"
        "| **AUC-ROC** | 0.945 |\n"
        "| **F1-Score (optimised)** | Best at threshold 0.35 |\n"
        "| **Precision** | ~0.972 |\n"
        "| **Threshold** | 0.35 (optimised from 0.50) |\n"
        "| **SMOTE** | Applied to training set only |\n"
        "| **Scaler** | StandardScaler (fit on train) |\n\n"
        "####  How the Score is Calculated\n"
        "1. Raw inputs are combined with derived ratios  \n"
        "   (Loan-to-Income, Debt-to-Income, EMI-to-Income).\n"
        "2. All features are standardised using the saved `StandardScaler`.\n"
        "3. The Gradient Boosting model outputs a **probability of default**.\n"
        "4. If probability >= **0.35** → classified as **High Risk**.\n\n"
        "####  Threshold Rationale\n"
        "The default threshold of 0.50 was lowered to **0.35** because  \n"
        "missing a true defaulter (false negative) is costlier than  \n"
        "flagging a safe borrower for review (false positive).  \n"
        "This threshold maximises the F1-score across the test set.\n"
    )

    st.markdown("####  Artifacts Used")
    st.code(
        "gradient_boosting_model  <- trained model\n"
        "scaler                   <- StandardScaler\n"
        "feature_column          <- column order\n"
        "app.py                       <- this application",
        language="bash"
    )

st.markdown("---")
st.caption(
    "Credit Risk Predictor · Built with Streamlit & scikit-learn · "
    "Model: GradientBoostingClassifier · Data: Credit_Risk_Features.csv"
)
