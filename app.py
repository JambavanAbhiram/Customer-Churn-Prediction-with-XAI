import pickle
import warnings
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import streamlit as st
import sklearn

@st.cache_resource
def load_model(_sk_version: str = sklearn.__version__) -> Any:
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

def get_feature_config(model_pipeline: Any) -> Tuple[List[str], List[str], Dict[str, List[Any]]]:
    preprocessor = model_pipeline.named_steps.get("preprocessor")
    if preprocessor is None:
        raise ValueError("Saved model does not contain a 'preprocessor' step.")

    # transformers_: List of tuples (name, transformer, columns)
    transformers = preprocessor.transformers_

    numerical_columns: List[str] = []
    categorical_columns: List[str] = []
    for name, _transformer, cols in transformers:
        if name == "num":
            numerical_columns = list(cols)
        elif name == "cat":
            categorical_columns = list(cols)

    # Extract fitted categories from the OneHotEncoder for each categorical feature
    ohe = preprocessor.named_transformers_.get("cat")
    if hasattr(ohe, "categories_"):
        categories = {feature: list(cats) for feature, cats in zip(categorical_columns, ohe.categories_)}
    else:
        categories = {feature: [] for feature in categorical_columns}

    return numerical_columns, categorical_columns, categories


def coerce_numeric(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def render_sidebar_info():
    st.sidebar.title("Customer Churn Prediction")
    st.sidebar.markdown(
        "Predict whether a telco customer will churn using a trained XGBoost model."
    )
    st.sidebar.markdown(
        "Model is a scikit-learn Pipeline with preprocessing (scaling + one-hot) and XGBClassifier."
    )


def main():
    st.set_page_config(page_title="Churn Prediction", page_icon="📊", layout="centered")
    render_sidebar_info()

    # Silence version-related warnings in UI
    warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning:.*")
    warnings.filterwarnings("ignore", message=".*serialized model.*")

    # Simple two-step flow using session state: 'form' -> 'result'
    if "step" not in st.session_state:
        st.session_state.step = "form"
    if "last_input_df" not in st.session_state:
        st.session_state.last_input_df = None
    if "last_proba" not in st.session_state:
        st.session_state.last_proba = None
    if "last_pred" not in st.session_state:
        st.session_state.last_pred = None

    # Hard-require the same sklearn version used to create the pickle
    required_sklearn = "1.7.0"
    current_sklearn = sklearn.__version__
    if current_sklearn != required_sklearn:
        st.error(
            f"scikit-learn runtime mismatch: required {required_sklearn}, found {current_sklearn}.\n"
            "Activate your virtual environment and reinstall dependencies."
        )
        st.info(
            "Tip: Run `.\\.venv\\Scripts\\python -m streamlit run streamlit_app.py` to ensure the venv Python is used."
        )

    model_pipeline = load_model()
    try:
        numerical_columns, categorical_columns, categories = get_feature_config(model_pipeline)
    except Exception as e:
        st.error(f"Failed to read feature configuration from model: {e}")
        return

    preferred_order = [
        "gender",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "InternetService",
        "Contract",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
    ]

    def render_form():
        st.title("Customer Churn Prediction")
        st.caption("Enter customer details to predict if they are likely to churn.")

        with st.form("churn_form"):
            col_left, col_right = st.columns(2)
            input_values: Dict[str, Any] = {}

            # Left column controls
            with col_left:
                # gender (categorical)
                if "gender" in categorical_columns:
                    opts = categories.get("gender", ["Male", "Female"]) 
                    input_values["gender"] = st.selectbox("Gender", [str(o) for o in opts], index=0)

                # Dependents
                if "Dependents" in categorical_columns:
                    opts = categories.get("Dependents", ["Yes", "No"]) 
                    input_values["Dependents"] = st.selectbox("Dependents", [str(o) for o in opts], index=0)

                # PhoneService
                if "PhoneService" in categorical_columns:
                    opts = categories.get("PhoneService", ["Yes", "No"]) 
                    input_values["PhoneService"] = st.selectbox("Phone Service", [str(o) for o in opts], index=0)

                # Contract
                if "Contract" in categorical_columns:
                    opts = categories.get("Contract", ["Month-to-month", "One year", "Two year"]) 
                    input_values["Contract"] = st.selectbox("Contract Type", [str(o) for o in opts], index=0)

                # MonthlyCharges
                if "MonthlyCharges" in numerical_columns:
                    input_values["MonthlyCharges"] = st.number_input("Monthly Charges", min_value=0.0, step=0.01, value=50.0, format="%f")

            # Right column controls
            with col_right:
                # Partner
                if "Partner" in categorical_columns:
                    opts = categories.get("Partner", ["Yes", "No"]) 
                    input_values["Partner"] = st.selectbox("Partner", [str(o) for o in opts], index=0)

                # tenure
                if "tenure" in numerical_columns:
                    input_values["tenure"] = st.number_input("Tenure (Months)", min_value=0, max_value=72, step=1, value=24)

                # InternetService
                if "InternetService" in categorical_columns:
                    opts = categories.get("InternetService", ["DSL", "Fiber optic", "No"]) 
                    input_values["InternetService"] = st.selectbox("Internet Service", [str(o) for o in opts], index=0)

                # PaymentMethod
                if "PaymentMethod" in categorical_columns:
                    opts = categories.get("PaymentMethod", [
                        "Electronic check",
                        "Mailed check",
                        "Bank transfer (automatic)",
                        "Credit card (automatic)",
                    ]) 
                    input_values["PaymentMethod"] = st.selectbox("Payment Method", [str(o) for o in opts], index=0)

                # TotalCharges
                if "TotalCharges" in numerical_columns:
                    input_values["TotalCharges"] = st.number_input("Total Charges", min_value=0.0, step=0.01, value=1200.0, format="%f")

            # Any remaining features not in the preferred UI are rendered in an expander
            remaining_features: List[str] = [
                f for f in (numerical_columns + categorical_columns) if f not in preferred_order
            ]
            if remaining_features:
                with st.expander("Additional features"):
                    for feature in remaining_features:
                        if feature in numerical_columns:
                            input_values[feature] = st.number_input(feature, value=0.0, step=0.1, format="%f")
                        else:
                            opts = [str(o) for o in categories.get(feature, [])] or ["Unknown"]
                            input_values[feature] = st.selectbox(feature, opts, index=0)

            submitted = st.form_submit_button("Predict Churn")

        if not submitted:
            return

        # Build single-row DataFrame with all expected features
        all_features: List[str] = numerical_columns + [c for c in categorical_columns if c not in numerical_columns]
        input_row: Dict[str, Any] = {}
        for feature in all_features:
            val = input_values.get(feature)
            if feature in numerical_columns:
                input_row[feature] = coerce_numeric(val)
            else:
                input_row[feature] = val

        input_df = pd.DataFrame([input_row])
        try:
            prediction = model_pipeline.predict(input_df)
            proba = None
            if hasattr(model_pipeline, "predict_proba"):
                p = model_pipeline.predict_proba(input_df)
                if isinstance(p, np.ndarray) and p.ndim == 2 and p.shape[1] >= 2:
                    proba = float(p[0, 1])
            st.session_state.last_input_df = input_df
            st.session_state.last_proba = proba
            st.session_state.last_pred = int(prediction[0]) if hasattr(prediction, "__iter__") else int(prediction)
            st.session_state.step = "result"
        except Exception as e:
            st.error(f"Prediction failed: {e}")

        try:
            # Preprocess the input for SHAP (numeric array)
            X_preprocessed = model_pipeline.named_steps["preprocessor"].transform(input_df)
            feature_names = model_pipeline.named_steps["preprocessor"].get_feature_names_out()
            X_preprocessed_df = pd.DataFrame(X_preprocessed, columns=feature_names)

            # Initialize TreeExplainer for XGBClassifier
            import shap
            explainer = shap.TreeExplainer(model_pipeline.named_steps["classifier"])
            shap_values = explainer(X_preprocessed_df)

            # Save to session state for display in render_results()
            st.session_state.shap_explainer = explainer
            st.session_state.shap_values = shap_values
            st.session_state.shap_input = X_preprocessed_df
        except Exception as e:
            st.warning(f"SHAP explanation could not be generated: {e}")

    def render_results():
        st.title("Prediction Results")
        pred_label = "Likely to Churn" if st.session_state.last_pred == 1 else "Not Likely to Churn"
        proba_text = f"{st.session_state.last_proba:.2%}" if st.session_state.last_proba is not None else "N/A"

        # Styled block for prediction
        st.markdown(
            f"""
            <div style='background:#ffffff;border-radius:16px;box-shadow:0 10px 25px rgba(0,0,0,0.1);padding:24px;'>
                <h2 style='text-align:center;margin:0 0 8px 0;font-size:32px;color:#1f2937;'>Prediction Results</h2>
                <p style='text-align:center;color:#374151;margin:8px 0 0 0;'>Customer Churn Status:</p>
                <p style='text-align:center;font-size:40px;font-weight:800;margin:8px 0;color:{'#dc2626' if st.session_state.last_pred == 1 else '#16a34a'}'>
                    {pred_label}
                </p>
                <p style='text-align:center;color:#6b7280;font-size:20px'>(Churn Probability: <span style='font-weight:600;color:#374151'>{proba_text}</span>)</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if hasattr(st.session_state, "shap_values"):
            st.subheader("SHAP Explanation")
            fig = plt.gcf()
            shap.plots.bar(st.session_state.shap_values[0], show=False)
            st.pyplot(fig)

        with st.expander("View model input row"):
            st.dataframe(st.session_state.last_input_df)

        if st.button("Predict Another Customer"):
            st.session_state.step = "form"
            st.session_state.last_input_df = None
            st.session_state.last_pred = None
            st.session_state.last_proba = None


    if st.session_state.step == "form":
        render_form()
    else:
        render_results()

if __name__ == "__main__":
    main()


