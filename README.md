# Customer Churn Prediction
A Customer Churn Predictor that determines whether a customer is likely to churn or stay based on historical data. The model not only predicts but also provides explainability using SHAP (SHapley Additive exPlations) for interpretability and business insight. The model applies multiple Machine Learning algorithms to find out the best performing model for churn predictions. 

## Dataset
The dataset that has been used for training this model is the **Source [Telco_Customer_Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)**, borrowed from Kaggle.

## Preprocessing
1. Dropped irrelevant columns to reduce noise.
2. Encoded the categorical features using OneHotEncoder.
3. Scaled the numerical features via ColumnTransform and Pipeline methods.
4. Addressed class imbalance using SMOTE (Synthetic Minority Oversampling Technique).

## Model
1. **Algorithms tested**: Logistic Regression, KNN, SVM, Random Forest, and XGBoost.
2. XGBoost achieved the highest mean accuracy.
3. Optimized the accuracy by hyperparameter tuning via Random Search CV.
4. **Explainability**: Integrated with SHAP for interpretability and visualization of feature importance.
5. **Interface**: Integrated with Streamlit for User Interface.
**Final accuracy: 80.1%**

## XAI Integration
XAI integration helps stakeholders understand why a customer is predicted to churn or not likely to churn. It visualizes the feature importance in a prediction. 

## Installation
``` bash
git clone https://github.com/JambavanAbhiram/Customer-Churn-Prediction-with-XAI.git
cd smart-ticket-prioritization
pip install -r requirements.txt
streamlit run app.py
