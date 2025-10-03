import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv(r'D:\Projects\Customer Churn Prediction XAI\Telco Customer Churn.csv')
df = df.drop(columns=['customerID'])
df["TotalCharges"] = df["TotalCharges"].replace(" ", "0.0").astype(float)
df["Churn"] = df["Churn"].replace({'Yes': 1, 'No': 0})

numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = [col for col in df.columns if col not in numerical_features + ['Churn']]

x = df.drop(columns=['Churn'])
y = df['Churn']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

x_train_processed = preprocessor.fit_transform(x_train)
x_test_processed = preprocessor.transform(x_test)

smote = SMOTE(random_state=42)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train_processed, y_train)

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(x_train_resampled, y_train_resampled)

params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.05],
    'subsample': [0.7, 0.9],
    'colsample_bytree': [0.7, 0.9]
}

random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=params,
    n_iter=4,
    scoring='accuracy',
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)
random_search.fit(x_train_resampled, y_train_resampled)

best_model = random_search.best_estimator_

y_pred = best_model.predict(x_test_processed)
print("Accuracy:", accuracy_score(y_test, y_pred))

full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', best_model)
])

pickle.dump(full_pipeline, open("model.pkl", "wb"))