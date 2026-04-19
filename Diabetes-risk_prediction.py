#LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

#LOAD DATASET
df=pd.read_csv(r"C:\Users\91708\Documents\diabetes_prediction_dataset.csv")

print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

#Fixing invalid zero values
columns_to_fix = ['bmi', 'HbA1c_level', 'blood_glucose_level']

for col in columns_to_fix:
    if col in df.columns:
        df[col] = df[col].replace(0, df[col].mean())

#Handling Categorical Data
df = pd.get_dummies(df, columns=['smoking_history', 'gender'], drop_first=True)

#Feature & Target Split
X = df.drop("diabetes", axis=1)
y = df["diabetes"]

#SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Normalization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Model Training (Classification Model)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#Prediction
y_pred = model.predict(X_test)

#Evaluation
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

#Feature Importance
importances = model.feature_importances_
features = X.columns

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:\n", importance_df)


plt.figure()
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.title("Feature Importance (Importance Score, NOT Feature Values)")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.subplots_adjust(left=0.3)
plt.show()

