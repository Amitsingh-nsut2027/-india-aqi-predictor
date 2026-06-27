import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

print("Training model on city_day.csv...")

df = pd.read_csv('city_day.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.drop(columns=['Xylene'], inplace=True)
df.dropna(subset=['AQI'], inplace=True)

model_df = df.drop(columns=['Date', 'AQI_Bucket', 'Benzene', 'Toluene'])
model_df = pd.get_dummies(model_df, columns=['City'], drop_first=True)

X = model_df.drop(columns=['AQI'])
y = model_df['AQI']

imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

pickle.dump(rf, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
pickle.dump(imputer, open('imputer.pkl', 'wb'))
pickle.dump(list(X.columns), open('feature_columns.pkl', 'wb'))

print("Model trained and saved.")
