import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

@st.cache_resource
def load_or_train_model():
    if os.path.exists('model.pkl'):
        model = pickle.load(open('model.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        imputer = pickle.load(open('imputer.pkl', 'rb'))
        feature_columns = pickle.load(open('feature_columns.pkl', 'rb'))
    else:
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
        scaler.fit(X_imputed)
        X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        feature_columns = list(X.columns)
    return model, scaler, imputer, feature_columns

model, scaler, imputer, feature_columns = load_or_train_model()

st.set_page_config(page_title="India AQI Predictor", page_icon="🌫️", layout="centered")

st.markdown("""
<style>
.metric-row { display: flex; gap: 12px; margin-bottom: 1.5rem; }
.metric-box { flex: 1; background: #f5f5f5; border-radius: 10px; padding: 14px 16px; }
.metric-label { font-size: 12px; color: #888; margin-bottom: 4px; }
.metric-val { font-size: 22px; font-weight: 600; color: #111; }
.result-card { border-radius: 12px; padding: 1.5rem; border-left: 6px solid; margin: 1rem 0; }
.result-aqi { font-size: 40px; font-weight: 700; margin-bottom: 4px; }
.result-cat { font-size: 16px; font-weight: 600; }
.result-desc { font-size: 13px; margin-top: 8px; color: #555; }
.insight-row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 1rem; }
.insight-box { background: #f5f5f5; border-radius: 10px; padding: 14px; }
.insight-title { font-size: 12px; color: #888; margin-bottom: 6px; font-weight: 600; }
.insight-body { font-size: 13px; color: #333; line-height: 1.5; }
.scale-row { display: flex; gap: 4px; margin: 1rem 0; }
.scale-item { flex: 1; border-radius: 6px; padding: 6px 4px; text-align: center; }
.scale-range { font-size: 10px; font-weight: 600; }
.scale-name { font-size: 9px; margin-top: 2px; }
div[data-testid="stSidebar"] { display: none; }
</style>
""", unsafe_allow_html=True)

st.markdown("## 🌫️ India Air Quality Index Predictor")
st.markdown("Predict AQI from pollutant levels across 26 Indian cities · Random Forest · CPCB 2015–2020")
st.markdown("---")

st.markdown("""
<div class="metric-row">
  <div class="metric-box"><div class="metric-label">Model accuracy</div><div class="metric-val">91.2%</div></div>
  <div class="metric-box"><div class="metric-label">Avg error (MAE)</div><div class="metric-val">±20 AQI</div></div>
  <div class="metric-box"><div class="metric-label">Training rows</div><div class="metric-val">19,880</div></div>
</div>
""", unsafe_allow_html=True)

cities = [
    'Ahmedabad', 'Aizawl', 'Amaravati', 'Amritsar', 'Bengaluru',
    'Bhopal', 'Brajrajnagar', 'Chandigarh', 'Chennai', 'Coimbatore',
    'Delhi', 'Ernakulam', 'Gurugram', 'Guwahati', 'Hyderabad',
    'Jaipur', 'Jorapokhar', 'Kochi', 'Kolkata', 'Lucknow',
    'Mumbai', 'Patna', 'Shillong', 'Talcher', 'Thiruvananthapuram',
    'Visakhapatnam'
]

season_presets = {
    "Winter (Oct–Jan) ❄️": {"pm25": 250.0, "pm10": 300.0, "no": 15.0, "no2": 60.0, "nox": 80.0, "nh3": 25.0, "co": 2.5, "so2": 20.0, "o3": 20.0},
    "Summer (Apr–Jun) ☀️": {"pm25": 60.0,  "pm10": 90.0,  "no": 8.0,  "no2": 30.0, "nox": 40.0, "nh3": 12.0, "co": 1.0, "so2": 10.0, "o3": 55.0},
    "Monsoon (Jul–Sep) 🌧️": {"pm25": 35.0,  "pm10": 55.0,  "no": 5.0,  "no2": 20.0, "nox": 28.0, "nh3": 10.0, "co": 0.8, "so2": 8.0,  "o3": 25.0},
    "Custom": None
}

col1, col2 = st.columns(2)
with col1:
    city = st.selectbox("🏙️ Select City", cities, index=cities.index('Delhi'))
with col2:
    season = st.selectbox("📅 Load Season Preset", list(season_presets.keys()))

preset = season_presets[season]

st.markdown("### 🧪 Pollutant Levels")
st.caption("Drag sliders to adjust — defaults load from season preset")

col1, col2, col3 = st.columns(3)

default_pm25 = preset["pm25"] if preset else 60.0
default_pm10 = preset["pm10"] if preset else 100.0
default_no   = preset["no"]   if preset else 10.0
default_no2  = preset["no2"]  if preset else 40.0
default_nox  = preset["nox"]  if preset else 50.0
default_nh3  = preset["nh3"]  if preset else 15.0
default_co   = preset["co"]   if preset else 1.5
default_so2  = preset["so2"]  if preset else 15.0
default_o3   = preset["o3"]   if preset else 30.0

with col1:
    pm25 = st.slider("PM2.5 (µg/m³)", 0.0, 500.0, default_pm25, step=1.0)
    no   = st.slider("NO (µg/m³)",    0.0, 200.0, default_no,   step=1.0)
    co   = st.slider("CO (mg/m³)",    0.0, 20.0,  default_co,   step=0.1)

with col2:
    pm10 = st.slider("PM10 (µg/m³)", 0.0, 600.0, default_pm10, step=1.0)
    no2  = st.slider("NO2 (µg/m³)",  0.0, 300.0, default_no2,  step=1.0)
    so2  = st.slider("SO2 (µg/m³)",  0.0, 300.0, default_so2,  step=1.0)

with col3:
    nox  = st.slider("NOx (µg/m³)",  0.0, 400.0, default_nox,  step=1.0)
    nh3  = st.slider("NH3 (µg/m³)",  0.0, 200.0, default_nh3,  step=1.0)
    o3   = st.slider("O3 (µg/m³)",   0.0, 300.0, default_o3,   step=1.0)

st.markdown("---")

if st.button("🔍 Predict AQI", use_container_width=True):

    input_dict = {col: 0 for col in feature_columns}
    input_dict['PM2.5'] = pm25
    input_dict['PM10']  = pm10
    input_dict['NO']    = no
    input_dict['NO2']   = no2
    input_dict['NOx']   = nox
    input_dict['NH3']   = nh3
    input_dict['CO']    = co
    input_dict['SO2']   = so2
    input_dict['O3']    = o3

    city_col = f'City_{city}'
    if city_col in input_dict:
        input_dict[city_col] = 1

    input_df      = pd.DataFrame([input_dict])
    input_imputed = imputer.transform(input_df)
    prediction    = round(float(model.predict(input_imputed)[0]), 1)

    if prediction <= 50:
        color, cat, advice = "#00a000", "Good ✅", "Air is safe. Great day for outdoor activities."
    elif prediction <= 100:
        color, cat, advice = "#5a8030", "Satisfactory 🟡", "Acceptable air quality. Sensitive individuals should limit prolonged exertion."
    elif prediction <= 200:
        color, cat, advice = "#888800", "Moderate 🟠", "Sensitive groups may experience discomfort. Limit outdoor time if needed."
    elif prediction <= 300:
        color, cat, advice = "#cc6400", "Poor 🔴", "Everyone may experience health effects. Limit prolonged outdoor exertion."
    elif prediction <= 400:
        color, cat, advice = "#cc0000", "Very Poor 🟣", "Wear N95 masks outdoors. Run air purifiers indoors. Avoid exercise outside."
    else:
        color, cat, advice = "#7e0023", "Severe ⚫", "Health emergency. Stay indoors. Avoid all outdoor exposure. Use air purification."

    top_pollutant = "PM2.5" if pm25 > pm10 * 0.6 else "PM10"
    top_val       = pm25 if top_pollutant == "PM2.5" else pm10
    safe_limit    = 60 if top_pollutant == "PM2.5" else 100

    st.markdown(f"""
    <div class="result-card" style="background:{color}11; border-color:{color};">
        <div class="result-aqi" style="color:{color};">{prediction}</div>
        <div class="result-cat" style="color:{color};">{cat}</div>
        <div class="result-desc">{advice}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="scale-row">
      <div class="scale-item" style="background:#00e40022; border:0.5px solid #00e400;">
        <div class="scale-range" style="color:#00a000;">0–50</div>
        <div class="scale-name" style="color:#00a000;">Good</div>
      </div>
      <div class="scale-item" style="background:#92d05022; border:0.5px solid #92d050;">
        <div class="scale-range" style="color:#5a8030;">51–100</div>
        <div class="scale-name" style="color:#5a8030;">Satisfactory</div>
      </div>
      <div class="scale-item" style="background:#ffff0033; border:0.5px solid #cccc00;">
        <div class="scale-range" style="color:#888800;">101–200</div>
        <div class="scale-name" style="color:#888800;">Moderate</div>
      </div>
      <div class="scale-item" style="background:#ff7e0022; border:0.5px solid #ff7e00;">
        <div class="scale-range" style="color:#cc6400;">201–300</div>
        <div class="scale-name" style="color:#cc6400;">Poor</div>
      </div>
      <div class="scale-item" style="background:#ff000022; border:0.5px solid #ff0000;">
        <div class="scale-range" style="color:#cc0000;">301–400</div>
        <div class="scale-name" style="color:#cc0000;">Very Poor</div>
      </div>
      <div class="scale-item" style="background:#7e002322; border:0.5px solid #7e0023;">
        <div class="scale-range" style="color:#7e0023;">400+</div>
        <div class="scale-name" style="color:#7e0023;">Severe</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="insight-row">
      <div class="insight-box">
        <div class="insight-title">🔥 Top contributor</div>
        <div class="insight-body">{top_pollutant} ({top_val} µg/m³) is the primary driver.
        Safe limit is {safe_limit} µg/m³ per CPCB norms.</div>
      </div>
      <div class="insight-box">
        <div class="insight-title">💡 Health advisory</div>
        <div class="insight-body">{advice}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📊 Input Summary")
    summary = pd.DataFrame({
        'Pollutant': ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3'],
        'Value':     [pm25, pm10, no, no2, nox, nh3, co, so2, o3],
        'Unit':      ['µg/m³','µg/m³','µg/m³','µg/m³','µg/m³','µg/m³','mg/m³','µg/m³','µg/m³']
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("Model: Random Forest · R² = 0.91 · MAE = 20.54 · Trained on CPCB India AQI Dataset (2015–2020) · 26 cities · 19,880 rows")

st.markdown("---")
st.markdown("""
<div style="text-align:center; padding: 1.2rem 0 0.5rem 0;">
    <p style="font-size:16px; font-weight:600; margin-bottom:4px; color:inherit;">Amit Singh</p>
    <p style="font-size:13px; color:gray; margin-bottom:6px;">B.Tech CSE · NSUT Delhi · Batch 2027</p>
    <p style="font-size:13px; color:gray; margin-bottom:10px;">
        📧 <a href="mailto:amit.singh.ug23@gmail.com" style="color:gray;">amit.singh.ug23@gmail.com</a>
    </p>
    <p style="font-size:12px; color:#aaa; margin-bottom:4px;">Open to collaborations in ML · Data Science · AI projects</p>
    <p style="font-size:12px; color:#aaa;">
        🔗 <a href="https://github.com/Amitsingh-nsut2027" target="_blank" style="color:#aaa;">github.com/Amitsingh-nsut2027</a>
        &nbsp;·&nbsp;
        <a href="https://www.linkedin.com/in/amit-singh-nsut" target="_blank" style="color:#aaa;">LinkedIn</a>
    </p>
</div>
""", unsafe_allow_html=True)