# 🌍 India AQI Predictor using Machine Learning

> Predict the Air Quality Index (AQI) of major Indian cities using Machine Learning and receive instant health recommendations.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikitlearn)
![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Project Overview

Air pollution is one of India's biggest public health challenges. Every year, millions of people are exposed to unhealthy air, especially during winter months.

This project uses **Machine Learning** to predict the **Air Quality Index (AQI)** based on pollutant concentrations across **26 Indian cities** using historical air quality data.

The application allows users to:

- Select an Indian city
- Adjust pollutant levels using interactive sliders
- Predict AQI instantly
- View the corresponding health advisory

The project combines **data analysis**, **machine learning**, and **web deployment** into a complete end-to-end application.

---

## 🚀 Live Demo

🌐 **Streamlit App**

https://amitsingh-nsut2027--india-aqi-predictor-app-l6zqh7.streamlit.app

---

## 📊 Dataset

The project is trained on India's historical air quality dataset containing:

- 26 Indian Cities
- Multiple Pollutants
- Daily AQI Records
- 5 Years of Data

### Features Used

- PM2.5
- PM10
- NO
- NO₂
- NOx
- NH₃
- CO
- SO₂
- O₃
- Benzene
- Toluene
- Xylene
- City

Target Variable:

- AQI

---

# 🧠 Machine Learning Models

Several regression algorithms were trained and compared.

| Model | R² Score |
|--------|----------|
| Linear Regression | 0.82 |
| Ridge Regression | Evaluated |
| Lasso Regression | Evaluated |
| ⭐ Random Forest Regressor | **0.91** |

### Why Random Forest?

AQI does not increase linearly with pollutant concentration.

Tree-based models can capture:

- Non-linear relationships
- Feature interactions
- Complex decision boundaries

which makes Random Forest significantly more accurate.

---

# 📈 Project Workflow

```
Data Collection
        │
        ▼
Data Cleaning
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Feature Engineering
        │
        ▼
Model Training
        │
        ▼
Model Evaluation
        │
        ▼
Model Serialization (.pkl)
        │
        ▼
Streamlit Deployment
```

---

# 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn
- Joblib

---

# 📷 Application Preview

> Add screenshots inside an `images/` folder.

```
images/
├── home.png
├── prediction.png
├── dashboard.png
```

Example:

```markdown
![Home](images/home.png)
```

---

# 📂 Project Structure

```
India-AQI-Predictor
│
├── app.py
├── model.pkl
├── dataset.csv
├── requirements.txt
├── README.md
│
├── notebooks/
│      EDA.ipynb
│      Model_Training.ipynb
│
├── images/
│      home.png
│      prediction.png
│
└── utils.py
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/Amitsingh-nsut2027/-india-aqi-predictor.git
```

Move into the project directory

```bash
cd -india-aqi-predictor
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

# 📊 Results

✔ Random Forest achieved the highest prediction accuracy.

**Performance**

- R² Score: **0.91**
- Low prediction error
- Handles non-linear pollution patterns effectively

---

# 💡 Future Improvements

- Real-time AQI prediction using APIs
- Weather integration
- Deep Learning models (LSTM/XGBoost)
- AQI forecasting for next 24–72 hours
- Interactive pollution heatmaps
- Mobile responsive interface
- Docker deployment

---

# 🤝 Contributing

Contributions are always welcome!

If you'd like to improve this project:

1. Fork the repository
2. Create a new branch
3. Commit your changes
4. Open a Pull Request

---

# 👨‍💻 Author

**Amit Singh**

B.Tech Computer Science Engineering (2023–2027)

Netaji Subhas University of Technology (NSUT), Delhi

📧 Email: amit.singh.ug23@gmail.com

🔗 LinkedIn: https://linkedin.com/in/amitsingh-nsut2027

🐙 GitHub: https://github.com/Amitsingh-nsut2027

---

# ⭐ If you found this project useful

Please consider giving it a ⭐ on GitHub.

It motivates me to build more open-source Machine Learning projects.
