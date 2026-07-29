# 🚢 Titanic Disaster: Statistical EDA & Streamlit Analytics Dashboard

[![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)

A complete, production-grade Data Science & Analytics project featuring a rigorous **§3.1 Exploratory Data Analysis (EDA) Jupyter Notebook** and an **Interactive Streamlit Dashboard** converting analytical findings into executive KPIs, interactive filters, Plotly visualizations, and business insight callout cards.

---

## 📊 Dashboard Preview & Layout Structure

```
+-----------------------------------------------------------------------------------+
|  🚢 Titanic Disaster: Statistical EDA & Insights Dashboard                        |
|  Interactive analytics converting §3.1 EDA into executive metrics & insights.     |
+-----------------------------------------------------------------------------------+
|  📌 Key Performance Indicators (KPIs)                                            |
|  [ Survival Rate: 38.4% ]  [ Passengers: 891 ]  [ Avg Fare: £32.20 ] [ Cabin: 23% ] |
+-----------------------------------------------------------------------------------+
|  📈 Interactive Plotly Visualizations & Insight Callout Cards                     |
|                                                                                   |
|  [ Chart 1: Survival Rate by Gender & Class ]  [ Chart 2: Fare vs Survival (Log) ] |
|  💡 Insight 1: 74.2% female vs 18.9% male.      💡 Insight 2: £48.40 vs £22.12 fare|
|                                                                                   |
|  [ Chart 3: Age Distribution & Vulnerability ] [ Chart 4: Survival by Embarked ]  |
|  💡 Insight 3: Child survival spike (<10 yrs).  💡 Insight 4: Cherbourg rate 55.4% |
+-----------------------------------------------------------------------------------+
```

---

## 🌟 Key Project Features & Deliverables

### 1. Complete §3.1 EDA Workflow (`eda_workflow.ipynb` & `titanic_statistical_eda.ipynb`)
- **Data Quality & Missing Value Strategy**: Imputed missing `Age` using **Median** ($28.0$ years) to prevent outlier distortion; imputed `Embarked` using **Mode** (`'S'`); transformed `Cabin` into a binary indicator `Has_Cabin`.
- **Univariate Analysis & Skewness**: Quantified right-skewness in `Fare` ($\text{Skewness} = 4.79$) and recommended logarithmic transformation $\log(1 + \text{Fare})$.
- **Outlier Detection (IQR Rule)**: Calculated $Q1, Q3, \text{IQR}$ bounds and identified 116 legitimate luxury fare outliers (£66.34 to £512.33).
- **Formal Hypothesis Testing**: Conducted a Welch's Two-Sample t-test ($t = 6.8391, p = 2.699 \times 10^{-11} < 0.05$), proving survivors paid a statistically significantly higher average fare (£48.40 vs £22.12).

### 2. Interactive Streamlit Dashboard (`app.py`)
- **1 Executive KPI Row (`st.metric`)**: Overall Survival Rate %, Filtered Passengers Count, Average Ticket Fare £, and Cabin Ownership Rate %.
- **4 Multi-Dimensional Interactive Filters**: Passenger Class (`Pclass`), Gender (`Sex`), Age Range Slider (0–80 years), and Embarkation Port (`Embarked`).
- **4 Interactive Plotly Charts**: Custom color palettes, hover templates, log scales, and grouped bar/box plots.
- **1 Insight Text Block Per Chart**: Highlights key business implications right alongside each visual.

---

## 🛠️ Tech Stack & Dependencies

- **Language**: Python 3.11
- **Dashboard Framework**: Streamlit
- **Visualization**: Plotly Express, Plotly Graph Objects, Seaborn, Matplotlib
- **Data Processing & Analytics**: Pandas, NumPy, SciPy (Stats)
- **Notebook Environment**: Jupyter / NbFormat

---

## 🚀 How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/titanic-eda-streamlit-dashboard.git
cd titanic-eda-streamlit-dashboard
```

### 2. Set Up Virtual Environment & Install Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
The dashboard will open automatically in your browser at `http://localhost:8501`.

### 4. Run the Jupyter Notebook
```bash
jupyter notebook eda_workflow.ipynb
```

---

## 📤 How to Push to GitHub

1. Initialize Git and commit files locally:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Titanic §3.1 EDA workflow & Streamlit Dashboard"
   ```
2. Create a new repository on GitHub (e.g. `titanic-eda-streamlit-dashboard`).
3. Connect remote and push:
   ```bash
   git remote add origin https://github.com/<your-username>/titanic-eda-streamlit-dashboard.git
   git branch -M main
   git push -u origin main
   ```

---

## 📝 License
This project is open-source under the MIT License.
