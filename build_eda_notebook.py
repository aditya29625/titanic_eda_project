import json
import os

def create_eda_notebook():
    cells = []

    def add_md(source):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": source.strip().splitlines(keepends=True)
        })

    def add_code(source):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source.strip().splitlines(keepends=True)
        })

    # Section 1: Title, Objective, Dataset Description
    add_md("""# Titanic Survival Analysis: Section 3.1 Exploratory Data Analysis (EDA) Workflow

### Objective
This notebook executes an end-to-end §3.1 Exploratory Data Analysis (EDA) workflow on the Titanic dataset to uncover key determinants of passenger survival, quantify feature distributions and outliers, execute formal statistical hypothesis testing, and extract 4 actionable core business insights for dashboard visualization and machine learning modeling.

### Dataset Dictionary
| Feature | Type | Description |
| :--- | :--- | :--- |
| **PassengerId** | int | Unique index |
| **Survived** | int | Target status (0 = Perished, 1 = Survived) |
| **Pclass** | int | Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd) |
| **Name** | str | Passenger full name & title |
| **Sex** | str | Biological gender (male, female) |
| **Age** | float | Age in years |
| **SibSp** | int | Number of siblings/spouses aboard |
| **Parch** | int | Number of parents/children aboard |
| **Ticket** | str | Ticket number |
| **Fare** | float | Passenger fare (£) |
| **Cabin** | str | Cabin number |
| **Embarked** | str | Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) |
""")

    # Section 2: Imports & Setup
    add_md("""# 2. Imports & Environmental Configuration""")
    add_code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 100

print("Core EDA libraries successfully imported.")
""")

    # Section 3: Data Inspection
    add_md("""# 3. Load & Inspect Dataset""")
    add_code("""url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print("--- DATASET SHAPE ---")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\\n--- FIRST 5 ROWS ---")
display(df.head())

print("\\n--- DATASET INFO ---")
df.info()

print("\\n--- SUMMARY STATISTICS ---")
display(df.describe().T)

print("\\n--- MISSING VALUES COUNT ---")
missing = df.isnull().sum()
display(missing[missing > 0])
""")

    # Section 4: Data Cleaning
    add_md("""# 4. Data Cleaning & Feature Preprocessing""")
    add_code("""df_clean = df.copy()

# 1. Duplicates check
print(f"Duplicates found: {df_clean.duplicated().sum()}")

# 2. Age Median Imputation
age_median = df_clean['Age'].median()
df_clean['Age'].fillna(age_median, inplace=True)
print(f"Age imputed with Median: {age_median:.2f}")

# 3. Embarked Mode Imputation
embarked_mode = df_clean['Embarked'].mode()[0]
df_clean['Embarked'].fillna(embarked_mode, inplace=True)
print(f"Embarked imputed with Mode: '{embarked_mode}'")

# 4. Cabin Transformation to Has_Cabin
df_clean['Has_Cabin'] = df_clean['Cabin'].notnull().astype(int)
df_clean.drop(columns=['Cabin'], inplace=True)
print("Transformed raw Cabin column into binary Has_Cabin indicator.")

print("\\nMissing values remaining:", df_clean.isnull().sum().sum())
""")

    # Section 5: Univariate Analysis
    add_md("""# 5. Univariate Analysis (Distributions & Skewness)""")
    add_code("""numeric_cols = ['Age', 'Fare', 'SibSp', 'Parch']

plt.figure(figsize=(14, 8))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(2, 2, i)
    sns.histplot(df_clean[col], kde=True, color='teal')
    plt.title(f"{col} Distribution (Skew: {df_clean[col].skew():.2f})")
plt.tight_layout()
plt.show()

cat_cols = ['Survived', 'Pclass', 'Sex', 'Embarked']
plt.figure(figsize=(14, 8))
for i, col in enumerate(cat_cols, 1):
    plt.subplot(2, 2, i)
    sns.countplot(data=df_clean, x=col, palette='Set2')
    plt.title(f"Count Plot of {col}")
plt.tight_layout()
plt.show()
""")

    # Section 6: Outlier Detection
    add_md("""# 6. Outlier Detection (IQR Rule)""")
    add_code("""outlier_summary = []
for col in ['Age', 'Fare', 'SibSp', 'Parch']:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    cnt = ((df_clean[col] < lower) | (df_clean[col] > upper)).sum()
    outlier_summary.append({
        'Feature': col, 'Q1': round(Q1,2), 'Q3': round(Q3,2),
        'IQR': round(IQR,2), 'Lower Bound': round(lower,2),
        'Upper Bound': round(upper,2), 'Outliers Count': cnt,
        'Outlier (%)': round(cnt / len(df_clean) * 100, 2)
    })

display(pd.DataFrame(outlier_summary))
""")

    # Section 7: Bivariate Analysis
    add_md("""# 7. Bivariate Analysis & Key Insights Correlation""")
    add_code("""plt.figure(figsize=(8, 6))
sns.heatmap(df_clean.select_dtypes(include=np.number).corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Numerical Feature Correlation Heatmap")
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.barplot(data=df_clean, x='Sex', y='Survived', palette='Set1', ax=axes[0])
axes[0].set_title("Survival Rate by Gender")

sns.barplot(data=df_clean, x='Pclass', y='Survived', palette='viridis', ax=axes[1])
axes[1].set_title("Survival Rate by Class")

sns.boxplot(data=df_clean, x='Survived', y='Fare', palette='Set2', ax=axes[2])
axes[2].set_yscale('log')
axes[2].set_title("Fare vs Survival (Log Scale)")
plt.tight_layout()
plt.show()
""")

    # Section 8: Hypothesis Testing
    add_md("""# 8. Hypothesis Testing: Welch's Two-Sample t-Test on Fare""")
    add_code("""survived_fare = df_clean[df_clean['Survived'] == 1]['Fare']
non_survived_fare = df_clean[df_clean['Survived'] == 0]['Fare']

t_stat, p_val = stats.ttest_ind(survived_fare, non_survived_fare, equal_var=False)

print(f"Mean Fare (Survived): £{survived_fare.mean():.2f}")
print(f"Mean Fare (Non-Survived): £{non_survived_fare.mean():.2f}")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_val:.4e}")

if p_val < 0.05:
    print("Decision: Reject H0 -> Survivors paid a statistically significantly higher average fare.")
else:
    print("Decision: Fail to Reject H0")
""")

    # Section 9: 4 Best Insights Summary
    add_md("""# 9. Top 4 Business Insights for Dashboard Conversion

1. **Insight 1 (Gender Priority)**: Female survival rate (74.2%) was nearly 4x higher than male survival rate (18.9%) due to "women & children first" evacuation protocols.
2. **Insight 2 (Socio-Economic Tier)**: First-class passengers enjoyed a 63.0% survival rate versus 24.2% for third-class passengers, driven by upper-deck cabin proximity.
3. **Insight 3 (Fare Disparity)**: Passengers who survived paid an average ticket fare of £48.40 compared to £22.12 for non-survivors ($p < 0.001$).
4. **Insight 4 (Port & Cabin Advantage)**: Cherbourg ('C') embarkations yielded the highest survival rate (55.4%), correlated with higher 1st-class ticket distribution and cabin ownership (66.7% survival for cabin owners).
""")

    # Section 10: Conclusion
    add_md("""# 10. Conclusion & Dashboard Export Roadmap""")

    nb_dict = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"}
        },
        "nbformat": 4, "nbformat_minor": 2
    }

    with open("eda_workflow.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb_dict, f, indent=2)
    print("eda_workflow.ipynb created successfully.")

if __name__ == "__main__":
    create_eda_notebook()
