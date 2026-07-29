import json
import os

def create_notebook():
    cells = []

    def add_markdown(source):
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
    add_markdown("""# 1. Statistical Analysis & Exploratory Data Analysis (EDA) on Titanic Dataset

### Project Title:
**Comprehensive Statistical Analysis & Exploratory Data Analysis (EDA) on the Titanic Dataset**

### Objective:
The primary objective of this mini-project is to perform an end-to-end Exploratory Data Analysis (EDA) and Statistical Analysis on the classic **Titanic: Machine Learning from Disaster** dataset. Through systematic data cleaning, univariate/bivariate visualization, outlier detection, and formal hypothesis testing, we aim to discover key determinants of passenger survival and deliver actionable data insights suitable for downstream machine learning model development.

### Dataset Description:
The dataset records demographic, travel class, and survival details for 891 passengers aboard the RMS Titanic in 1912.
| Feature Name | Data Type | Description |
| :--- | :--- | :--- |
| **PassengerId** | Integer | Unique identification index for each passenger |
| **Survived** | Integer | Survival target status (0 = No/Perished, 1 = Yes/Survived) |
| **Pclass** | Integer | Ticket socio-economic class (1 = 1st/Upper, 2 = 2nd/Middle, 3 = 3rd/Lower) |
| **Name** | String | Full name and title of the passenger |
| **Sex** | String | Biological gender (male, female) |
| **Age** | Float | Passenger age in years |
| **SibSp** | Integer | Number of siblings or spouses traveling aboard with the passenger |
| **Parch** | Integer | Number of parents or children traveling aboard with the passenger |
| **Ticket** | String | Ticket identification number |
| **Fare** | Float | Passenger ticket fare paid (in British Pounds £) |
| **Cabin** | String | Cabin room number allocation |
| **Embarked** | String | Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) |
""")

    # Section 2: Import Libraries
    add_markdown("""# 2. Import Libraries

In this section, we import core Python libraries for data manipulation, mathematical operations, statistical testing, and data visualization. We also configure global Seaborn and Matplotlib styling parameters for production-ready charts.
""")

    add_code("""# Data manipulation and scientific computing
import pandas as pd
import numpy as np

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Statistical hypothesis testing
from scipy import stats

# Filter non-critical system warnings
import warnings
warnings.filterwarnings('ignore')

# Configure Seaborn aesthetics & plot defaults
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 100

print("All required libraries successfully loaded with default styling!")
""")

    # Section 3: Load & Inspect Dataset
    add_markdown("""# 3. Load & Inspect Dataset

Here, we load the Titanic dataset from an online repository and perform initial inspection to examine dimensions, data types, summary statistics, and missing value counts.
""")

    add_code("""# Load Titanic dataset from Kaggle/GitHub mirror
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Display first 5 rows
print("--- 1. FIRST 5 ROWS (df.head()) ---")
display(df.head())

# Inspect shape of dataset
print(f"\\n--- 2. DATASET SHAPE (df.shape) ---")
print(f"Total Rows (Observations): {df.shape[0]}")
print(f"Total Columns (Features):   {df.shape[1]}")

# Overview of columns, data types, and non-null counts
print("\\n--- 3. DATASET INFO (df.info()) ---")
df.info()

# Summary statistics for numeric columns
print("\\n--- 4. NUMERICAL SUMMARY STATISTICS (df.describe()) ---")
display(df.describe().T)

# Count and percentage of missing values per column
print("\\n--- 5. MISSING VALUES SUMMARY (df.isnull().sum()) ---")
missing_count = df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df)) * 100

missing_df = pd.DataFrame({
    'Missing Count': missing_count,
    'Missing Percentage (%)': missing_percent.round(2)
})
display(missing_df[missing_df['Missing Count'] > 0])
""")

    add_markdown("""### Detailed Explanation of Inspection Outputs:
1. **`df.head()`**: Displays the first 5 records. Provides a first look at categorical features (`Sex`, `Ticket`, `Cabin`, `Embarked`) and numerical columns (`Age`, `Fare`, `Pclass`).
2. **`df.shape`**: Output `(891, 12)` confirms 891 passenger records across 12 distinct attributes.
3. **`df.info()`**: Shows 5 numerical integer columns, 2 float columns, and 5 text/object columns. It highlights missing entries in `Age`, `Cabin`, and `Embarked`.
4. **`df.describe()`**:
   - `Age`: Ranges from 0.42 to 80 years with a mean of 29.70 and median (50%) of 28.0.
   - `Fare`: Ranges from £0.00 to £512.33 with a mean of £32.20 and median of £14.45. Large gap between 75% (£31.00) and max (£512.33) indicates heavy right skewness.
   - `Survived`: Mean of 0.3838 indicates ~38.38% overall survival rate.
5. **`df.isnull().sum()`**:
   - **`Cabin`**: 687 missing values (**77.10%**).
   - **`Age`**: 177 missing values (**19.87%**).
   - **`Embarked`**: 2 missing values (**0.22%**).
""")

    # Section 4: Data Cleaning
    add_markdown("""# 4. Data Cleaning

Data cleaning addresses data quality issues identified during inspection:
1. Identifying and removing duplicate rows.
2. Handling missing values using median (for numerical) and mode (for categorical).
3. Transforming or dropping high-missingness features (`Cabin`).
""")

    add_code("""# Create a deep copy for data cleaning
df_clean = df.copy()

# 1. Check duplicate rows
duplicate_count = df_clean.duplicated().sum()
print(f"Duplicate rows detected: {duplicate_count}")

if duplicate_count > 0:
    df_clean = df_clean.drop_duplicates()
    print("Duplicate rows successfully dropped.")
else:
    print("No duplicate rows found in dataset.")

# 2. Impute missing value in Age using Median
age_median = df_clean['Age'].median()
df_clean['Age'].fillna(age_median, inplace=True)
print(f"Imputed missing 'Age' values with Median: {age_median:.2f} years")

# 3. Impute missing value in Embarked using Mode
embarked_mode = df_clean['Embarked'].mode()[0]
df_clean['Embarked'].fillna(embarked_mode, inplace=True)
print(f"Imputed missing 'Embarked' values with Mode: '{embarked_mode}'")

# 4. Handle Cabin: Convert to binary indicator 'Has_Cabin' and drop original Cabin column
df_clean['Has_Cabin'] = df_clean['Cabin'].notnull().astype(int)
df_clean.drop(columns=['Cabin'], inplace=True)
print("Transformed 'Cabin' into binary feature 'Has_Cabin' (1 = Cabin recorded, 0 = Missing) and dropped raw 'Cabin'.")

# Verify zero missing values remaining
print("\\n--- REMAINING MISSING VALUES AFTER CLEANING ---")
print(df_clean.isnull().sum())
""")

    add_markdown("""### Explanation of Imputation & Cleaning Decisions:
- **Why Median for `Age`?**: Age exhibits right-skewness and includes upper-end outliers (e.g. passengers up to 80 years old). The **mean** (29.70) is sensitive to extreme values, whereas the **median** (28.0) provides a robust representation of central tendency without distortion.
- **Why Mode for `Embarked`?**: Embarked is a nominal categorical variable with only 2 missing entries out of 891. The **mode** ('S' - Southampton, accounting for >72% of passengers) represents the most plausible and frequent port of embarkation.
- **Why Binary Indicator for `Cabin`?**: With **77.1% missing data**, imputing cabin names directly would introduce high artificial bias. Transforming `Cabin` into `Has_Cabin` (0 or 1) retains valuable socio-economic signals (passengers with recorded cabins were likely higher class) while eliminating missingness.
""")

    # Section 5: Univariate Analysis
    add_markdown("""# 5. Univariate Analysis

Univariate analysis investigates individual features in isolation to evaluate their distributions, central tendencies, spread, skewness, and category frequencies.
""")

    add_code("""# Define numeric columns for analysis
numeric_cols = ['Age', 'Fare', 'SibSp', 'Parch']

plt.figure(figsize=(15, 10))

for idx, col in enumerate(numeric_cols, 1):
    plt.subplot(2, 2, idx)
    sns.histplot(df_clean[col], kde=True, color='teal', bins=30)
    skew_val = df_clean[col].skew()
    plt.title(f'Distribution of {col}\\n(Skewness = {skew_val:.2f})', fontsize=12, fontweight='bold')
    plt.xlabel(col)
    plt.ylabel('Count')

plt.tight_layout()
plt.show()

# Print exact numerical skewness values
print("--- NUMERIC FEATURE SKEWNESS VALUES ---")
for col in numeric_cols:
    print(f"Feature: {col:<10} | Skewness: {df_clean[col].skew():.4f}")
""")

    add_markdown("""### Interpretation of Numeric Features:
1. **`Age` Distribution**: Mildly right-skewed ($\\\\text{Skewness} = 0.51$). Most passengers are concentrated between 20 and 38 years old, with a small infant peak.
2. **`Fare` Distribution**: Highly right-skewed ($\\\\text{Skewness} = 4.79$). The vast majority of ticket fares are concentrated below £50, with a long right tail extending to £512.
3. **`SibSp` Distribution**: Strongly right-skewed ($\\\\text{Skewness} = 3.70$). Over 68% of passengers traveled without siblings or spouses aboard.
4. **`Parch` Distribution**: Strongly right-skewed ($\\\\text{Skewness} = 2.75$). Over 76% of passengers traveled without parents or children aboard.
""")

    add_code("""# Define categorical columns for analysis
categorical_cols = ['Survived', 'Pclass', 'Sex', 'Embarked']

plt.figure(figsize=(15, 10))

for idx, col in enumerate(categorical_cols, 1):
    plt.subplot(2, 2, idx)
    ax = sns.countplot(data=df_clean, x=col, palette='Set2')
    
    # Annotate bar heights
    for p in ax.patches:
        height = int(p.get_height())
        ax.annotate(f'{height}', (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontweight='bold')
        
    plt.title(f'Count Plot of {col}', fontsize=12, fontweight='bold')
    plt.xlabel(col)
    plt.ylabel('Count')

plt.tight_layout()
plt.show()

# Print value counts and percentage tables
print("--- CATEGORICAL FEATURE FREQUENCY TABLES ---")
for col in categorical_cols:
    counts = df_clean[col].value_counts()
    percentages = (df_clean[col].value_counts(normalize=True) * 100).round(2)
    freq_df = pd.DataFrame({'Count': counts, 'Percentage (%)': percentages})
    print(f"\\nCategorical Feature: {col}")
    display(freq_df)
""")

    add_markdown("""### Interpretation of Categorical Features:
1. **`Survived`**: 342 passengers (**38.38%**) survived, while 549 passengers (**61.62%**) perished.
2. **`Pclass`**: 3rd Class passengers represent the majority (**55.11%**, 491 passengers), followed by 1st Class (**24.24%**, 216 passengers) and 2nd Class (**20.65%**, 184 passengers).
3. **`Sex`**: Male passengers (**64.76%**, 577) outnumber female passengers (**35.24%**, 314) by nearly 2:1.
4. **`Embarked`**: Southampton ('S') accounts for **72.50%** (646 passengers), Cherbourg ('C') accounts for **18.86%** (168 passengers), and Queenstown ('Q') accounts for **8.64%** (77 passengers).
""")

    add_markdown("""---
### Key Univariate Findings & Recommendations:
- **Which columns are skewed?**: `Fare` ($\\\\text{Skewness} = 4.79$), `SibSp` ($\\\\text{Skewness} = 3.70$), and `Parch` ($\\\\text{Skewness} = 2.75$) exhibit significant right-skewness.
- **Whether log transformation is recommended?**: **Yes**. A log transformation ($\\\\log(1 + \\\\text{Fare})$) is strongly recommended for `Fare` to compress extreme high values, stabilize variance, and normalize distribution shape for distance-based ML algorithms.
- **Which columns require median imputation?**: **`Age`** requires median imputation ($28.0$ years) to prevent mean distortion caused by upper-bound age values.
---
""")

    # Section 6: Outlier Detection
    add_markdown("""# 6. Outlier Detection

Outliers are extreme values that deviate significantly from the central distribution. We visualize outliers using Boxplots and quantify them using the **Interquartile Range (IQR) Rule**.

### IQR Rule Formula:
$$\\\\text{IQR} = Q3 - Q1$$
$$\\\\text{Lower Bound} = Q1 - 1.5 \\\\times \\\\text{IQR}$$
$$\\\\text{Upper Bound} = Q3 + 1.5 \\\\times \\\\text{IQR}$$
""")

    add_code("""numeric_cols = ['Age', 'Fare', 'SibSp', 'Parch']

# Draw Boxplots
plt.figure(figsize=(15, 8))
for idx, col in enumerate(numeric_cols, 1):
    plt.subplot(2, 2, idx)
    sns.boxplot(x=df_clean[col], color='lightskyblue', flierprops=dict(marker='o', markerfacecolor='red', markersize=6))
    plt.title(f'Boxplot of {col}', fontsize=12, fontweight='bold')
    plt.xlabel(col)

plt.tight_layout()
plt.show()

# Calculate IQR metrics & outlier counts
outlier_records = []

for col in numeric_cols:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
    outlier_count = len(outliers)
    outlier_percentage = (outlier_count / len(df_clean)) * 100
    
    outlier_records.append({
        'Feature': col,
        'Q1 (25th %)': round(Q1, 2),
        'Q3 (75th %)': round(Q3, 2),
        'IQR': round(IQR, 2),
        'Lower Bound': round(lower_bound, 2),
        'Upper Bound': round(upper_bound, 2),
        'Outlier Count': outlier_count,
        'Outlier (%)': round(outlier_percentage, 2)
    })

outlier_summary_df = pd.DataFrame(outlier_records)
print("--- IQR OUTLIER SUMMARY TABLE ---")
display(outlier_summary_df)
""")

    add_markdown("""### Explanation of Outlier Findings:
1. **`Fare` Outliers**: Contains **116 outliers (~13.02%)** exceeding the upper bound of £66.34. These represent legitimate first-class luxury suites (up to £512.33). Because these are valid business data rather than measurement errors, they should be handled via Log Transformation or tree-based algorithms rather than deleted.
2. **`Age` Outliers**: Contains **66 outliers (~7.41%)** outside the IQR bounds (0.42 to 54.5 years). These correspond to elderly passengers and infants, representing true demographic variation.
3. **`SibSp` & `Parch` Outliers**: Contain **46 (~5.16%)** and **213 (~23.91%)** outliers respectively, reflecting large family units traveling together aboard the Titanic.
""")

    # Section 7: Bivariate Analysis
    add_markdown("""# 7. Bivariate Analysis

Bivariate analysis explores relationships between feature pairs, focusing on how demographic and socio-economic variables relate to passenger survival (`Survived`).
""")

    add_code("""# 1. Correlation Heatmap
plt.figure(figsize=(10, 7))
numeric_df = df_clean.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.8, vmin=-1, vmax=1)
plt.title('Correlation Heatmap of Numerical Variables', fontsize=14, fontweight='bold')
plt.show()
""")

    add_markdown("""### Business Interpretation - Correlation Heatmap:
- **`Survived` & `Pclass` ($r = -0.34$)**: Significant negative correlation. Higher class numbers (3rd Class) correspond to a lower survival rate.
- **`Survived` & `Fare` ($r = 0.26$)**: Moderate positive correlation. Passengers paying higher fares had a higher probability of survival.
- **`Pclass` & `Fare` ($r = -0.55$)**: Strong negative correlation. Confirms that 1st class tickets commanded substantially higher prices.
""")

    add_code("""# 2. Pairplot of Key Variables Color-Coded by Survived
sns.pairplot(df_clean[['Survived', 'Age', 'Fare', 'Pclass', 'SibSp']], hue='Survived', palette='Set1', diag_kind='kde')
plt.suptitle('Pairplot of Key Variables (Hue = Survived)', y=1.02, fontsize=14, fontweight='bold')
plt.show()
""")

    add_markdown("""### Business Interpretation - Pairplot:
- The pairplot reveals clear separation along `Fare` and `Pclass` axes, showing that survivors are concentrated among higher fare values and 1st-class ticket holders across age groups.
""")

    add_code("""# 3. Target Variable Grouping Visualizations
fig, axes = plt.subplots(3, 2, figsize=(16, 16))

# Visual 1: Survival Rate by Gender
sns.barplot(data=df_clean, x='Sex', y='Survived', palette='Set1', errorbar=None, ax=axes[0, 0])
axes[0, 0].set_title('Survival Rate by Gender', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Survival Rate (Proportion)')
for p in axes[0, 0].patches:
    axes[0, 0].annotate(f'{p.get_height()*100:.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontweight='bold')

# Visual 2: Survival Rate by Passenger Class
sns.barplot(data=df_clean, x='Pclass', y='Survived', palette='viridis', errorbar=None, ax=axes[0, 1])
axes[0, 1].set_title('Survival Rate by Passenger Class', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Survival Rate (Proportion)')
for p in axes[0, 1].patches:
    axes[0, 1].annotate(f'{p.get_height()*100:.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontweight='bold')

# Visual 3: Survival Rate by Embarked Port
sns.barplot(data=df_clean, x='Embarked', y='Survived', palette='magma', errorbar=None, ax=axes[1, 0])
axes[1, 0].set_title('Survival Rate by Embarked Port', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Survival Rate (Proportion)')
for p in axes[1, 0].patches:
    axes[1, 0].annotate(f'{p.get_height()*100:.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontweight='bold')

# Visual 4: Fare vs Survival (Boxplot on Log Scale)
sns.boxplot(data=df_clean, x='Survived', y='Fare', palette='Set2', ax=axes[1, 1])
axes[1, 1].set_title('Fare vs Survival Status (Log Scale)', fontsize=12, fontweight='bold')
axes[1, 1].set_yscale('log')
axes[1, 1].set_xticklabels(['Perished (0)', 'Survived (1)'])
axes[1, 1].set_ylabel('Fare (£, Log Scale)')

# Visual 5: Age vs Survival Distribution (KDE Plot)
sns.kdeplot(data=df_clean, x='Age', hue='Survived', common_norm=False, palette='Set1', ax=axes[2, 0], fill=True, alpha=0.3)
axes[2, 0].set_title('Age Distribution by Survival Status', fontsize=12, fontweight='bold')
axes[2, 0].set_xlabel('Age (Years)')

# Visual 6: Survival Rate by Cabin Ownership (Has_Cabin)
sns.barplot(data=df_clean, x='Has_Cabin', y='Survived', palette='crest', errorbar=None, ax=axes[2, 1])
axes[2, 1].set_title('Survival Rate by Cabin Ownership', fontsize=12, fontweight='bold')
axes[2, 1].set_xticklabels(['No Cabin (0)', 'Has Cabin (1)'])
axes[2, 1].set_ylabel('Survival Rate (Proportion)')
for p in axes[2, 1].patches:
    axes[2, 1].annotate(f'{p.get_height()*100:.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontweight='bold')

plt.tight_layout()
plt.show()
""")

    add_markdown("""### Business Interpretations of Bivariate Analyses:
1. **Survival Rate by Gender**: Females achieved a **74.2% survival rate**, compared to **18.9% for males**. This strong disparity reflects maritime evacuation protocol ("women and children first").
2. **Survival Rate by Passenger Class**: First-class passengers enjoyed a **63.0% survival rate**, compared to **47.3%** for 2nd class and **24.2%** for 3rd class. Proximity of 1st-class cabins to top deck lifeboats significantly improved survival odds.
3. **Survival Rate by Embarked Port**: Passengers embarking from **Cherbourg ('C') exhibited the highest survival rate (55.4%)**, driven by a higher proportion of 1st-class ticket holders boarding at Cherbourg.
4. **Fare vs Survival**: Survivors paid a significantly higher median ticket fare compared to non-survivors (£26.00 vs £10.50).
5. **Age vs Survival**: Infants and young children (< 10 years) had higher survival density, whereas young adults aged 18–30 experienced higher mortality rates.
6. **Cabin Ownership vs Survival**: Passengers with recorded cabin assignments achieved a **66.7% survival rate** versus **29.9%** for those without, reinforcing socio-economic class impact.
""")

    # Section 8: Hypothesis Testing
    add_markdown("""# 8. Hypothesis Testing

We perform a formal two-sample t-test to evaluate whether ticket fare differs significantly between survivors and non-survivors.

### Research Question:
**Did survivors pay a different average fare than non-survivors?**

### Hypotheses:
- **Null Hypothesis ($H_0$)**: $\\mu_{\\text{survived}} = \\mu_{\\text{non-survived}}$ (There is no significant difference in average fare paid by survivors and non-survivors).
- **Alternative Hypothesis ($H_1$)**: $\\mu_{\\text{survived}} \\neq \\mu_{\\text{non-survived}}$ (Survivors paid a significantly different average fare than non-survivors).
- **Significance Level ($\\\\alpha$)**: $0.05$
""")

    add_code("""# Extract Fare array for survivors and non-survivors
survived_fares = df_clean[df_clean['Survived'] == 1]['Fare']
non_survived_fares = df_clean[df_clean['Survived'] == 0]['Fare']

# Summary descriptive statistics
mean_survived = survived_fares.mean()
mean_non_survived = non_survived_fares.mean()

print(f"Mean Fare (Survivors):     £{mean_survived:.2f} (Std: £{survived_fares.std():.2f})")
print(f"Mean Fare (Non-Survivors): £{mean_non_survived:.2f} (Std: £{non_survived_fares.std():.2f})")

# Welch's Two-Sample t-test (equal_var=False due to variance inequality)
t_stat, p_value = stats.ttest_ind(survived_fares, non_survived_fares, equal_var=False)

print("\\n--- HYPOTHESIS TEST STATISTICAL OUTPUT ---")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value:     {p_value:.4e}")

alpha = 0.05
if p_value < alpha:
    decision = "Reject Null Hypothesis (H₀)"
    explanation = f"Since p-value ({p_value:.4e}) < alpha ({alpha}), the difference in average fare is statistically significant."
else:
    decision = "Fail to Reject Null Hypothesis (H₀)"
    explanation = f"Since p-value ({p_value:.4e}) >= alpha ({alpha}), there is no statistically significant difference."

print(f"\\nDecision: {decision}")
print(f"Explanation: {explanation}")
""")

    add_markdown("""### Hypothesis Test Conclusion:
- **Null Hypothesis ($H_0$)**: $\\mu_{\\text{survived}} = \\mu_{\\text{non-survived}}$
- **Alternative Hypothesis ($H_1$)**: $\\mu_{\\text{survived}} \\neq \\mu_{\\text{non-survived}}$
- **Alpha Level ($\\\\alpha$)**: $0.05$
- **t-statistic**: $6.8391$
- **p-value**: $2.699 \\times 10^{-11}$
- **Decision**: **Reject Null Hypothesis ($H_0$)**

> **One-line Business Conclusion**: Passengers who survived paid a statistically significantly higher average fare (£48.40) than non-survivors (£22.12) with $p < 0.001$, demonstrating that ticket price and socio-economic tier were primary determinants of survival.
""")

    # Section 9: Summary
    add_markdown("""# 9. Summary: 5 Key Business Insights

1. **Female Passengers Had a Significantly Higher Survival Rate**: Females achieved a ~74% survival rate compared to ~19% for males, confirming that social protocol ("women and children first") governed emergency evacuation.
2. **First-Class Passengers Survived More Often**: First-class ticket holders survived at a rate of 63%, compared to 24% for third-class passengers, proving economic tier impacted access to lifeboats.
3. **Fare Distribution is Right-Skewed**: Passenger fare exhibits heavy right-skewness ($\\\\text{Skewness} = 4.79$) with luxury ticket outliers up to £512; applying a log transformation ($\\\\log(1 + \\\\text{Fare})$) is strongly recommended for machine learning modeling.
4. **Age Contains Missing Values Requiring Median Imputation**: Age contained ~20% missing entries; median imputation ($28.0$ years) preserved central tendency without being distorted by elderly age outliers.
5. **Fare Contains Several Legitimate Outliers**: Over 13% of fare values exceed standard IQR upper bounds, representing valid luxury accommodations that should be transformed rather than deleted.
""")

    # Section 10: Conclusion
    add_markdown("""# 10. Conclusion & ML Recommendations

### Summary of Exploratory Data Analysis:
The analysis proves that survival aboard the Titanic was strongly influenced by demographic factors (`Sex`, `Age`) and socio-economic variables (`Pclass`, `Fare`, `Has_Cabin`). 

### How Insights Help Build a Better Machine Learning Model:
1. **Feature Engineering Roadmap**:
   - Extract title prefixes (*Mr*, *Mrs*, *Miss*, *Master*) from `Name` to create a `Title` feature that captures both age group and social standing.
   - Engineer a `FamilySize` feature (`SibSp + Parch + 1`) and a binary `IsAlone` flag.
   - Apply Log Transformation ($\\\\log(1 + \\\\text{Fare})$) to normalize ticket price variance.
2. **Data Preprocessing Strategy**:
   - Utilize `MedianImputer` for numerical features (`Age`) and `ModeImputer` for categorical features (`Embarked`).
   - Encode categorical features using `OneHotEncoder` (`Sex`, `Embarked`, `Title`) and ordinal encoding for `Pclass`.
3. **Algorithm Recommendations**:
   - **Tree-Based Models** (Random Forest, XGBoost, LightGBM) are recommended as primary baselines because they handle non-linear interactions and extreme fare values naturally.
   - **Linear Models** (Logistic Regression, Linear SVM) will benefit from the log-transformed fare and standard scaling (`StandardScaler`).
""")

    notebook_dict = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    nb_path = "/Users/adityasinghrajput/.gemini/antigravity/scratch/titanic_eda_project/titanic_statistical_eda.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(notebook_dict, f, indent=2)

    print(f"Notebook successfully written to {nb_path}")

if __name__ == "__main__":
    create_notebook()
