import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Titanic EDA & Insights Dashboard",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for custom KPI cards, styling, and text formatting
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load & Cache Dataset
# ---------------------------------------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    # Clean data for dashboard
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Has_Cabin'] = df['Cabin'].notnull().astype(int)
    df['Survival_Label'] = df['Survived'].map({0: 'Perished', 1: 'Survived'})
    df['Class_Label'] = df['Pclass'].map({1: '1st Class', 2: '2nd Class', 3: '3rd Class'})
    return df

df = load_data()

# ---------------------------------------------------------
# Sidebar Filters (Interactive Controls)
# ---------------------------------------------------------
st.sidebar.header("🔍 Dashboard Filters")

# Filter 1: Passenger Class Multiselect
class_options = sorted(df['Class_Label'].unique())
selected_classes = st.sidebar.multiselect(
    "Select Passenger Class:",
    options=class_options,
    default=class_options
)

# Filter 2: Gender Selectbox
gender_options = ["All", "female", "male"]
selected_gender = st.sidebar.selectbox(
    "Select Gender:",
    options=gender_options,
    index=0
)

# Filter 3: Age Range Slider
min_age = float(df['Age'].min())
max_age = float(df['Age'].max())
selected_age_range = st.sidebar.slider(
    "Select Age Range (Years):",
    min_value=0.0,
    max_value=80.0,
    value=(min_age, max_age),
    step=1.0
)

# Filter 4: Embarked Port Multiselect
port_map = {'S': 'Southampton', 'C': 'Cherbourg', 'Q': 'Queenstown'}
df['Port_Name'] = df['Embarked'].map(port_map)
port_options = list(port_map.values())
selected_ports = st.sidebar.multiselect(
    "Select Port of Embarkation:",
    options=port_options,
    default=port_options
)

# ---------------------------------------------------------
# Apply Filters to Dataframe
# ---------------------------------------------------------
filtered_df = df.copy()

if selected_classes:
    filtered_df = filtered_df[filtered_df['Class_Label'].isin(selected_classes)]

if selected_gender != "All":
    filtered_df = filtered_df[filtered_df['Sex'] == selected_gender]

filtered_df = filtered_df[
    (filtered_df['Age'] >= selected_age_range[0]) & 
    (filtered_df['Age'] <= selected_age_range[1])
]

if selected_ports:
    filtered_df = filtered_df[filtered_df['Port_Name'].isin(selected_ports)]

# ---------------------------------------------------------
# Main Title & Header
# ---------------------------------------------------------
st.markdown('<div class="main-header">🚢 Titanic Disaster: Statistical EDA & Insights Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Interactive analytics converting §3.1 Exploratory Data Analysis into executive key metrics and visual insights.</div>', unsafe_allow_html=True)

if filtered_df.empty:
    st.warning("⚠️ No passengers match the selected filter criteria. Please adjust your filters in the sidebar.")
    st.stop()

# ---------------------------------------------------------
# KPI Row (st.metric Cards)
# ---------------------------------------------------------
st.subheader("📌 Key Performance Indicators (KPIs)")

col1, col2, col3, col4 = st.columns(4)

total_passengers = len(filtered_df)
overall_survival_rate = (filtered_df['Survived'].mean() * 100) if total_passengers > 0 else 0
avg_fare = filtered_df['Fare'].mean() if total_passengers > 0 else 0
cabin_rate = (filtered_df['Has_Cabin'].mean() * 100) if total_passengers > 0 else 0

with col1:
    st.metric(
        label="Overall Survival Rate",
        value=f"{overall_survival_rate:.1f}%",
        delta=f"{(overall_survival_rate - 38.4):+.1f}% vs Overall"
    )

with col2:
    st.metric(
        label="Filtered Passengers",
        value=f"{total_passengers:,}",
        delta=f"{total_passengers} total"
    )

with col3:
    st.metric(
        label="Average Ticket Fare",
        value=f"£{avg_fare:.2f}",
        delta=f"£{avg_fare - 32.20:+.2f} vs Avg"
    )

with col4:
    st.metric(
        label="Cabin Ownership Rate",
        value=f"{cabin_rate:.1f}%",
        delta="Socio-economic Indicator"
    )

st.divider()

# ---------------------------------------------------------
# Chart Section: 4 Best Insights + Plotly Visualizations
# ---------------------------------------------------------

col_chart1, col_chart2 = st.columns(2)

# ---------------------------------------------------------
# Insight 1 & Chart 1: Survival Rate by Gender & Class
# ---------------------------------------------------------
with col_chart1:
    st.subheader("1. Survival Rate by Gender & Class")
    
    gender_class_df = filtered_df.groupby(['Sex', 'Class_Label'])['Survived'].mean().reset_index()
    gender_class_df['Survival_Rate'] = (gender_class_df['Survived'] * 100).round(1)
    
    fig1 = px.bar(
        gender_class_df,
        x='Class_Label',
        y='Survival_Rate',
        color='Sex',
        barmode='group',
        text='Survival_Rate',
        color_discrete_map={'female': '#EC4899', 'male': '#3B82F6'},
        labels={'Survival_Rate': 'Survival Rate (%)', 'Class_Label': 'Ticket Class', 'Sex': 'Gender'},
        title="Survival Percentage by Gender & Ticket Class"
    )
    fig1.update_traces(texttemplate='%{text}%', textposition='outside')
    fig1.update_layout(height=400, yaxis_range=[0, 105], margin=dict(l=20, r=20, t=50, b=20))
    
    st.plotly_chart(fig1, width='stretch')
    
    st.info(
        "💡 **Insight 1 (Gender Protocol & Social Tier)**: Females achieved a **74.2% overall survival rate** compared to **18.9% for males**, "
        "confirming that evacuation protocol ('women and children first') was strictly enforced. Furthermore, 1st Class females survived at **96.8%**, "
        "proving that socio-economic status significantly multiplied gender-based survival chances."
    )

# ---------------------------------------------------------
# Insight 2 & Chart 2: Ticket Fare Distribution vs Survival
# ---------------------------------------------------------
with col_chart2:
    st.subheader("2. Ticket Fare vs Survival (Log Scale)")
    
    fig2 = px.box(
        filtered_df,
        x='Survival_Label',
        y='Fare',
        color='Survival_Label',
        points='outliers',
        log_y=True,
        color_discrete_map={'Perished': '#EF4444', 'Survived': '#10B981'},
        labels={'Fare': 'Ticket Fare (£, Log Scale)', 'Survival_Label': 'Outcome'},
        title="Distribution of Ticket Fare by Survival Outcome"
    )
    fig2.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
    
    st.plotly_chart(fig2, width='stretch')
    
    st.info(
        "💡 **Insight 2 (Financial Premium & Stat Sig)**: Passengers who survived paid a statistically significantly higher mean fare "
        "(**£48.40 vs £22.12**, Welch's t-test $p < 0.001$). The log-scaled plot highlights that extreme luxury fare outliers (£100–£512) "
        "belonged almost exclusively to survivors."
    )

st.divider()

col_chart3, col_chart4 = st.columns(2)

# ---------------------------------------------------------
# Insight 3 & Chart 3: Age Distribution & Survival Density
# ---------------------------------------------------------
with col_chart3:
    st.subheader("3. Age Distribution by Survival Outcome")
    
    fig3 = px.histogram(
        filtered_df,
        x='Age',
        color='Survival_Label',
        barmode='overlay',
        nbins=40,
        opacity=0.6,
        color_discrete_map={'Perished': '#EF4444', 'Survived': '#10B981'},
        labels={'Age': 'Age (Years)', 'count': 'Passenger Count'},
        title="Age Distribution & Demographic Vulnerability"
    )
    fig3.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
    
    st.plotly_chart(fig3, width='stretch')
    
    st.info(
        "💡 **Insight 3 (Demographic Vulnerability)**: Children under 10 years experienced a survival spike (~59% survival), "
        "whereas young adults aged 18–30 suffered the highest mortality density. Median imputation ($28.0$ years) successfully preserved central "
        "tendency without distorting age-based survival trends."
    )

# ---------------------------------------------------------
# Insight 4 & Chart 4: Survival by Port of Embarkation
# ---------------------------------------------------------
with col_chart4:
    st.subheader("4. Survival Rate by Embarkation Port")
    
    port_survival = filtered_df.groupby('Port_Name')['Survived'].agg(['mean', 'count']).reset_index()
    port_survival['Survival_Rate'] = (port_survival['mean'] * 100).round(1)
    
    fig4 = px.bar(
        port_survival,
        x='Port_Name',
        y='Survival_Rate',
        color='Port_Name',
        text='Survival_Rate',
        color_discrete_sequence=px.colors.qualitative.Dark2,
        labels={'Survival_Rate': 'Survival Rate (%)', 'Port_Name': 'Port of Embarkation'},
        title="Survival Rate by Embarkation Port"
    )
    fig4.update_traces(texttemplate='%{text}%', textposition='outside')
    fig4.update_layout(height=400, yaxis_range=[0, 105], margin=dict(l=20, r=20, t=50, b=20))
    
    st.plotly_chart(fig4, width='stretch')
    
    st.info(
        "💡 **Insight 4 (Port & Cabin Proximity Advantage)**: Cherbourg ('C') embarkations yielded the highest survival rate (**55.4%**), "
        "driven by a high concentration of 1st-class ticket holders boarding at Cherbourg. Passengers with recorded cabins achieved a **66.7% survival rate** "
        "compared to **29.9%** for non-cabin passengers."
    )

st.divider()
st.caption("🚀 Built for §3.1 EDA Workflow & Interactive Streamlit Dashboard | Dataset: Kaggle Titanic Mirror")
