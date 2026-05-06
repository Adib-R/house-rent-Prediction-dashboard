import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# PAGE CONFIG
st.set_page_config(page_title="Indian House Rent Prediction Dashboard", layout="wide")

# CUSTOM CSS
st.markdown("""
<style>
.block-container {
    max-width: 1100px;
    margin: auto;
    padding-top: 1.5rem;
}
.card {
    background-color: #1C1F26;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}
h1, h2, h3 {
    text-align: center;
}
@media (max-width: 768px) {
    .block-container {
        padding: 1rem;
    }
}
</style>
""", unsafe_allow_html=True)

# LOAD DATA
@st.cache_data
def load_data():
    return pd.read_csv("data.csv")

df = load_data()

# ✅ KEEP ORIGINAL DATA
df_raw = df.copy()

# DATA CLEANING
df.fillna(df.median(numeric_only=True), inplace=True)
df["rent"] = df["rent"].clip(df["rent"].quantile(0.01), df["rent"].quantile(0.99))

# FEATURE ENGINEERING
df["bath_per_bed"] = df["bathrooms"] / (df["beds"] + 1)
df["room_density"] = df["area"] / (df["beds"] + 1)
df["bed_bath_ratio"] = df["beds"] / (df["bathrooms"] + 1)
df["area_per_room"] = df["area"] / (df["beds"] + df["bathrooms"] + 1)
df["locality_target"] = df.groupby("locality")["rent"].transform("mean")

# MODEL TRAINING
@st.cache_resource
def train_models(data):

    df_ml = data.drop(columns=["house_type", "area_rate", "locality"])
    df_ml = pd.get_dummies(df_ml, drop_first=True)

    X = df_ml.drop("rent", axis=1)
    y = np.log1p(df_ml["rent"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=12, random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=400,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        actual = np.expm1(y_test)
        predicted = np.expm1(pred)

        results[name] = {
            "model": model,
            "r2": r2_score(actual, predicted),
            "rmse": np.sqrt(mean_squared_error(actual, predicted)),
            "mae": mean_absolute_error(actual, predicted)
        }

    best_model = results["Random Forest"]["model"]
    cv_score = cross_val_score(best_model, X_train, y_train, cv=5, scoring="r2").mean()

    return best_model, results, X.columns, X_test, y_test, cv_score


model, results, feature_cols, X_test, y_test, cv_score = train_models(df)

# HEADER
st.markdown("""
<h1>Indian House Rent Prediction System</h1>
<h3 style='color:#00FFAA;'>Machine Learning-Based Rental Price Estimation</h3>
""", unsafe_allow_html=True)

# KPI SECTION
col1, col2, col3 = st.columns(3)

col1.markdown(f"<div class='card'><h4>Total Property Listings</h4><h2>{len(df)}</h2></div>", unsafe_allow_html=True)
col2.markdown(f"<div class='card'><h4>Number of Cities</h4><h2>{df['city'].nunique()}</h2></div>", unsafe_allow_html=True)
col3.markdown(f"<div class='card'><h4>Average Rental Price</h4><h2>₹{int(df['rent'].mean())}</h2></div>", unsafe_allow_html=True)

st.markdown("---")

# SIDEBAR NAVIGATION
menu = st.sidebar.radio(
    "Navigation Menu",
    ["Exploratory Data Analysis", "Model Evaluation", "Rent Prediction"]
)

# ===================== EDA =====================
if menu == "Exploratory Data Analysis":

    st.markdown("## Exploratory Data Analysis")

    # 🔥 ADDED FOR TESTING (IMPORTANT)
    st.subheader("Raw Data (Before Preprocessing)")
    st.dataframe(df_raw.head())

    st.subheader("Processed Data (After Preprocessing)")
    st.dataframe(df.head())

    st.subheader("Missing Values Comparison")
    st.write("Before:", df_raw.isnull().sum())
    st.write("After:", df.isnull().sum())

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Distribution of Rental Prices")
        st.plotly_chart(px.histogram(df, x="rent"), use_container_width=True)

    with col2:
        st.markdown("### Average Rent by City")
        city_avg = df.groupby("city")["rent"].mean().reset_index()
        st.plotly_chart(px.bar(city_avg, x="city", y="rent"), use_container_width=True)

# (Rest of your code unchanged)
