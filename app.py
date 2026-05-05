import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(page_title="House Rent Dashboard", layout="wide")

# ======================
# LOAD DATA
# ======================
file_path = os.path.join(os.path.dirname(__file__), "data.csv")
df = pd.read_csv(file_path)

# ======================
# CLEANING
# ======================
df = df.dropna()
df = df[df["rent"] < df["rent"].quantile(0.95)]

# ======================
# FEATURE ENGINEERING
# ======================
df["bath_per_bed"] = df["bathrooms"] / (df["beds"] + 1)
df["room_density"] = df["area"] / (df["beds"] + 1)

freq = df["locality"].value_counts()
df["locality_freq"] = df["locality"].map(freq)
df = df.drop(columns=["locality"])

# ======================
# MODEL TRAINING FUNCTION
# ======================
@st.cache_resource
def train_models(data):

    df_ml = data.drop(columns=["house_type", "area_rate"])
    df_ml = pd.get_dummies(df_ml, drop_first=True)

    X = df_ml.drop("rent", axis=1)
    y = np.log1p(df_ml["rent"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # MODELS
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        r2 = r2_score(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))

        results[name] = {
            "model": model,
            "r2": r2,
            "rmse": rmse
        }

    # BEST MODEL (Random Forest)
    best_model = results["Random Forest"]["model"]

    # CROSS VALIDATION (extra)
    cv_score = cross_val_score(best_model, X, y, cv=5, scoring='r2').mean()

    return best_model, results, X.columns, X_test, y_test, cv_score


model, results, feature_cols, X_test, y_test, cv_score = train_models(df)

# ======================
# HEADER
# ======================
st.title("🏠 Indian House Rent Prediction")
st.caption(f"{len(df)} listings | {df['city'].nunique()} cities")

# ======================
# SIDEBAR
# ======================
menu = st.sidebar.radio("Navigation", ["EDA", "Model", "Prediction"])

# ======================
# EDA
# ======================
if menu == "EDA":
    st.subheader("📊 Data Insights")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(df, x="rent")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        city_avg = df.groupby("city")["rent"].mean().reset_index()
        fig = px.bar(city_avg, x="city", y="rent")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    ### Key Insights
    - Mumbai has highest rent  
    - Bathrooms strongly affect rent  
    - Area significantly impacts rent  
    - Location plays major role  
    """)

# ======================
# MODEL PERFORMANCE
# ======================
elif menu == "Model":
    st.subheader("🤖 Model Comparison")

    for name, res in results.items():
        st.write(f"{name} → R²: {res['r2']:.2f}, RMSE: {res['rmse']:.2f}")

    st.success(f"Best Model: Random Forest")
    st.info(f"Cross Validation Score (R²): {cv_score:.2f}")

    # Prediction graph
    y_pred = model.predict(X_test)

    actual = np.expm1(y_test)
    predicted = np.expm1(y_pred)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual, y=predicted, mode='markers'))
    fig.add_trace(go.Scatter(x=actual, y=actual, mode='lines'))

    fig.update_layout(
        title="Actual vs Predicted Rent",
        xaxis_title="Actual",
        yaxis_title="Predicted"
    )

    st.plotly_chart(fig)

    # Feature importance
    importance = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    st.subheader("Top Features")
    st.bar_chart(importance.head(10).set_index("Feature"))

# ======================
# PREDICTION
# ======================
elif menu == "Prediction":
    st.subheader("🏠 Predict Rent")

    col1, col2, col3 = st.columns(3)

    with col1:
        area = st.number_input("Area", 300, 5000, 1000)
        city = st.selectbox("City", df["city"].unique())

    with col2:
        bathrooms = st.slider("Bathrooms", 1, 5, 2)
        furnishing = st.selectbox("Furnishing", df["furnishing"].unique())

    with col3:
        bedrooms = st.slider("Bedrooms", 1, 5, 2)

    if st.button("Predict"):

        input_df = pd.DataFrame(np.zeros((1, len(feature_cols))), columns=feature_cols)

        input_df["area"] = area
        input_df["bathrooms"] = bathrooms
        input_df["beds"] = bedrooms
        input_df["bath_per_bed"] = bathrooms / (bedrooms + 1)
        input_df["room_density"] = area / (bedrooms + 1)
        input_df["locality_freq"] = df["locality_freq"].mean()

        for col in feature_cols:
            if col == f"city_{city}":
                input_df[col] = 1
            elif col == f"furnishing_{furnishing}":
                input_df[col] = 1

        prediction = np.expm1(model.predict(input_df)[0])

        st.success(f"Estimated Rent: ₹{int(prediction)}")

        low = int(prediction * 0.9)
        high = int(prediction * 1.1)

        st.write(f"Range: ₹{low} - ₹{high}")
