import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

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
df.fillna(df.median(numeric_only=True), inplace=True)

# CLIP OUTLIERS (better than removing)
lower = df["rent"].quantile(0.01)
upper = df["rent"].quantile(0.99)
df["rent"] = df["rent"].clip(lower, upper)

# ======================
# FEATURE ENGINEERING
# ======================
df["bath_per_bed"] = df["bathrooms"] / (df["beds"] + 1)
df["room_density"] = df["area"] / (df["beds"] + 1)

# NEW FEATURES 🔥
df["bed_bath_ratio"] = df["beds"] / (df["bathrooms"] + 1)
df["area_per_room"] = df["area"] / (df["beds"] + df["bathrooms"] + 1)

# TARGET ENCODING (IMPORTANT)
locality_mean = df.groupby("locality")["rent"].mean()
df["locality_target"] = df["locality"].map(locality_mean)

# ======================
# MODEL TRAINING
# ======================
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

        "Decision Tree": DecisionTreeRegressor(
            max_depth=12,
            random_state=42
        ),

        # 🔥 IMPROVED RANDOM FOREST
        "Random Forest": RandomForestRegressor(
            n_estimators=400,
            max_depth=20,
            min_samples_split=4,
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1
        ),

        # 🔥 NEW MODEL (VERY IMPORTANT)
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            random_state=42
        )
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        actual = np.expm1(y_test)
        predicted = np.expm1(pred)

        r2 = r2_score(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)

        results[name] = {
            "model": model,
            "r2": r2,
            "rmse": rmse,
            "mae": mae
        }

    # Best model selection
    best_model_name = max(results, key=lambda x: results[x]["r2"])
    best_model = results[best_model_name]["model"]

    # Cross-validation
    cv_score = cross_val_score(best_model, X_train, y_train, cv=5, scoring="r2").mean()

    return best_model, best_model_name, results, X.columns, X_test, y_test, cv_score


model, best_model_name, results, feature_cols, X_test, y_test, cv_score = train_models(df)

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
        fig = px.histogram(df, x="rent", title="Rent Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        city_avg = df.groupby("city")["rent"].mean().reset_index()
        fig = px.bar(city_avg, x="city", y="rent", title="Average Rent by City")
        st.plotly_chart(fig, use_container_width=True)

# ======================
# MODEL
# ======================
elif menu == "Model":
    st.subheader("🤖 Model Comparison")

    result_df = pd.DataFrame(results).T[["r2", "rmse", "mae"]]
    result_df.columns = ["R² Score", "RMSE", "MAE"]

    st.dataframe(result_df)

    st.success(f"Best Model: {best_model_name}")
    st.info(f"Cross Validation Score (R²): {cv_score:.2f}")

    y_pred = model.predict(X_test)

    actual = np.expm1(y_test)
    predicted = np.expm1(y_pred)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual, y=predicted, mode='markers', name="Predicted"))
    fig.add_trace(go.Scatter(x=actual, y=actual, mode='lines', name="Perfect Fit"))

    fig.update_layout(
        title="Actual vs Predicted Rent",
        xaxis_title="Actual Rent",
        yaxis_title="Predicted Rent"
    )

    st.plotly_chart(fig)

    if hasattr(model, "feature_importances_"):
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

        input_df = pd.DataFrame(
            np.zeros((1, len(feature_cols))),
            columns=feature_cols
        )

        input_df["area"] = area
        input_df["bathrooms"] = bathrooms
        input_df["beds"] = bedrooms
        input_df["bath_per_bed"] = bathrooms / (bedrooms + 1)
        input_df["room_density"] = area / (bedrooms + 1)
        input_df["bed_bath_ratio"] = bedrooms / (bathrooms + 1)
        input_df["area_per_room"] = area / (bedrooms + bathrooms + 1)

        input_df["locality_target"] = df[df["city"] == city]["locality_target"].mean()

        for col in feature_cols:
            if col == f"city_{city}":
                input_df[col] = 1
            elif col == f"furnishing_{furnishing}":
                input_df[col] = 1

        prediction = np.expm1(model.predict(input_df)[0])

        st.success(f"Estimated Rent: ₹{int(prediction)}")
        st.info(f"Expected Range: ₹{int(prediction*0.9)} - ₹{int(prediction*1.1)}")
