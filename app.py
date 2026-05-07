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

# LOAD DATA
@st.cache_data
def load_data():
    return pd.read_csv("data.csv")

df = load_data()

# 🔥 KEEP ORIGINAL DATA
df_raw = df.copy()

# =========================
# ✅ MISSING VALUE HANDLING (FIXED)
# =========================

# CATEGORICAL (MODE)
for col in ["city", "furnishing"]:
    df[col].fillna(df[col].mode()[0], inplace=True)

# NUMERICAL (MEDIAN)
df.fillna(df.median(numeric_only=True), inplace=True)

# =========================
# OUTLIER HANDLING
# =========================
df["rent"] = df["rent"].clip(
    df["rent"].quantile(0.01),
    df["rent"].quantile(0.99)
)

# =========================
# FEATURE ENGINEERING
# =========================
df["bath_per_bed"] = df["bathrooms"] / (df["beds"] + 1)
df["room_density"] = df["area"] / (df["beds"] + 1)
df["bed_bath_ratio"] = df["beds"] / (df["bathrooms"] + 1)
df["area_per_room"] = df["area"] / (df["beds"] + df["bathrooms"] + 1)
df["locality_target"] = df.groupby("locality")["rent"].transform("mean")

# =========================
# MODEL TRAINING
# =========================
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
            n_estimators=300,
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

# =========================
# UI
# =========================
st.title("Indian House Rent Prediction")

menu = st.sidebar.radio(
    "Menu",
    ["EDA", "Model Evaluation", "Prediction"]
)

# =========================
# EDA
# =========================
if menu == "EDA":

    st.subheader("Raw Data (Before Preprocessing)")
    st.dataframe(df_raw.head())

    st.subheader("Processed Data (After Preprocessing)")
    st.dataframe(df.head())

    st.subheader("Missing Values Before")
    st.write(df_raw.isnull().sum())

    st.subheader("Missing Values After")
    st.write(df.isnull().sum())

    st.write("Shape Before:", df_raw.shape)
    st.write("Shape After:", df.shape)

# =========================
# MODEL
# =========================
elif menu == "Model Evaluation":

    result_df = pd.DataFrame(results).T[["r2", "rmse", "mae"]]
    result_df.columns = ["R²", "RMSE", "MAE"]

    st.dataframe(result_df)

    st.success("Best Model: Random Forest")
    st.info(f"Cross Validation Score: {cv_score:.2f}")

    st.plotly_chart(px.bar(result_df.reset_index(), x="index", y="R²"))

    # Actual vs Predicted
    y_pred = model.predict(X_test)
    actual = np.expm1(y_test)
    predicted = np.expm1(y_pred)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual, y=predicted, mode='markers'))
    fig.add_trace(go.Scatter(x=actual, y=actual, mode='lines'))
    st.plotly_chart(fig)

# =========================
# PREDICTION
# =========================
elif menu == "Prediction":

    area = st.number_input("Area", 300, 5000, 1000)
    bathrooms = st.slider("Bathrooms", 1, 5, 2)
    bedrooms = st.slider("Bedrooms", 1, 5, 2)
    city = st.selectbox("City", df["city"].unique())
    furnishing = st.selectbox("Furnishing", df["furnishing"].unique())

    if st.button("Predict"):

        input_df = pd.DataFrame(np.zeros((1, len(feature_cols))), columns=feature_cols)

        input_df["area"] = area
        input_df["bathrooms"] = bathrooms
        input_df["beds"] = bedrooms
        input_df["bath_per_bed"] = bathrooms / (bedrooms + 1)
        input_df["room_density"] = area / (bedrooms + 1)
        input_df["bed_bath_ratio"] = bedrooms / (bathrooms + 1)
        input_df["area_per_room"] = area / (bedrooms + bathrooms + 1)
        input_df["locality_target"] = df["locality_target"].mean()

        for col in feature_cols:
            if col == f"city_{city}":
                input_df[col] = 1
            elif col == f"furnishing_{furnishing}":
                input_df[col] = 1

        prediction = np.expm1(model.predict(input_df)[0])

        st.success(f"Estimated Rent: ₹{int(prediction)}")
