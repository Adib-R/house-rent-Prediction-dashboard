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

st.set_page_config(
    page_title="Indian House Rent Prediction Dashboard",
    layout="wide"
)

# =========================
# THEME TOGGLE
# =========================

theme = st.sidebar.selectbox(
    "Select Theme",
    ["Dark", "Light"]
)

if theme == "Dark":

    bg_color = "#0E1117"
    card_color = "#1C1F26"
    text_color = "#FFFFFF"
    accent_color = "#00FFAA"
    sidebar_color = "#161A23"

else:

    bg_color = "#F5F7FA"
    card_color = "#FFFFFF"
    text_color = "#111111"
    accent_color = "#0077CC"
    sidebar_color = "#FFFFFF"

# =========================
# CUSTOM CSS
# =========================

st.markdown(f"""
<style>

.stApp {{
    background-color: {bg_color};
    color: {text_color};
}}

.block-container {{
    max-width: 1100px;
    margin: auto;
    padding-top: 1.5rem;
}}

.card {{
    background-color: {card_color};
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    color: {text_color};
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
}}

h1, h2, h3, h4, h5, h6, p {{
    color: {text_color};
}}

[data-testid="stSidebar"] {{
    background-color: {sidebar_color};
}}

[data-testid="stSidebar"] * {{
    color: {text_color};
}}

.stButton > button {{
    width: 100%;
    border-radius: 10px;
    background-color: {accent_color};
    color: black;
    font-weight: bold;
    border: none;
    padding: 10px;
}}

.stButton > button:hover {{
    opacity: 0.9;
}}

@media (max-width: 768px) {{
    .block-container {{
        padding: 1rem;
    }}
}}

</style>
""", unsafe_allow_html=True)

# =========================
# LOAD DATA
# =========================

@st.cache_data
def load_data():
    return pd.read_csv("data.csv")

df = load_data()

# =========================
# DATA CLEANING
# =========================

df.fillna(df.median(numeric_only=True), inplace=True)

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
df["area_per_room"] = df["area"] / (
    df["beds"] + df["bathrooms"] + 1
)

df["locality_target"] = (
    df.groupby("locality")["rent"]
    .transform("mean")
)

# =========================
# MODEL TRAINING
# =========================

@st.cache_resource
def train_models(data):

    df_ml = data.drop(
        columns=["house_type", "area_rate", "locality"]
    )

    df_ml = pd.get_dummies(df_ml, drop_first=True)

    X = df_ml.drop("rent", axis=1)
    y = np.log1p(df_ml["rent"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    models = {

        "Linear Regression": LinearRegression(),

        "Decision Tree": DecisionTreeRegressor(
            max_depth=12,
            random_state=42
        ),

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

            "rmse": np.sqrt(
                mean_squared_error(actual, predicted)
            ),

            "mae": mean_absolute_error(actual, predicted)
        }

    best_model = results["Random Forest"]["model"]

    cv_score = cross_val_score(
        best_model,
        X_train,
        y_train,
        cv=5,
        scoring="r2"
    ).mean()

    return (
        best_model,
        results,
        X.columns,
        X_test,
        y_test,
        cv_score
    )

model, results, feature_cols, X_test, y_test, cv_score = train_models(df)

# =========================
# HEADER
# =========================

st.markdown(f"""
<h1 style='text-align:center;color:{text_color};'>
Indian House Rent Prediction System
</h1>

<h3 style='text-align:center;color:{accent_color};'>
Machine Learning-Based Rental Price Estimation
</h3>
""", unsafe_allow_html=True)

# =========================
# KPI SECTION
# =========================

col1, col2, col3 = st.columns(3)

col1.markdown(
    f"""
    <div class='card'>
    <h4>Total Property Listings</h4>
    <h2>{len(df)}</h2>
    </div>
    """,
    unsafe_allow_html=True
)

col2.markdown(
    f"""
    <div class='card'>
    <h4>Number of Cities</h4>
    <h2>{df['city'].nunique()}</h2>
    </div>
    """,
    unsafe_allow_html=True
)

col3.markdown(
    f"""
    <div class='card'>
    <h4>Average Rental Price</h4>
    <h2>₹{int(df['rent'].mean())}</h2>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# =========================
# SIDEBAR NAVIGATION
# =========================

menu = st.sidebar.radio(
    "Navigation Menu",
    [
        "Exploratory Data Analysis",
        "Model Evaluation",
        "Rent Prediction"
    ]
)

# =========================
# EDA SECTION
# =========================

if menu == "Exploratory Data Analysis":

    st.markdown("## Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### Distribution of Rental Prices")

        fig = px.histogram(df, x="rent")

        fig.update_layout(
            paper_bgcolor=bg_color,
            plot_bgcolor=card_color,
            font_color=text_color
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        st.markdown("### Average Rent by City")

        city_avg = (
            df.groupby("city")["rent"]
            .mean()
            .reset_index()
        )

        fig = px.bar(city_avg, x="city", y="rent")

        fig.update_layout(
            paper_bgcolor=bg_color,
            plot_bgcolor=card_color,
            font_color=text_color
        )

        st.plotly_chart(fig, use_container_width=True)

# =========================
# MODEL EVALUATION
# =========================

elif menu == "Model Evaluation":

    st.markdown("## Model Performance Analysis")

    result_df = pd.DataFrame(results).T[
        ["r2", "rmse", "mae"]
    ]

    result_df.columns = [
        "R² Score",
        "Root Mean Squared Error (RMSE)",
        "Mean Absolute Error (MAE)"
    ]

    st.dataframe(result_df, use_container_width=True)

    st.success("Final Selected Model: Random Forest Regressor")

    st.info(f"Cross-Validation Score (R²): {cv_score:.2f}")

    # R2 GRAPH

    st.markdown("### R² Score Comparison Across Models")

    r2_df = result_df["R² Score"].reset_index()

    r2_df.columns = ["Model", "R² Score"]

    fig = px.bar(
        r2_df,
        x="Model",
        y="R² Score",
        color="Model",
        text_auto=True
    )

    fig.update_layout(
        yaxis_range=[0, 1],
        paper_bgcolor=bg_color,
        plot_bgcolor=card_color,
        font_color=text_color
    )

    st.plotly_chart(fig, use_container_width=True)

    # ERROR GRAPH

    st.markdown("### Error Metrics Comparison")

    error_df = result_df[
        [
            "Root Mean Squared Error (RMSE)",
            "Mean Absolute Error (MAE)"
        ]
    ].reset_index()

    error_df = error_df.melt(
        id_vars="index",
        var_name="Metric",
        value_name="Value"
    )

    error_df.rename(
        columns={"index": "Model"},
        inplace=True
    )

    fig = px.bar(
        error_df,
        x="Model",
        y="Value",
        color="Metric",
        barmode="group"
    )

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=card_color,
        font_color=text_color
    )

    st.plotly_chart(fig, use_container_width=True)

    # ACTUAL VS PREDICTED

    st.markdown("### Actual vs Predicted Rental Prices")

    y_pred = model.predict(X_test)

    actual = np.expm1(y_test)
    predicted = np.expm1(y_pred)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=actual,
            y=predicted,
            mode='markers',
            name="Predicted Values"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=actual,
            y=actual,
            mode='lines',
            name="Ideal Prediction Line"
        )
    )

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=card_color,
        font_color=text_color
    )

    st.plotly_chart(fig, use_container_width=True)

    # FEATURE IMPORTANCE

    if hasattr(model, "feature_importances_"):

        st.markdown("### Feature Importance Analysis")

        feat_df = pd.DataFrame({

            "Feature": feature_cols,

            "Importance": model.feature_importances_

        }).sort_values(
            by="Importance",
            ascending=False
        ).head(10)

        fig = px.bar(
            feat_df,
            x="Importance",
            y="Feature",
            orientation='h'
        )

        fig.update_layout(
            paper_bgcolor=bg_color,
            plot_bgcolor=card_color,
            font_color=text_color
        )

        st.plotly_chart(fig, use_container_width=True)

# =========================
# RENT PREDICTION
# =========================

elif menu == "Rent Prediction":

    st.markdown("## Predict Monthly House Rent")

    col1, col2 = st.columns(2)

    with col1:

        area = st.number_input(
            "Property Area (sq. ft.)",
            300,
            5000,
            1000
        )

        city = st.selectbox(
            "Select City",
            df["city"].unique()
        )

    with col2:

        bathrooms = st.slider(
            "Number of Bathrooms",
            1,
            5,
            2
        )

        bedrooms = st.slider(
            "Number of Bedrooms",
            1,
            5,
            2
        )

    furnishing = st.selectbox(
        "Furnishing Status",
        df["furnishing"].unique()
    )

    if st.button("Estimate Rent"):

        input_df = pd.DataFrame(
            np.zeros((1, len(feature_cols))),
            columns=feature_cols
        )

        input_df["area"] = area
        input_df["bathrooms"] = bathrooms
        input_df["beds"] = bedrooms

        input_df["bath_per_bed"] = (
            bathrooms / (bedrooms + 1)
        )

        input_df["room_density"] = (
            area / (bedrooms + 1)
        )

        input_df["bed_bath_ratio"] = (
            bedrooms / (bathrooms + 1)
        )

        input_df["area_per_room"] = (
            area / (bedrooms + bathrooms + 1)
        )

        input_df["locality_target"] = (
            df["locality_target"].mean()
        )

        for col in feature_cols:

            if col == f"city_{city}":
                input_df[col] = 1

            elif col == f"furnishing_{furnishing}":
                input_df[col] = 1

        prediction = np.expm1(
            model.predict(input_df)[0]
        )

        st.markdown(f"""
        <div style="display:flex;justify-content:center;">
        <div style="
        background:{card_color};
        padding:25px;
        border-radius:12px;
        width:300px;
        text-align:center;
        box-shadow:0 2px 10px rgba(0,0,0,0.1);
        ">
        <h2 style="color:{accent_color};">
        ₹{int(prediction)}
        </h2>

        <p style="color:{text_color};">
        Estimated Monthly Rent
        </p>

        </div>
        </div>
        """, unsafe_allow_html=True)

# =========================
# FOOTER
# =========================

st.markdown("---")

st.markdown(
    f"""
    <p style='text-align:center;color:gray;'>
    Academic Project (PT-2) |
    Indian House Rent Prediction System
    </p>
    """,
    unsafe_allow_html=True
)
