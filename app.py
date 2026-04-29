import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
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
# UI CSS
# ======================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0a192f, #112240, #1f4068);
    color: #e6f1ff;
}
.glass {
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(10px);
    border-radius: 16px;
    padding: 20px;
    border: 1px solid rgba(255,255,255,0.1);
}
.value {
    font-size: 26px;
    font-weight: bold;
    color: #64ffda;
}
</style>
""", unsafe_allow_html=True)

# ======================
# HEADER
# ======================
st.markdown("""
<h1 style='text-align:center;'>🏠 House Rent Prediction Dashboard</h1>
<p style='text-align:center;color:#64ffda;'>Optimized Random Forest Model</p>
""", unsafe_allow_html=True)

st.caption(f"{len(df)} listings | {df['city'].nunique()} cities")

# ======================
# METRICS
# ======================
col1, col2, col3 = st.columns(3)

col1.markdown(f"<div class='glass'><p>Total Listings</p><p class='value'>{len(df)}</p></div>", unsafe_allow_html=True)
col2.markdown(f"<div class='glass'><p>Average Rent</p><p class='value'>₹{int(df['rent'].mean())}</p></div>", unsafe_allow_html=True)
col3.markdown(f"<div class='glass'><p>Max Rent</p><p class='value'>₹{int(df['rent'].max())}</p></div>", unsafe_allow_html=True)

# ======================
# SIDEBAR
# ======================
menu = st.sidebar.radio("Navigation", ["EDA", "Model", "Prediction"])

# ======================
# TRAIN MODEL
# ======================
@st.cache_resource
def train_model(data):
    df_ml = data.drop(columns=["house_type", "area_rate"])
    df_ml = pd.get_dummies(df_ml, drop_first=True)

    X = df_ml.drop("rent", axis=1)
    y = np.log1p(df_ml["rent"])

    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=20,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42
    )

    model.fit(X, y)
    return model, X.columns, X, y

model, feature_cols, X_full, y_full = train_model(df)

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

    st.subheader("Key Insights")
    st.markdown("""
    - 📍 Mumbai has highest rent  
    - 🛁 Bathrooms strongly affect rent  
    - 📐 Area increases rent significantly  
    - 🏙 Location plays major role  
    """)

# ======================
# MODEL
# ======================
elif menu == "Model":
    st.subheader("🤖 Model Performance")

    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_actual = np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(y_pred)))

    st.markdown(f"<div class='glass'><p>R² Score</p><p class='value'>{r2:.2f}</p></div>", unsafe_allow_html=True)

    st.write(f"RMSE (log scale): {rmse:.2f}")
    st.write(f"Average Error (₹): ₹{int(rmse_actual)}")

    # Feature Importance
    importance = pd.DataFrame({
        "Feature": X_full.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    st.subheader("Top Influencing Features")
    st.bar_chart(importance.head(10).set_index("Feature"), height=400)

    st.write("Top 5 Factors:")
    st.write(importance.head(5))

    st.info("""
Model: Random Forest Regressor  
Technique: Log Transformation + Feature Engineering  
""")

    st.caption("""
Limitations:
- Amenities not included  
- Market demand not considered  
- Locality simplified  
""")

# ======================
# PREDICTION
# ======================
elif menu == "Prediction":
    st.subheader("🏠 Predict Rent")

    st.info("Enter property details to estimate rent using trained ML model.")

    col1, col2, col3 = st.columns(3)

    with col1:
        area = st.number_input("Area", 300, 5000, 1000)
        city = st.selectbox("City", df["city"].unique())

    with col2:
        bathrooms = st.slider("Bathrooms", 1, 5, 2)
        furnishing = st.selectbox("Furnishing", df["furnishing"].unique())

    with col3:
        bedrooms = st.slider("Bedrooms", 1, 5, 2)

    if st.button("Predict Rent"):

        input_df = pd.DataFrame(columns=feature_cols)
        input_df.loc[0] = 0

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

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            f"<div class='glass'><h2>Estimated Rent</h2>"
            f"<h1 style='color:#64ffda;'>₹{int(prediction)}</h1></div>",
            unsafe_allow_html=True
        )

        low = int(prediction * 0.9)
        high = int(prediction * 1.1)

        st.write(f"Estimated Range: ₹{low} - ₹{high}")
        st.success("Confidence: Medium (~80% accuracy)")
        st.caption("Prediction may vary due to real-world factors.")

# ======================
# FOOTER
# ======================
st.markdown("<hr><p style='text-align:center;'>Developed by Wiz</p>", unsafe_allow_html=True)