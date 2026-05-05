import streamlit as stimport pandas as pdimport numpy as npimport osimport plotly.express as pximport plotly.graph_objects as go

from sklearn.ensemble import RandomForestRegressorfrom sklearn.linear_model import LinearRegressionfrom sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import train_test_split, cross_val_scorefrom sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

======================

PAGE CONFIG

======================

st.set_page_config(page_title="🏠 House Rent Dashboard", layout="wide")

======================

CSS (FIXED RESPONSIVE)

======================

st.markdown("""

""", unsafe_allow_html=True)

======================

LOAD DATA (CACHED)

======================

file_path = os.path.join(os.path.dirname(file), "data.csv")

@st.cache_datadef load_data():return pd.read_csv(file_path)

df = load_data()

======================

CLEANING

======================

df.fillna(df.median(numeric_only=True), inplace=True)

lower = df["rent"].quantile(0.01)upper = df["rent"].quantile(0.99)df["rent"] = df["rent"].clip(lower, upper)

======================

FEATURE ENGINEERING

======================

df["bath_per_bed"] = df["bathrooms"] / (df["beds"] + 1)df["room_density"] = df["area"] / (df["beds"] + 1)df["bed_bath_ratio"] = df["beds"] / (df["bathrooms"] + 1)df["area_per_room"] = df["area"] / (df["beds"] + df["bathrooms"] + 1)

locality_mean = df.groupby("locality")["rent"].mean()df["locality_target"] = df["locality"].map(locality_mean)

======================

MODEL

======================

@st.cache_resourcedef train_models(data):df_ml = data.drop(columns=["house_type", "area_rate", "locality"])df_ml = pd.get_dummies(df_ml, drop_first=True)

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
        min_samples_split=4,
        min_samples_leaf=1,
        max_features="sqrt",
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

# ✅ BEST MODEL SELECTION
best_model_name = max(results, key=lambda x: results[x]["r2"])
best_model = results[best_model_name]["model"]

cv_score = cross_val_score(best_model, X_train, y_train, cv=5, scoring="r2").mean()

return best_model, best_model_name, results, X.columns, X_test, y_test, cv_score

with st.spinner("Training model..."):model, best_model_name, results, feature_cols, X_test, y_test, cv_score = train_models(df)

======================

HEADER

======================

st.markdown("""

KPI CARDS

col1, col2, col3 = st.columns([1,1,1], gap="large")

with col1:st.markdown(f"Total Listings{len(df)}", unsafe_allow_html=True)

with col2:st.markdown(f"Cities{df['city'].nunique()}", unsafe_allow_html=True)

with col3:st.markdown(f"Average Rent₹{int(df['rent'].mean())}", unsafe_allow_html=True)

st.markdown("---")

======================

SIDEBAR

======================

st.sidebar.subheader("📂 Navigation")menu = st.sidebar.radio("", ["📊 EDA", "🤖 Model", "🏠 Prediction"])

======================

EDA

======================

if menu == "📊 EDA":st.markdown("## 📊 Exploratory Data Analysis")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Rent Distribution")
    fig = px.histogram(df, x="rent", color_discrete_sequence=["#00FFAA"])
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("#### Average Rent by City")
    city_avg = df.groupby("city")["rent"].mean().reset_index()
    fig = px.bar(city_avg, x="city", y="rent", color_discrete_sequence=["#00FFAA"])
    st.plotly_chart(fig, use_container_width=True)

======================

MODEL

======================

elif menu == "🤖 Model":st.markdown("## 🤖 Model Performance")

result_df = pd.DataFrame(results).T[["r2", "rmse", "mae"]]
result_df.columns = ["R² Score", "RMSE", "MAE"]

st.dataframe(result_df)

st.success(f"✔ Final Model: {best_model_name}")
st.info(f"Cross Validation Score (R²): {cv_score:.2f}")

# 📊 Comparison Chart
fig = px.bar(result_df, barmode="group")
st.plotly_chart(fig)

# 📉 Actual vs Predicted
y_pred = model.predict(X_test)
actual = np.expm1(y_test)
predicted = np.expm1(y_pred)

fig = go.Figure()
fig.add_trace(go.Scatter(x=actual, y=predicted, mode='markers', name="Predicted"))
fig.add_trace(go.Scatter(x=actual, y=actual, mode='lines', name="Perfect Fit"))

fig.update_layout(title="Actual vs Predicted Rent")
st.plotly_chart(fig)

# ⭐ Feature Importance
if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
    feat_df = pd.DataFrame({"Feature": feature_cols, "Importance": importances})
    feat_df = feat_df.sort_values(by="Importance", ascending=False).head(10)

    fig = px.bar(feat_df, x="Importance", y="Feature", orientation='h')
    st.plotly_chart(fig)

======================

PREDICTION

======================

elif menu == "🏠 Prediction":st.markdown("## 🏠 Predict House Rent")

col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Area (sqft)", 300, 5000, 1000)
    city = st.selectbox("City", df["city"].unique())

with col2:
    bathrooms = st.slider("Bathrooms", 1, 5, 2)
    bedrooms = st.slider("Bedrooms", 1, 5, 2)

furnishing = st.selectbox("Furnishing", df["furnishing"].unique())

st.markdown("---")

if st.button("Predict Rent"):

    input_df = pd.DataFrame(np.zeros((1, len(feature_cols))), columns=feature_cols)

    input_df["area"] = area
    input_df["bathrooms"] = bathrooms
    input_df["beds"] = bedrooms
    input_df["bath_per_bed"] = bathrooms / (bedrooms + 1)
    input_df["room_density"] = area / (bedrooms + 1)
    input_df["bed_bath_ratio"] = bedrooms / (bathrooms + 1)
    input_df["area_per_room"] = area / (bedrooms + bathrooms + 1)

    # ✅ safer locality encoding
    input_df["locality_target"] = df["locality_target"].mean()

    for col in feature_cols:
        if col == f"city_{city}":
            input_df[col] = 1
        elif col == f"furnishing_{furnishing}":
            input_df[col] = 1

    prediction = np.expm1(model.predict(input_df)[0])

    st.markdown(f"""
    <div style="display:flex;justify-content:center;">
    <div style="background-color:#1C1F26;padding:25px;border-radius:12px;width:300px;text-align:center;">
    <h2 style="color:#00FFAA;">₹{int(prediction)}</h2>
    <p>Estimated Monthly Rent</p>
    </div>
    </div>
    """, unsafe_allow_html=True)

    st.caption("Prediction based on historical data")

======================

FOOTER

======================

st.markdown("---")st.markdown("PT-2 Project | House Rent Prediction", unsafe_allow_html=True)
