import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.title("Advanced Heart Disease Dashboard")
st.markdown("Explore data, visualize KPIs, and predict heart disease.")

@st.cache_data
def load_data():
    df = pd.read_csv("data/heart_disease_uci.csv")
    df.loc[df["chol"] == 0, "chol"] = np.nan
    df.loc[df["trestbps"] == 0, "trestbps"] = np.nan
    numeric_cols = ["trestbps", "chol", "thalch", "oldpeak", "ca"]
    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)
    categorical_cols = ["slope", "restecg", "thal", "fbs", "exang"]
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
    df["ca"] = df["ca"].astype("int64")
    cat_cols = ["sex", "dataset", "cp", "restecg", "slope", "thal"]
    for col in cat_cols:
        df[col] = df[col].astype("category")
    df["has_disease"] = (df["num"] > 0).astype(int)
    df["age_group"] = pd.cut(df["age"], bins=[20, 40, 50, 60, 80], labels=["20-40", "41-50", "51-60", "61-80"])
    return df

df = load_data()

var_labels = {"age": "Age", "trestbps": "Blood Pressure", "chol": "Cholesterol", "thalch": "Max Heart Rate",
              "oldpeak": "ST Depression", "ca": "Vessels", "num": "Diagnosis", "sex": "Sex", "cp": "Chest Pain",
              "dataset": "Center", "age_group": "Age Group", "has_disease": "Heart Disease"}

numeric_vars = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
categorical_vars = ["sex", "cp", "dataset", "age_group", "has_disease"]

features = ["age", "sex", "cp", "trestbps", "chol", "thalch", "oldpeak", "ca"]
X = df[features].copy()
y = df["has_disease"]
label_encoders = {}
for col in ["sex", "cp"]:
    label_encoders[col] = LabelEncoder()
    X[col] = label_encoders[col].fit_transform(X[col])
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

st.sidebar.header("Filters and Options")
with st.sidebar.expander("How to use"):
    st.markdown("- **Filters** : Select subgroups (sex, center, etc.).")
    st.markdown("- **Chart** : Choose a type and a variable for analysis.")
    st.markdown("- **KPIs** : View prevalences and enable the table.")
    st.markdown("- **Prediction** : Enter patient data to estimate risk.")

sex_filter = st.sidebar.multiselect("Sex", options=df["sex"].cat.categories, default=df["sex"].cat.categories)
dataset_filter = st.sidebar.multiselect("Center", options=df["dataset"].cat.categories, default=df["dataset"].cat.categories)
age_group_filter = st.sidebar.multiselect("Age Group", options=df["age_group"].cat.categories, default=df["age_group"].cat.categories)
cp_filter = st.sidebar.multiselect("Chest Pain", options=df["cp"].cat.categories, default=df["cp"].cat.categories)

filtered_df = df[df["sex"].isin(sex_filter) & df["dataset"].isin(dataset_filter) & 
                df["age_group"].isin(age_group_filter) & df["cp"].isin(cp_filter)]

if filtered_df.empty:
    st.error("No data matches the selected filters. Please adjust the filters.")
else:
    chart_type = st.sidebar.selectbox("Chart Type", ["Histogram", "Boxplot", "Violin Plot", "Scatter Plot", "Bar Plot", "Pie Chart", "Sunburst", "Heatmap"])
    variable = st.sidebar.selectbox("Main Variable", options=numeric_vars + categorical_vars, format_func=lambda x: var_labels[x])
    secondary_variable = st.sidebar.selectbox("Secondary Variable (Scatter)", options=numeric_vars, format_func=lambda x: var_labels[x], index=1) if chart_type == "Scatter Plot" else None
    color_by = st.sidebar.selectbox("Color by", options=categorical_vars, format_func=lambda x: var_labels[x])
    group_by = st.sidebar.selectbox("Group by (Bar/Pie/Sunburst)", options=categorical_vars, format_func=lambda x: var_labels[x]) if chart_type in ["Bar Plot", "Pie Chart", "Sunburst"] else None
    show_stats = st.sidebar.checkbox("Descriptive Statistics", value=True)
    show_kpi_table = st.sidebar.checkbox("KPI Table", value=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Prevalence", f"{filtered_df["has_disease"].mean() * 100:.2f}%", 
                  delta=f"{(filtered_df["has_disease"].mean() - df["has_disease"].mean()) * 100:.1f}%")
    with col2:
        st.metric("Average Age", f"{filtered_df["age"].mean():.1f} years", 
                  delta=f"{filtered_df["age"].mean() - df["age"].mean():.1f} years")
    with col3:
        st.metric("Average Cholesterol", f"{filtered_df["chol"].mean():.1f} mg/dl", 
                  delta=f"{filtered_df["chol"].mean() - df["chol"].mean():.1f} mg/dl")
    with col4:
        st.metric("Patients", len(filtered_df))

    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Analysis", "Key Indicators", "Prediction"])

    with tab1:
        st.header("Data Overview")
        if show_stats:
            st.subheader("Descriptive Statistics")
            stats_df = filtered_df[numeric_vars].describe().round(2)
            stats_df.index = ["Count", "Mean", "Std", "Min", "Q1", "Median", "Q3", "Max"]
            stats_df.columns = [var_labels[col] for col in stats_df.columns]
            st.dataframe(stats_df, use_container_width=True)

        st.subheader("Diagnosis Distribution")
        fig = px.pie(values=filtered_df["num"].value_counts().values, names=filtered_df["num"].value_counts().index,
                     title="Diagnosis Distribution", color_discrete_sequence=px.colors.sequential.Viridis)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("Detailed Analysis")
        
        if chart_type == "Histogram" and variable in numeric_vars:
            st.subheader(f"Distribution of {var_labels[variable]}")
            fig, ax = plt.subplots()
            sns.histplot(filtered_df[variable], kde=True, ax=ax, color="dodgerblue")
            ax.set_xlabel(var_labels[variable])
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
        
        elif chart_type == "Boxplot" and variable in numeric_vars:
            st.subheader(f"Boxplot of {var_labels[variable]}")
            fig, ax = plt.subplots()
            sns.boxplot(y=filtered_df[variable], x=filtered_df[color_by], ax=ax, palette="Set2")
            ax.set_xlabel(var_labels[color_by])
            ax.set_ylabel(var_labels[variable])
            st.pyplot(fig)
        
        elif chart_type == "Violin Plot" and variable in numeric_vars:
            st.subheader(f"Violin Plot of {var_labels[variable]}")
            fig, ax = plt.subplots()
            sns.violinplot(y=filtered_df[variable], x=filtered_df[color_by], ax=ax, palette="Pastel1")
            ax.set_xlabel(var_labels[color_by])
            ax.set_ylabel(var_labels[variable])
            st.pyplot(fig)
        
        elif chart_type == "Scatter Plot" and variable in numeric_vars and secondary_variable:
            st.subheader(f"{var_labels[variable]} vs {var_labels[secondary_variable]}")
            fig = px.scatter(filtered_df, x=variable, y=secondary_variable, color=color_by,
                             hover_data=["sex", "cp", "dataset", "age_group"],
                             title=f"{var_labels[variable]} vs {var_labels[secondary_variable]}",
                             color_discrete_sequence=px.colors.qualitative.Set1)
            st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "Bar Plot" and group_by:
            st.subheader(f"Distribution by {var_labels[group_by]}")
            counts = filtered_df[group_by].value_counts()
            fig = px.bar(x=counts.index, y=counts.values, color=counts.index,
                         labels={"x": var_labels[group_by], "y": "Count"},
                         title=f"Distribution by {var_labels[group_by]}",
                         color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "Pie Chart" and group_by:
            st.subheader(f"Distribution by {var_labels[group_by]}")
            fig = px.pie(values=filtered_df[group_by].value_counts().values,
                         names=filtered_df[group_by].value_counts().index,
                         title=f"Distribution by {var_labels[group_by]}",
                         color_discrete_sequence=px.colors.sequential.Plasma)
            st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "Heatmap":
            st.subheader("Correlation Matrix")
            corr = filtered_df[numeric_vars + ["has_disease"]].corr()
            fig = go.Figure(data=go.Heatmap(z=corr.values, x=[var_labels[col] for col in corr.columns],
                                            y=[var_labels[col] for col in corr.columns],
                                            colorscale="RdBu", zmin=-1, zmax=1, text=corr.values.round(2),
                                            texttemplate="%{text}"))
            fig.update_layout(title="Correlations between Variables")
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.warning("Please select a valid combination (e.g., numeric variable for Histogram/Boxplot).")

    with tab3:
        st.header("Key Indicators (KPIs)")
        
        prevalence = filtered_df["has_disease"].mean() * 100
        st.markdown(f"**Global Prevalence** : {prevalence:.2f}%")
        
        if group_by:
            st.subheader(f"Prevalence by {var_labels[group_by]}")
            prevalence_group = filtered_df.groupby(group_by)["has_disease"].mean() * 100
            fig = px.bar(x=prevalence_group.index, y=prevalence_group.values, color=prevalence_group.index,
                         labels={"x": var_labels[group_by], "y": "Proportion (%)"},
                         title=f"Prevalence by {var_labels[group_by]}",
                         color_discrete_sequence=px.colors.qualitative.Bold)
            st.plotly_chart(fig, use_container_width=True)
        
        if show_kpi_table:
            st.subheader("Key Indicators Table")
            prevalence_sex = filtered_df.groupby("sex")["has_disease"].mean() * 100
            prevalence_age = filtered_df.groupby("age_group")["has_disease"].mean() * 100
            prevalence_cp = filtered_df.groupby("cp")["has_disease"].mean() * 100
            prevalence_dataset = filtered_df.groupby("dataset")["has_disease"].mean() * 100
            kpi_summary = pd.DataFrame({
                "Global Prevalence (%)": [prevalence],
                "Men (%)": [prevalence_sex.get("Male", np.nan)],
                "Women (%)": [prevalence_sex.get("Female", np.nan)],
                "Age 20-40 (%)": [prevalence_age.get("20-40", np.nan)],
                "Age 41-50 (%)": [prevalence_age.get("41-50", np.nan)],
                "Age 51-60 (%)": [prevalence_age.get("51-60", np.nan)],
                "Age 61-80 (%)": [prevalence_age.get("61-80", np.nan)],
                "Asymptomatic Pain (%)": [prevalence_cp.get("asymptomatic", np.nan)],
                "Cleveland (%)": [prevalence_dataset.get("Cleveland", np.nan)],
                "Switzerland (%)": [prevalence_dataset.get("Switzerland", np.nan)]
            }).round(2)
            st.dataframe(kpi_summary, use_container_width=True)

    with tab4:
        st.header("Heart Disease Prediction")
        with st.form("prediction_form"):
            st.markdown("**Enter patient data**")
            age = st.slider("Age", 20, 80, 50)
            sex = st.selectbox("Sex", ["Male", "Female"])
            cp = st.selectbox("Chest Pain", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
            trestbps = st.number_input("Blood Pressure (mm Hg)", 80.0, 200.0, 120.0)
            chol = st.number_input("Cholesterol (mg/dl)", 100.0, 600.0, 200.0)
            thalch = st.number_input("Max Heart Rate (bpm)", 60.0, 220.0, 140.0)
            oldpeak = st.number_input("ST Depression (mm)", -2.0, 6.0, 0.0)
            ca = st.slider("Number of vessels (0-3)", 0, 3, 0)
            submitted = st.form_submit_button("Predict")

            if submitted:
                input_data = pd.DataFrame({
                    "age": [age],
                    "sex": [sex],
                    "cp": [cp],
                    "trestbps": [trestbps],
                    "chol": [chol],
                    "thalch": [thalch],
                    "oldpeak": [oldpeak],
                    "ca": [ca]
                })
                for col in ["sex", "cp"]:
                    input_data[col] = label_encoders[col].transform(input_data[col])
                prob = model.predict_proba(input_data)[0][1] * 100
                pred = model.predict(input_data)[0]
                st.success(f"**Probability of heart disease** : {prob:.2f}%")
                st.markdown(f"**Predicted diagnosis** : {"Heart Disease" if pred == 1 else "No Disease"}")


