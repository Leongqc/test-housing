import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load cleaned data
data_cleaned = pd.read_csv('cleaned_data.csv')

# Sidebar menu
st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to", ["Introduction", "Exploratory Data Analysis", "Prediction"])

# Introduction section
if option == "Introduction":
    st.title("Introduction")
    st.write("""
    In today's rapidly evolving job market, students often face uncertainty regarding industry demands, job roles, salary expectations, and the impact of their educational background on their future careers. This dashboard aims to address these concerns by analyzing salary data based on various factors such as job roles, gender, age, years of experience, and educational background. Through this analysis, students can gain insights into industry trends and make informed decisions about their career paths.

    The dataset used in this analysis includes information on age, gender, education level, job title, years of experience, and salary. The data has been cleaned and categorized to provide a comprehensive view of the job market. Additionally, a machine learning model has been trained to predict salaries based on these factors, helping students to better understand potential salary outcomes in their chosen fields.
    """)

# Exploratory Data Analysis (EDA) section
elif option == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis")

    st.header("Salary Distribution")
    plt.figure(figsize=(10, 6))
    sns.histplot(data_cleaned['Salary'], kde=True)
    st.pyplot(plt)

    st.header("Gender Distribution")
    gender_counts = data_cleaned['Gender'].value_counts()
    plt.figure(figsize=(6, 6))
    plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title('Gender Distribution')
    st.pyplot(plt)

    st.header("Salary vs Age")
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Age', y='Salary', data=data_cleaned)
    st.pyplot(plt)

    st.header("Job Roles vs Gender")
    top_job_roles = data_cleaned['Job Title'].value_counts().nlargest(10)
    plt.figure(figsize=(14, 8))
    sns.countplot(y='Job Title', hue='Gender', data=data_cleaned, order=top_job_roles.index)
    plt.title('Job Roles vs Gender')
    st.pyplot(plt)

    st.header("Average Salary by Job Role")
    mean_salary_by_job = data_cleaned.groupby('Job Title')['Salary'].mean().nlargest(10)
    plt.figure(figsize=(14, 8))
    sns.barplot(x=mean_salary_by_job, y=mean_salary_by_job.index)
    plt.title('Top 10 Job Roles by Average Salary')
    st.pyplot(plt)

# Prediction section
elif option == "Prediction":
    st.title("Salary Prediction")

    # Feature selection
    X = data_cleaned[['Job Title', 'Years of Experience', 'Education Level', 'Gender']]
    y = data_cleaned['Salary']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define categorical and numerical columns
    categorical_cols = ['Job Title', 'Education Level', 'Gender']
    numerical_cols = ['Years of Experience']

    # Preprocessing pipelines for both numerical and categorical data
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Define the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Create and evaluate the pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', model)])

    # Fit the model
    pipeline.fit(X_train, y_train)

    # Make predictions
    y_pred = pipeline.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    st.write(f'Mean Squared Error: {mse}')

    st.header("Feature Importance")
    importances = model.feature_importances_
    feature_names = numerical_cols + list(pipeline.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(categorical_cols))
    feature_importance = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    st.bar_chart(feature_importance)

    st.header("Make a Prediction")
    job_title = st.selectbox('Job Title', data_cleaned['Job Title'].unique())
    years_experience = st.slider('Years of Experience', min_value=0, max_value=40, value=5)
    education_level = st.selectbox('Education Level', data_cleaned['Education Level'].unique())
    gender = st.selectbox('Gender', data_cleaned['Gender'].unique())

    if st.button('Predict Salary'):
        input_data = pd.DataFrame([[job_title, years_experience, education_level, gender]], columns=X.columns)
        predicted_salary = pipeline.predict(input_data)[0]
        st.write(f'The predicted salary for a {job_title} with {years_experience} years of experience, {education_level}, and {gender} is: RM {predicted_salary:.2f}')
