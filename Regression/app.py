import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Page configuration
st.set_page_config(page_title="Boston House Price Prediction App", layout="centered")

# Title and description
st.title("Boston House Price Prediction App")
st.markdown("""
This app predicts the **Boston House Price** based on various input parameters.
Adjust the parameters on the sidebar to get the predicted price.
""")
regression_model = st.selectbox('Select Regression Model', ['Linear Regression', 'Random Forest Regression'])
file_name = 'Regression/' + ''.join(regression_model.replace(" ", "_")).lower() + '_model.pkl'

col_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'PRICE']
df = pd.read_csv("boston_housing.csv", header=None, delimiter=r"\s+", names=col_names)

X = df.drop(columns='PRICE')
data = {col : 0 for col in col_names[:-1]}

# Sidebar for user input parameters
st.sidebar.header('Specify Input Parameters')

def user_input_features():
    data['LSTAT'] = st.sidebar.slider('LSTAT (Percentage of lower status of the population)', 0.0, 40.0, 10.0)
    data['RM'] = st.sidebar.slider('RM (Average number of rooms per dwelling)', 3.0, 9.0, 6.0)
    data['PTRATIO'] = st.sidebar.slider('PTRATIO (Pupil-teacher ratio by town)', 10.0, 22.0, 18.0)
    data['INDUS'] = st.sidebar.slider('INDUS (Proportion of non-retail business acres per town)', 0.0, 30.0, 10.0)
    data['TAX'] = st.sidebar.slider('TAX (Full-value property-tax rate per $10,000)', 100.0, 1000.0, 500.0)

    mean_values = df.drop(columns=['LSTAT', 'RM', 'PTRATIO', 'INDUS', 'TAX', 'PRICE']).mean()
    for feature, mean_value in mean_values.items():
        data[feature] = mean_value
    
    features = pd.DataFrame(data, index=[0])
    return features

df_input = user_input_features()

# Main Panel
st.subheader('Specified Input Parameters')
st.write(df_input)

# Load the pickled model
with open(file_name, 'rb') as file:
    model = pickle.load(file)

# Apply Model to Make Prediction
prediction = model.predict(df_input)

st.subheader('Prediction of PRICE')
st.write(f"Predicted Median Value of Owner-Occupied Homes: ${prediction[0] * 1000:.2f}")

# Feature importance for Linear Regression
if regression_model == 'Linear Regression':
    feature_importance = np.abs(model.coef_)
    feature_names = X.columns
    sorted_idx = np.argsort(feature_importance)
    
    fig, ax = plt.subplots()
    ax.barh(feature_names[sorted_idx], feature_importance[sorted_idx])
    ax.set_xlabel('Feature Importance')
    ax.set_title('Feature Importance for Linear Regression')
    st.pyplot(fig)

# Feature importance for Random Forest Regression
if regression_model == 'Random Forest Regression':
    feature_importance = model.feature_importances_
    feature_names = X.columns
    sorted_idx = np.argsort(feature_importance)
    
    fig, ax = plt.subplots()
    ax.barh(feature_names[sorted_idx], feature_importance[sorted_idx])
    ax.set_xlabel('Feature Importance')
    ax.set_title('Feature Importance for Random Forest Regression')
    st.pyplot(fig)

# Additional plots and enhancements
st.subheader('Correlation Matrix')
corr = df.corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="Greens", ax=ax)
st.pyplot(fig)