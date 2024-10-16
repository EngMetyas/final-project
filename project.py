import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import NotFittedError

# Set the page configuration
st.set_page_config(
    page_title="Disease Prediction App",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title of the web app
st.title("🩺 Disease Prediction App")

# Introduction message
st.markdown("""
Welcome to the **Disease Prediction App**! Select a dataset and enter the required values to determine if you're at risk of a particular disease.
""")

# Sidebar for navigation and dataset selection
st.sidebar.header("Configuration")

# Dataset selection
dataset_name = st.sidebar.selectbox(
    "Select Dataset",
    ("Heart Disease", "Brain Stroke", "Diabetes")
)

# Define the path to datasets
DATASETS = {
    "Heart Disease": "Heart_cleanedML.csv",
    "Brain Stroke": "brainstrokeML.csv",
    "Diabetes": "diabetesML1.csv"
}


@st.cache_resource
def load_data(name):
    """
    Load dataset based on the selected name.
    Caches the data to optimize performance.
    """
    try:
        data = pd.read_csv(DATASETS[name])
        return data
    except FileNotFoundError:
        st.error(f"Dataset file for {name} not found.")
        return pd.DataFrame()


@st.cache_resource
def preprocess_data(df):
    """
    Preprocess the dataset:
    - Handle missing values if any.
    - Encode categorical variables.
    - Split into features and target.
    """
    if df.empty:
        return None, None

    # Handle missing values if necessary
    df = df.dropna()

    label_encoders = {}
    for column in df.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le  # Store encoders for potential future use

    X = df.iloc[:, :-1]  # Features
    y = df.iloc[:, -1]  # Target variable

    return X, y


@st.cache_resource
def train_model(X, y):
    """
    Train the Random Forest Classifier.
    Caches the trained model to avoid retraining on every run.
    """
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        return model, accuracy, X.columns.tolist()
    except Exception as e:
        st.error(f"Error in model training: {e}")
        return None, None, []


# Load and preprocess data
df = load_data(dataset_name)
X, y = preprocess_data(df)

if X is not None and y is not None:
    model, accuracy, feature_names = train_model(X, y)

    if model:
        # Display dataset preview
        st.subheader(f"{dataset_name} Dataset Preview")
        st.dataframe(df.head())

        # Input fields for user values
        st.subheader("Enter Values to Predict the Outcome")

        # Create a form for user inputs
        with st.form(key='prediction_form'):
            user_inputs = {}
            for feature in feature_names:
                # Determine the input type based on feature data type
                if X[feature].dtype == 'int64' or X[feature].dtype == 'float64':
                    user_input = st.number_input(
                        label=f"Enter {feature}",
                        value=float(X[feature].mean()),
                        format="%.2f"
                    )
                else:
                    # For categorical features, provide a selectbox with unique options
                    unique_values = sorted(df[feature].unique())
                    user_input = st.selectbox(
                        label=f"Select {feature}",
                        options=unique_values
                    )
                user_inputs[feature] = user_input

            # Submit button
            submit_button = st.form_submit_button(label='Predict')

        if submit_button:
            try:
                # Prepare the input data for prediction
                input_data = pd.DataFrame([user_inputs])

                # Encode categorical inputs using the same label encoders
                for column in input_data.select_dtypes(include=['object', 'category']).columns:
                    le = LabelEncoder()
                    input_data[column] = le.fit_transform(input_data[column])

                prediction = model.predict(input_data)
                outcome = "Infected" if prediction[0] == 1 else "Not Infected"
                st.success(f"### Prediction Outcome: **{outcome}**")
            except NotFittedError:
                st.error("Model is not fitted yet.")
            except Exception as e:
                st.error(f"Error during prediction: {e}")

        # Display model accuracy
        st.subheader("Model Performance")
        st.write(f"**Accuracy:** {accuracy:.2f}")

        # Footer
        st.markdown("""
        ---
        ### About
        This application is a simple machine learning web app created using **Streamlit**. Select a disease, input the required values, and receive real-time predictions!

        ### Developed by:
        - Metyas Monir
        - Khaled Ayman
        - Noor Shrief

        ### Supervised by:
        Dr. Moshera Ghallab
        """)
else:
    st.warning("Please select a valid dataset to proceed.")
