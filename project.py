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

# Custom CSS for styling
st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #f3f4f6, #ffffff);
            color: #333;
            font-family: 'Arial', sans-serif;
        }
        .sidebar .sidebar-content {
            background-color: #ff5722;
            color: white;
        }
        .stButton > button {
            background-color: #6200ea;
            color: white;
            border: none;
            border-radius: 10px;
            padding: 15px 25px;
            font-size: 18px;
            transition: background-color 0.3s;
        }
        .stButton > button:hover {
            background-color: #3700b3;
        }
        h1 {
            color: #6200ea;
            text-align: center;
            font-size: 36px;
        }
        h2, h3 {
            color: #ff5722;
        }
        .form-container {
            background-color: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
            margin: 20px 0;
        }
        .footer {
            font-size: 0.9em;
            text-align: center;
            padding: 20px;
            color: #888;
        }
    </style>
""", unsafe_allow_html=True)

# Title of the web app
st.title("🩺 Disease Prediction App")

# Introduction message
st.markdown("""
Welcome to the *Disease Prediction App*! Select a dataset from the sidebar and input the required values to determine if you're at risk of a particular disease.
""")

# Sidebar for navigation and dataset selection
st.sidebar.header("Configuration")

# Dataset selection with emojis
dataset_name = st.sidebar.selectbox(
    "Select Dataset",
    ("Heart Disease ❤", "Brain Stroke 🧠", "Diabetes 🍭")
)

# Define the path to datasets
DATASETS = {
    "Heart Disease ❤": "HeartDiseaseML11.csv",
    "Brain Stroke 🧠": "brainstrokeML.csv",
    "Diabetes 🍭": "diabetesML1.csv"
}

@st.cache_resource
def load_data(name):
    """Load dataset based on the selected name. Caches the data to optimize performance."""
    try:
        data = pd.read_csv(DATASETS[name])
        return data
    except FileNotFoundError:
        st.error(f"Dataset file for {name} not found. Please check the file path.")
        return pd.DataFrame()

@st.cache_resource
def preprocess_data(df):
    """Preprocess the dataset: handle missing values and encode categorical variables."""
    if df.empty:
        return None, None, None

    # Drop rows with missing values
    df = df.dropna()

    # Reset index after dropping rows
    df.reset_index(drop=True, inplace=True)

    # Initialize label encoders dictionary
    label_encoders = {}

    # Identify categorical and numerical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Explicitly convert columns to appropriate data types
    for col in df.columns:
        if col in categorical_cols:
            df[col] = df[col].astype(str)
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Handle any remaining missing values after conversion
    df = df.dropna()

    # Encode categorical variables
    for column in categorical_cols:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    X = df.iloc[:, :-1]  # Features
    y = df.iloc[:, -1]  # Target variable

    return X, y, label_encoders

@st.cache_resource
def train_model(X, y):
    """Train the Random Forest Classifier and return the model and accuracy."""
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
X, y, label_encoders = preprocess_data(df)

if X is not None and y is not None:
    model, accuracy, feature_names = train_model(X, y)

    if model:
        # Display selected disease name above the input form
        st.subheader(f"Selected Disease: {dataset_name}")

        # User input form
        st.subheader("Enter Values to Predict the Outcome")
        with st.form(key='prediction_form', clear_on_submit=True):
            user_inputs = {}
            for feature in feature_names:
                if X[feature].dtype in [np.int64, np.float64]:
                    min_value = X[feature].min()
                    max_value = X[feature].max()
                    mean_value = X[feature].mean()
                    user_input = st.number_input(
                        label=f"Enter {feature} ({min_value} - {max_value})",
                        value=float(mean_value),
                        format="%.2f",
                        step=0.1
                    )
                else:
                    # Get the original categories from the label encoder
                    le = label_encoders.get(feature)
                    if le:
                        categories = le.classes_
                        user_input = st.selectbox(
                            label=f"Select {feature}",
                            options=categories
                        )
                    else:
                        user_input = st.text_input(f"Enter {feature}")
                user_inputs[feature] = user_input

            # Submit button
            submit_button = st.form_submit_button(label='Predict')

        if submit_button:
            try:
                # Prepare the input data for prediction
                input_data = pd.DataFrame([user_inputs])

                # Encode categorical inputs using the same label encoders
                for column in input_data.columns:
                    if column in label_encoders:
                        le = label_encoders[column]
                        if input_data[column][0] in le.classes_:
                            input_data[column] = le.transform([input_data[column][0]])  # Handle known categorical data
                        else:
                            st.error(f"Error: The value '{input_data[column][0]}' for '{column}' was not seen during training. Please select a valid value.")
                            return  # Stop the prediction if unseen label is detected
                    else:
                        input_data[column] = pd.to_numeric(input_data[column], errors='coerce')  # Handle numerical data

                # Ensure all features are present in the input data
                for feature in feature_names:
                    if feature not in input_data.columns:
                        input_data[feature] = 0  # Set default value if feature is missing

                # Reorder columns to match the training data
                input_data = input_data[feature_names]

                # Make the prediction
                prediction = model.predict(input_data)
                outcome = "Infected" if prediction[0] == 1 else "Not Infected"
                st.success(f"### Prediction Outcome: *{outcome}*")
            except NotFittedError:
                st.error("Model is not fitted yet.")
            except Exception as e:
                st.error(f"Error during prediction: {e}")

        # Display model accuracy
        st.subheader("Model Performance")
        st.write(f"*Accuracy:* {accuracy:.2f}")

        # Footer
        st.markdown("""
        ---
        ### About
        This application is a simple machine learning web app created using *Streamlit*. Select a disease, input the required values, and receive real-time predictions!

        ### Developed by:
        - Metyas Monir
        - Khaled Ayman
        - Noor Shrief

        ### Supervised by:
        Dr. Moshera Ghallab
        """)
else:
    st.warning("Please select a valid dataset to proceed.")
