# Loan Prediction Model

This project involves building a loan prediction model using Python in Google Colab, creating a dashboard with Tableau, and integrating TabPy to send the dataset to an API for predicting loan results.

## Table of Contents

- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Setup and Installation](#setup-and-installation)
- [Data Preprocessing](#data-preprocessing)
- [Model Building](#model-building)
- [Tableau Dashboard](#tableau-dashboard)
- [TabPy Integration](#tabpy-integration)
- [Usage](#usage)

## Project Overview

This project aims to predict loan eligibility based on various features using a machine learning model. The workflow involves:
1. Data preprocessing and model building in Google Colab.
2. Visualization and interactive analysis using Tableau.
3. Integration of Tableau with TabPy to call the model API and get real-time predictions.

## Requirements

- Python 3.x
- Google Colab
- Tableau Desktop
- TabPy (Tableau Python Server)
- Required Python libraries: pandas, numpy, scikit-learn, joblib, Flask (for API)

## Setup and Installation

### Python Environment

1. **Google Colab**: No installation required. You can run the provided Jupyter notebook directly in Google Colab.
2. **Local Environment**:
    ```sh
    pip install pandas numpy scikit-learn joblib Flask
    ```

### Tableau

1. **Download and install Tableau Desktop** from [Tableau's official website](https://www.tableau.com/products/desktop).
2. **Download and install TabPy**:
    ```sh
    pip install tabpy
    tabpy
    ```

## Data Preprocessing

1. Load the dataset in Google Colab.
2. Perform necessary data cleaning and preprocessing steps such as handling missing values, encoding categorical variables, and scaling features.
3. Save the preprocessed data for model building.

## Model Building

1. Split the preprocessed data into training and testing sets.
2. Build a machine learning model (e.g., Logistic Regression, Random Forest).
3. Train the model using the training data.
4. Evaluate the model performance using the testing data.
5. Save the trained model using `joblib` for later use in the API.

## Tableau Dashboard

1. **Create a new Tableau Workbook** and load the dataset.
2. **Design the dashboard** to visualize data and predictions.
3. **Connect Tableau to TabPy**:
    - Go to Help > Settings and Performance > Manage External Service Connection.
    - Select TabPy/External API and enter the server URL (default: `http://localhost:9004`).

## TabPy Integration

1. **Start TabPy** on your local machine:
    ```sh
    tabpy
    ```
2. **Deploy the Flask API** for the model:
    ```python
    from flask import Flask, request, jsonify
    import joblib
    import pandas as pd

    app = Flask(__name__)
    model = joblib.load('loan_prediction_model.pkl')

    @app.route('/predict', methods=['POST'])
    def predict():
        data = request.json
        df = pd.DataFrame(data)
        prediction = model.predict(df)
        return jsonify({'prediction': prediction.tolist()})

    if __name__ == '__main__':
        app.run(port=5000)
    ```
3. **Call the API from Tableau** using Python functions.

## Usage

1. **Run the model and API in Google Colab**:
    - Open the Jupyter notebook.
    - Run all cells to preprocess data, build the model, and start the API.
2. **Open Tableau Dashboard**:
    - Load the data and set up visualizations.
    - Ensure Tableau is connected to TabPy.
    - Use calculated fields to call the API and get predictions.
3. **Interact with the dashboard** to get real-time loan predictions.
