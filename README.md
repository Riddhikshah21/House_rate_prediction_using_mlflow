# House Rate Prediction using MLflow

## Overview
This project demonstrates the basics of **MLflow** by building a simple **House Rate Prediction** model. It covers key MLflow concepts like **experiment tracking, model versioning, and deployment**.

## Features
- **Data Preprocessing**: Cleans and prepares housing data.
- **Model Training**: Uses a regression model to predict house prices.
- **MLflow Integration**: Logs parameters, metrics, and model artifacts.
- **Model Evaluation**: Analyzes model performance using MLflow UI.

## Technologies Used
- Python
- Scikit-learn
- Pandas & NumPy
- MLflow
- Jupyter Notebook

## Setup & Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/House_rate_prediction_using_mlflow.git
   cd House_rate_prediction_using_mlflow
   ```
2. Create and activate a virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Run MLflow tracking server (optional):
   ```sh
   mlflow ui
   ```
   Then open `http://localhost:5000` in your browser.

## MLflow Tracking Example
The notebook logs:
- Model parameters (e.g., learning rate, number of estimators)
- Performance metrics (e.g., RMSE, R-squared)
- Trained model artifacts

Check MLflow UI for logged experiments:
```sh
mlflow ui
```

## Results
After training, the model predicts house rates based on input features. You can compare different runs in MLflow UI.


