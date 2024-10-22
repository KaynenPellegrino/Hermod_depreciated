#!/usr/bin/env python3
"""
feedback_analyzer.py

Function: Feedback Data Analysis
Purpose: Analyzes collected feedback from users and system performance data to identify areas for improvement.
         Utilizes machine learning techniques to detect patterns, anomalies, or trends.
"""

import os
import sys
import yaml
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# ----------------------------
# Configuration and Logging
# ----------------------------

def load_config(config_path='config.yaml'):
    """
    Load configuration from a YAML file.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logging.error(f"Failed to load configuration file: {e}")
        sys.exit(1)


def setup_logging(log_dir='logs'):
    """
    Setup logging configuration.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'feedback_analyzer_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


# ----------------------------
# Data Retrieval
# ----------------------------

def fetch_feedback_data(engine, table_name):
    """
    Fetch user feedback data from the specified database table.
    """
    try:
        query = f"SELECT * FROM {table_name};"
        df = pd.read_sql(query, engine)
        logging.info(f"Fetched {len(df)} records from {table_name}.")
        return df
    except Exception as e:
        logging.error(f"Error fetching feedback data: {e}")
        sys.exit(1)


def fetch_performance_data(engine, table_name):
    """
    Fetch system performance data from the specified database table.
    """
    try:
        query = f"SELECT * FROM {table_name};"
        df = pd.read_sql(query, engine)
        logging.info(f"Fetched {len(df)} records from {table_name}.")
        return df
    except Exception as e:
        logging.error(f"Error fetching performance data: {e}")
        sys.exit(1)


# ----------------------------
# Data Preprocessing
# ----------------------------

def preprocess_feedback_data(df):
    """
    Preprocess user feedback data.
    """
    try:
        # Example: Convert timestamps to datetime objects
        df['feedback_time'] = pd.to_datetime(df['feedback_time'])

        # Example: Handle missing values
        df = df.dropna(subset=['user_id', 'feedback_text', 'rating'])

        # Example: Encode categorical variables if any
        # df['category'] = df['category'].astype('category').cat.codes

        logging.info("Preprocessed feedback data.")
        return df
    except Exception as e:
        logging.error(f"Error preprocessing feedback data: {e}")
        sys.exit(1)


def preprocess_performance_data(df):
    """
    Preprocess system performance data.
    """
    try:
        # Example: Convert timestamps to datetime objects
        df['metric_time'] = pd.to_datetime(df['metric_time'])

        # Example: Handle missing values
        df = df.fillna(method='ffill')

        # Example: Feature engineering
        df['cpu_usage_percent'] = df['cpu_usage'] * 100  # Assuming cpu_usage is in decimal

        logging.info("Preprocessed performance data.")
        return df
    except Exception as e:
        logging.error(f"Error preprocessing performance data: {e}")
        sys.exit(1)


# ----------------------------
# Data Analysis
# ----------------------------

def perform_pca(df, n_components=2):
    """
    Perform Principal Component Analysis for dimensionality reduction.
    """
    try:
        features = df.select_dtypes(include=[np.number]).columns.tolist()
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[features])
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(scaled_data)
        df_pca = pd.DataFrame(data=principal_components, columns=[f'PC{i + 1}' for i in range(n_components)])
        logging.info("Performed PCA on the data.")
        return df_pca, scaler, pca
    except Exception as e:
        logging.error(f"Error performing PCA: {e}")
        sys.exit(1)


def detect_anomalies(df, contamination=0.01):
    """
    Detect anomalies in the data using Isolation Forest.
    """
    try:
        model = IsolationForest(contamination=contamination, random_state=42)
        df['anomaly'] = model.fit_predict(df.select_dtypes(include=[np.number]))
        anomalies = df[df['anomaly'] == -1]
        logging.info(f"Detected {len(anomalies)} anomalies in the data.")
        return df, model
    except Exception as e:
        logging.error(f"Error detecting anomalies: {e}")
        sys.exit(1)


# ----------------------------
# Visualization
# ----------------------------

def plot_pca(df_pca, anomalies, output_dir='visualizations'):
    """
    Plot PCA results with anomalies highlighted.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        plt.figure(figsize=(10, 7))
        sns.scatterplot(x='PC1', y='PC2', data=df_pca, label='Normal')
        sns.scatterplot(x=anomalies['PC1'], y=anomalies['PC2'], color='red', label='Anomaly')
        plt.title('PCA - Principal Component Analysis')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'pca_anomalies.png'))
        plt.close()
        logging.info("Saved PCA visualization with anomalies.")
    except Exception as e:
        logging.error(f"Error creating PCA plot: {e}")


def plot_feature_importance(model, feature_names, output_dir='visualizations'):
    """
    Plot feature importance based on the Isolation Forest model.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        feature_importances = np.abs(model.feature_importances_)
        indices = np.argsort(feature_importances)[::-1]

        plt.figure(figsize=(12, 8))
        sns.barplot(x=np.array(feature_names)[indices], y=feature_importances[indices])
        plt.title('Feature Importance - Isolation Forest')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
        plt.close()
        logging.info("Saved feature importance visualization.")
    except Exception as e:
        logging.error(f"Error creating feature importance plot: {e}")


# ----------------------------
# Reporting
# ----------------------------

def generate_report(anomalies, output_dir='reports'):
    """
    Generate a report summarizing the findings.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir,
                                   f'feedback_analysis_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
        with open(report_path, 'w') as report_file:
            report_file.write("Feedback Analysis Report\n")
            report_file.write("========================\n\n")
            report_file.write(f"Total Anomalies Detected: {len(anomalies)}\n\n")
            report_file.write("Anomaly Details:\n")
            report_file.write(anomalies.to_string(index=False))
        logging.info(f"Generated analysis report at {report_path}.")
    except Exception as e:
        logging.error(f"Error generating report: {e}")


# ----------------------------
# Model Persistence
# ----------------------------

def save_model(model, scaler, pca, output_dir='models'):
    """
    Save the trained models for future use.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(model, os.path.join(output_dir, 'isolation_forest_model.joblib'))
        joblib.dump(scaler, os.path.join(output_dir, 'scaler.joblib'))
        joblib.dump(pca, os.path.join(output_dir, 'pca.joblib'))
        logging.info("Saved models to disk.")
    except Exception as e:
        logging.error(f"Error saving models: {e}")


# ----------------------------
# Main Function
# ----------------------------

def main():
    # Load configuration
    config = load_config()

    # Setup logging
    setup_logging(config.get('log_dir', 'logs'))

    logging.info("Starting Feedback Analyzer.")

    # Database connection
    db_config = config.get('database')
    if not db_config:
        logging.error("Database configuration not found in config.yaml.")
        sys.exit(1)

    db_url = f"{db_config['dialect']}://{db_config['username']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    engine = create_engine(db_url)

    # Fetch data
    feedback_df = fetch_feedback_data(engine, config['feedback_table'])
    performance_df = fetch_performance_data(engine, config['performance_table'])

    # Preprocess data
    feedback_df = preprocess_feedback_data(feedback_df)
    performance_df = preprocess_performance_data(performance_df)

    # Merge dataframes on a common key or timestamp if necessary
    # Example: Assuming both have a timestamp and we can merge on it
    merged_df = pd.merge_asof(feedback_df.sort_values('feedback_time'),
                              performance_df.sort_values('metric_time'),
                              left_on='feedback_time',
                              right_on='metric_time',
                              direction='nearest')
    logging.info(f"Merged data has {len(merged_df)} records.")

    # Handle any additional preprocessing on merged data
    # For example, feature selection
    analysis_df = merged_df.select_dtypes(include=[np.number]).dropna()

    # Perform PCA
    df_pca, scaler, pca = perform_pca(analysis_df)

    # Detect anomalies
    df_analyzed, model = detect_anomalies(analysis_df)

    # Identify anomalies in PCA space
    anomalies = pd.concat([df_pca, df_analyzed['anomaly']], axis=1)
    anomalies = anomalies[anomalies['anomaly'] == -1]

    # Visualization
    plot_pca(df_pca, anomalies, config.get('visualization_dir', 'visualizations'))
    plot_feature_importance(model, analysis_df.columns, config.get('visualization_dir', 'visualizations'))

    # Reporting
    generate_report(anomalies, config.get('report_dir', 'reports'))

    # Save models
    save_model(model, scaler, pca, config.get('model_dir', 'models'))

    logging.info("Feedback Analyzer completed successfully.")


if __name__ == "__main__":
    main()
