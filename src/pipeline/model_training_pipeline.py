# src/pipelines/model_training_pipeline.py

import logging

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from src.modules.auto_ml.model_ensemble_builder import ModelEnsembleBuilder
from src.modules.data_management.data_storage import DataStorage

logger = logging.getLogger(__name__)

class ModelTrainingPipeline:
    def __init__(self):
        self.builder = ModelEnsembleBuilder()
        self.storage = DataStorage()

    def run(self, training_data_query: str, model_save_path: str):
        logger.info("Starting Model Training Pipeline")
        data = self.storage.query_data(query=training_data_query)
        if not data:
            logger.error("No data retrieved for model training")
            return

        df = pd.DataFrame(data)
        X = df.drop('target', axis=1)
        y = df['target']

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create and train ensemble model
        ensemble = self.builder.create_stacking_ensemble(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
                ('dt', DecisionTreeClassifier(random_state=42))
            ],
            final_estimator=LogisticRegression()
        )
        metrics = self.builder.train_ensemble(ensemble, X_train, y_train, X_val, y_val)

        # Save the trained model
        self.builder.save_model(ensemble, 'stacking_model', model_save_path)
        logger.info(f"Model Training Pipeline completed. Metrics: {metrics}")
