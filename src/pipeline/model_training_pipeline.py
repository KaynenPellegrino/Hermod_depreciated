# src/pipeline/model_training_pipeline.py

import subprocess
import logging
from src.utils.configuration_manager import ConfigurationManager
from src.utils.logger import get_logger

def train_nlu_models(project_id: str):
    """
    Trains all NLU models for the given project.
    """
    logger = get_logger(__name__)
    logger.info(f"Starting NLU models training for project '{project_id}'.")

    # Define training script paths
    train_classification_script = 'src/modules/nlu/training_scripts/train_classification.py'
    train_qa_script = 'src/modules/nlu/training_scripts/train_qa.py'

    # Train Classification Model
    logger.info("Training Intent Classification Model.")
    subprocess.run([
        'python',
        train_classification_script
    ], check=True)

    # Train QA Model
    logger.info("Training Question-Answering Model.")
    subprocess.run([
        'python',
        train_qa_script
    ], check=True)

    logger.info("NLU models training completed successfully.")

if __name__ == "__main__":
    project_id = "proj_12345"  # Replace with your actual project ID
    train_nlu_models(project_id)
