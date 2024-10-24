# src/modules/self_optimization/post_mortem_learner.py

import os
import logging
from typing import Optional, List, Dict, Any

import joblib
import pandas as pd
from shap import kmeans
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

from utils.logger import get_logger
from utils.configuration_manager import ConfigurationManager
from src.modules.self_optimization.persistent_memory import PersistentMemory


class PostMortemLearner:
    """
    Analyzes past failures and learns from them to improve future performance.
    """

    def __init__(self, project_id: str, persistent_memory: PersistentMemory):
        """
        Initializes the PostMortemLearner with necessary configurations and dependencies.

        Args:
            project_id (str): Unique identifier for the project.
            persistent_memory (PersistentMemory): Instance of PersistentMemory for storing insights.
        """
        self.logger = get_logger(__name__)
        self.project_id = project_id
        self.persistent_memory = persistent_memory
        self.config_manager = ConfigurationManager()
        self.config = self.config_manager.get_configuration(project_id)

        self.failure_logs_path = self.config.get('failure_logs_path', f'memory/{project_id}/failure_logs.csv')
        self.model_path = os.path.join(self.config.get('model_dir', f'models/{project_id}/post_mortem_model.joblib'))
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.label_encoder = LabelEncoder()

        self.logger.info(f"PostMortemLearner initialized for project '{project_id}'.")

    def load_failure_logs(self) -> Optional[pd.DataFrame]:
        """
        Loads failure logs from the specified CSV file.

        Returns:
            Optional[pd.DataFrame]: DataFrame containing failure logs, or None if loading fails.
        """
        self.logger.info(f"Loading failure logs from '{self.failure_logs_path}'.")
        try:
            df = pd.read_csv(self.failure_logs_path, parse_dates=['timestamp'])
            self.logger.info(f"Loaded {len(df)} failure logs.")
            return df
        except FileNotFoundError:
            self.logger.error(f"Failure logs file '{self.failure_logs_path}' not found.")
            return None
        except pd.errors.ParserError as e:
            self.logger.error(f"Pandas parser error while reading '{self.failure_logs_path}': {e}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to load failure logs: {e}", exc_info=True)
            return None

    def analyze_failures(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Analyzes failure logs to identify common patterns.

        Args:
            df (pd.DataFrame): DataFrame containing failure logs.

        Returns:
            Optional[pd.DataFrame]: DataFrame with identified patterns, or None if analysis fails.
        """
        self.logger.info("Analyzing failure logs for common patterns.")
        try:
            # Feature Engineering
            df['error_message'] = df['error_message'].astype(str)
            tfidf_matrix = self.vectorizer.fit_transform(df['error_message'])
            pca = PCA(n_components=2)
            principal_components = pca.fit_transform(tfidf_matrix.toarray())
            df['pc1'] = principal_components[:, 0]
            df['pc2'] = principal_components[:, 1]

            # Clustering to identify patterns
            kmeans = KMeans(n_clusters=5, random_state=42)
            df['cluster'] = kmeans.fit_predict(df[['pc1', 'pc2']])

            self.logger.info("Failure logs analysis completed successfully.")
            return df
        except Exception as e:
            self.logger.error(f"Failed to analyze failures: {e}", exc_info=True)
            return None

    def learn_from_failures(self, analyzed_df: pd.DataFrame) -> bool:
        """
        Learns from identified failure patterns and updates persistent memory.

        Args:
            analyzed_df (pd.DataFrame): DataFrame containing analyzed failure logs with patterns.

        Returns:
            bool: True if learning is successful, False otherwise.
        """
        self.logger.info("Learning from identified failure patterns.")
        try:
            cluster_summary = analyzed_df.groupby('cluster').agg({
                'error_type': 'first',
                'error_message': 'first',
                'timestamp': 'count'
            }).rename(columns={'timestamp': 'occurrences'}).reset_index()

            # Store each cluster as a knowledge entry
            for _, row in cluster_summary.iterrows():
                title = f"Failure Pattern Cluster {row['cluster']}"
                content = f"Error Type: {row['error_type']}\nError Message: {row['error_message']}\nOccurrences: {row['occurrences']}"
                tags = ['failure_analysis', 'post_mortem', 'error_pattern']
                self.persistent_memory.add_knowledge(title=title, content=content, tags=tags)

            # Save the clustering model for future predictions
            import joblib
            joblib.dump(self.vectorizer, os.path.join(self.config.get('model_dir', f'models/{self.project_id}/'), 'tfidf_vectorizer.joblib'))
            joblib.dump(self.label_encoder, os.path.join(self.config.get('model_dir', f'models/{self.project_id}/'), 'label_encoder.joblib'))
            joblib.dump(kmeans, self.model_path)
            self.logger.info(f"Clustering model saved to '{self.model_path}'.")

            self.logger.info("Learning from failures completed successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to learn from failures: {e}", exc_info=True)
            return False

    def predict_failure_pattern(self, error_message: str) -> Optional[int]:
        """
        Predicts the failure pattern cluster for a given error message.

        Args:
            error_message (str): The error message to classify.

        Returns:
            Optional[int]: Cluster number if prediction is successful, else None.
        """
        self.logger.info(f"Predicting failure pattern for error message: '{error_message}'.")
        try:
            tfidf_vector = self.vectorizer.transform([error_message])
            pca = PCA(n_components=2)
            tfidf_pca = pca.fit_transform(tfidf_vector.toarray())
            kmeans = joblib.load(self.model_path)
            cluster = kmeans.predict(tfidf_pca)
            self.logger.info(f"Predicted cluster: {cluster[0]}")
            return cluster[0]
        except Exception as e:
            self.logger.error(f"Failed to predict failure pattern: {e}", exc_info=True)
            return None

    def run_post_mortem_analysis(self):
        """
        Runs the complete post-mortem analysis pipeline.
        """
        self.logger.info("Starting post-mortem analysis pipeline.")
        df = self.load_failure_logs()
        if df is None or df.empty:
            self.logger.warning("No failure logs to analyze.")
            return

        analyzed_df = self.analyze_failures(df)
        if analyzed_df is None or analyzed_df.empty:
            self.logger.warning("Analysis returned no data.")
            return

        success = self.learn_from_failures(analyzed_df)
        if success:
            self.logger.info("Post-mortem analysis pipeline completed successfully.")
        else:
            self.logger.error("Post-mortem analysis pipeline failed.")

    def run_sample_operations(self):
        """
        Runs sample operations to demonstrate usage of PostMortemLearner.
        """
        self.logger.info("Running sample operations on PostMortemLearner.")

        # Example: Run post-mortem analysis
        self.run_post_mortem_analysis()

        # Example: Predict failure pattern for a new error message
        sample_error = "NullPointerException at line 42 in module X"
        cluster = self.predict_failure_pattern(sample_error)
        if cluster is not None:
            print(f"Predicted Failure Pattern Cluster: {cluster}")
        else:
            print("Failed to predict failure pattern.")


# Example Usage and Test Cases
if __name__ == "__main__":
    # Initialize PersistentMemory
    project_id = os.getenv('PROJECT_ID', 'default_project')  # Ensure PROJECT_ID is set
    persistent_memory = PersistentMemory(project_id=project_id)

    # Initialize PostMortemLearner
    post_mortem_learner = PostMortemLearner(project_id=project_id, persistent_memory=persistent_memory)

    # Run sample operations
    post_mortem_learner.run_sample_operations()
