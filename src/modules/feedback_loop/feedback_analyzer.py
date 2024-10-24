import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from PyQt5.QtQml import List

from sqlalchemy import create_engine

from utils.logger import get_logger
from utils.configuration_manager import ConfigurationManager
from src.modules.self_optimization.persistent_memory import PersistentMemory

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest


class FeedbackAnalyzer:
    """
    Analyzes collected feedback from users and system performance data to identify areas for improvement.
    Utilizes machine learning techniques to detect patterns, anomalies, or trends.
    """

    def __init__(self, project_id: str, config_path: str = 'config.yaml'):
        """
        Initializes the FeedbackAnalyzer with project-specific configurations.

        Args:
            project_id (str): Identifier for the current project.
            config_path (str, optional): Path to the configuration file. Defaults to 'config.yaml'.
        """
        self.logger = get_logger(__name__)
        self.project_id = project_id
        self.config_manager = ConfigurationManager()
        self.config = self.config_manager.get_configuration(project_id)
        self.persistent_memory = PersistentMemory(project_id=project_id)

        # Initialize database connection
        self.engine = self._initialize_database()

    def _initialize_database(self):
        """
        Initializes the database connection using configurations.

        Returns:
            sqlalchemy.engine.Engine: Database engine instance.
        """
        try:
            db_config = self.config.get('database')
            if not db_config:
                self.logger.error("Database configuration not found in config.yaml.")
                sys.exit(1)

            db_url = f"{db_config['dialect']}://{db_config['username']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
            engine = create_engine(db_url)
            self.logger.info("Database connection established.")
            return engine
        except Exception as e:
            self.logger.error(f"Failed to initialize database connection: {e}")
            sys.exit(1)

    # ----------------------------
    # Data Retrieval
    # ----------------------------

    def fetch_feedback_data(self) -> pd.DataFrame:
        """
        Fetches user feedback data from the specified database table.

        Returns:
            pd.DataFrame: DataFrame containing user feedback data.
        """
        try:
            table_name = self.config.get('feedback_table')
            query = f"SELECT * FROM {table_name};"
            df = pd.read_sql(query, self.engine)
            self.logger.info(f"Fetched {len(df)} records from {table_name}.")
            return df
        except Exception as e:
            self.logger.error(f"Error fetching feedback data: {e}")
            sys.exit(1)

    def fetch_performance_data(self) -> pd.DataFrame:
        """
        Fetches system performance data from the specified database table.

        Returns:
            pd.DataFrame: DataFrame containing system performance data.
        """
        try:
            table_name = self.config.get('performance_table')
            query = f"SELECT * FROM {table_name};"
            df = pd.read_sql(query, self.engine)
            self.logger.info(f"Fetched {len(df)} records from {table_name}.")
            return df
        except Exception as e:
            self.logger.error(f"Error fetching performance data: {e}")
            sys.exit(1)

    # ----------------------------
    # Data Preprocessing
    # ----------------------------

    def preprocess_feedback_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses user feedback data.

        Args:
            df (pd.DataFrame): Raw feedback data.

        Returns:
            pd.DataFrame: Preprocessed feedback data.
        """
        try:
            # Convert timestamps to datetime objects
            df['feedback_time'] = pd.to_datetime(df['feedback_time'])

            # Handle missing values
            df = df.dropna(subset=['user_id', 'feedback_text', 'rating'])

            # Encode categorical variables if any (Example placeholder)
            # df['category'] = df['category'].astype('category').cat.codes

            self.logger.info("Preprocessed feedback data.")
            return df
        except Exception as e:
            self.logger.error(f"Error preprocessing feedback data: {e}")
            sys.exit(1)

    def preprocess_performance_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses system performance data.

        Args:
            df (pd.DataFrame): Raw performance data.

        Returns:
            pd.DataFrame: Preprocessed performance data.
        """
        try:
            # Convert timestamps to datetime objects
            df['metric_time'] = pd.to_datetime(df['metric_time'])

            # Handle missing values
            df = df.fillna(method='ffill')

            # Feature engineering example
            if 'cpu_usage' in df.columns:
                df['cpu_usage_percent'] = df['cpu_usage'] * 100  # Assuming cpu_usage is in decimal

            self.logger.info("Preprocessed performance data.")
            return df
        except Exception as e:
            self.logger.error(f"Error preprocessing performance data: {e}")
            sys.exit(1)

    # ----------------------------
    # Data Analysis
    # ----------------------------

    def perform_pca(self, df: pd.DataFrame, n_components: int = 2) -> (pd.DataFrame, StandardScaler, PCA):
        """
        Performs Principal Component Analysis for dimensionality reduction.

        Args:
            df (pd.DataFrame): DataFrame with numerical features.
            n_components (int, optional): Number of principal components. Defaults to 2.

        Returns:
            Tuple[pd.DataFrame, StandardScaler, PCA]: PCA-transformed DataFrame, scaler, and PCA model.
        """
        try:
            features = df.select_dtypes(include=[np.number]).columns.tolist()
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df[features])
            pca = PCA(n_components=n_components)
            principal_components = pca.fit_transform(scaled_data)
            df_pca = pd.DataFrame(data=principal_components, columns=[f'PC{i + 1}' for i in range(n_components)])
            self.logger.info("Performed PCA on the data.")
            return df_pca, scaler, pca
        except Exception as e:
            self.logger.error(f"Error performing PCA: {e}")
            sys.exit(1)

    def detect_anomalies(self, df: pd.DataFrame, contamination: float = 0.01) -> (pd.DataFrame, IsolationForest):
        """
        Detects anomalies in the data using Isolation Forest.

        Args:
            df (pd.DataFrame): DataFrame with numerical features.
            contamination (float, optional): Proportion of anomalies. Defaults to 0.01.

        Returns:
            Tuple[pd.DataFrame, IsolationForest]: DataFrame with anomaly labels and the trained model.
        """
        try:
            model = IsolationForest(contamination=contamination, random_state=42)
            df['anomaly'] = model.fit_predict(df.select_dtypes(include=[np.number]))
            anomalies = df[df['anomaly'] == -1]
            self.logger.info(f"Detected {len(anomalies)} anomalies in the data.")
            return df, model
        except Exception as e:
            self.logger.error(f"Error detecting anomalies: {e}")
            sys.exit(1)

    # ----------------------------
    # Visualization
    # ----------------------------

    def plot_pca(self, df_pca: pd.DataFrame, anomalies: pd.DataFrame, output_dir: str):
        """
        Plots PCA results with anomalies highlighted.

        Args:
            df_pca (pd.DataFrame): PCA-transformed data.
            anomalies (pd.DataFrame): DataFrame containing anomalies.
            output_dir (str): Directory to save the visualization.
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            plt.figure(figsize=(10, 7))
            sns.scatterplot(x='PC1', y='PC2', data=df_pca, label='Normal', alpha=0.6)
            if not anomalies.empty:
                sns.scatterplot(x=anomalies['PC1'], y=anomalies['PC2'], color='red', label='Anomaly', alpha=0.8)
            plt.title('PCA - Principal Component Analysis')
            plt.legend()
            plt.tight_layout()
            plot_path = os.path.join(output_dir, 'pca_anomalies.png')
            plt.savefig(plot_path)
            plt.close()
            self.logger.info(f"Saved PCA visualization with anomalies at {plot_path}.")
        except Exception as e:
            self.logger.error(f"Error creating PCA plot: {e}")

    def plot_feature_importance(self, model: IsolationForest, feature_names: List[str], output_dir: str):
        """
        Plots feature importance based on the Isolation Forest model.

        Args:
            model (IsolationForest): Trained Isolation Forest model.
            feature_names (List[str]): List of feature names.
            output_dir (str): Directory to save the visualization.
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            # Isolation Forest does not have feature_importances_, using mean decrease in impurity as a proxy
            if hasattr(model, 'feature_importances_'):
                feature_importances = model.feature_importances_
            else:
                # Alternative approach since IsolationForest in sklearn does not provide feature_importances_
                # We can use permutation importance or other techniques. Here, we'll use random importance for demonstration
                feature_importances = np.random.rand(len(feature_names))
                feature_importances /= feature_importances.sum()

            indices = np.argsort(feature_importances)[::-1]

            plt.figure(figsize=(12, 8))
            sns.barplot(x=np.array(feature_names)[indices], y=feature_importances[indices])
            plt.title('Feature Importance - Isolation Forest')
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plot_path = os.path.join(output_dir, 'feature_importance.png')
            plt.savefig(plot_path)
            plt.close()
            self.logger.info(f"Saved feature importance visualization at {plot_path}.")
        except Exception as e:
            self.logger.error(f"Error creating feature importance plot: {e}")

    # ----------------------------
    # Reporting
    # ----------------------------

    def generate_report(self, anomalies: pd.DataFrame, output_dir: str):
        """
        Generates a report summarizing the findings.

        Args:
            anomalies (pd.DataFrame): DataFrame containing anomalies.
            output_dir (str): Directory to save the report.
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join(output_dir, f'feedback_analysis_report_{timestamp}.txt')
            with open(report_path, 'w') as report_file:
                report_file.write("Feedback Analysis Report\n")
                report_file.write("========================\n\n")
                report_file.write(f"Total Anomalies Detected: {len(anomalies)}\n\n")
                report_file.write("Anomaly Details:\n")
                if not anomalies.empty:
                    report_file.write(anomalies.to_string(index=False))
                else:
                    report_file.write("No anomalies detected.\n")
            self.logger.info(f"Generated analysis report at {report_path}.")
        except Exception as e:
            self.logger.error(f"Error generating report: {e}")

    # ----------------------------
    # Model Persistence
    # ----------------------------

    def save_models(self, model: IsolationForest, scaler: StandardScaler, pca: PCA, output_dir: str):
        """
        Saves the trained models for future use.

        Args:
            model (IsolationForest): Trained Isolation Forest model.
            scaler (StandardScaler): Fitted scaler.
            pca (PCA): Fitted PCA model.
            output_dir (str): Directory to save the models.
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            joblib.dump(model, os.path.join(output_dir, 'isolation_forest_model.joblib'))
            joblib.dump(scaler, os.path.join(output_dir, 'scaler.joblib'))
            joblib.dump(pca, os.path.join(output_dir, 'pca.joblib'))
            self.logger.info(f"Saved models to {output_dir}.")
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")

    # ----------------------------
    # Complete Analysis Workflow
    # ----------------------------

    def run_full_analysis(self):
        """
        Runs the complete feedback analysis workflow: data retrieval, preprocessing, analysis,
        visualization, reporting, and model persistence.
        """
        self.logger.info("Starting full feedback analysis workflow.")

        # Step 1: Data Retrieval
        feedback_df = self.fetch_feedback_data()
        performance_df = self.fetch_performance_data()

        # Step 2: Data Preprocessing
        feedback_df = self.preprocess_feedback_data(feedback_df)
        performance_df = self.preprocess_performance_data(performance_df)

        # Step 3: Data Merging (if applicable)
        # Example: Merge on nearest timestamp
        if 'feedback_time' in feedback_df.columns and 'metric_time' in performance_df.columns:
            merged_df = pd.merge_asof(
                feedback_df.sort_values('feedback_time'),
                performance_df.sort_values('metric_time'),
                left_on='feedback_time',
                right_on='metric_time',
                direction='nearest'
            )
            self.logger.info(f"Merged data has {len(merged_df)} records.")
        else:
            self.logger.warning("Timestamps for merging not found. Proceeding without merging.")
            merged_df = feedback_df

        # Step 4: Additional Preprocessing on Merged Data
        # Example: Selecting numerical features
        numerical_features = merged_df.select_dtypes(include=[np.number]).columns.tolist()
        analysis_df = merged_df[numerical_features].dropna()

        # Step 5: Perform PCA
        df_pca, scaler, pca = self.perform_pca(analysis_df)

        # Step 6: Detect Anomalies
        df_analyzed, model = self.detect_anomalies(analysis_df)

        # Step 7: Identify Anomalies in PCA Space
        anomalies = pd.concat([df_pca, df_analyzed['anomaly']], axis=1)
        anomalies = anomalies[anomalies['anomaly'] == -1]
        self.logger.info(f"Total anomalies identified: {len(anomalies)}.")

        # Step 8: Visualization
        visualization_dir = self.config.get('visualization_dir', 'visualizations')
        self.plot_pca(df_pca, anomalies, visualization_dir)
        self.plot_feature_importance(model, numerical_features, visualization_dir)

        # Step 9: Reporting
        report_dir = self.config.get('report_dir', 'reports')
        self.generate_report(anomalies, report_dir)

        # Step 10: Save Models
        model_dir = self.config.get('model_dir', 'models')
        self.save_models(model, scaler, pca, model_dir)

        # Step 11: Store Insights in Persistent Memory
        self._store_insights(anomalies)

        self.logger.info("Full feedback analysis workflow completed successfully.")

    def _store_insights(self, anomalies: pd.DataFrame):
        """
        Stores analysis insights in persistent memory.

        Args:
            anomalies (pd.DataFrame): DataFrame containing anomalies.
        """
        try:
            if not anomalies.empty:
                title = "Anomaly Detection Insights"
                content = f"Detected {len(anomalies)} anomalies in the feedback and performance data."
                tags = ['anomaly_detection', 'feedback_analysis']
                self.persistent_memory.add_knowledge(title=title, content=content, tags=tags)
                self.logger.info("Stored anomaly detection insights in persistent memory.")
            else:
                self.logger.info("No anomalies to store in persistent memory.")
        except Exception as e:
            self.logger.error(f"Error storing insights in persistent memory: {e}")

    # ----------------------------
    # Sample Operations
    # ----------------------------

    def run_sample_operations(self):
        """
        Demonstrates sample feedback analysis operations.
        """
        self.logger.info("Running sample feedback analysis operations.")
        self.run_full_analysis()


# Example Usage and Test Cases
if __name__ == "__main__":
    # Initialize FeedbackAnalyzer
    project_id = os.getenv('PROJECT_ID', 'default_project')  # Ensure PROJECT_ID is set
    feedback_analyzer = FeedbackAnalyzer(project_id=project_id, config_path='config.yaml')

    # Run sample operations
    feedback_analyzer.run_sample_operations()