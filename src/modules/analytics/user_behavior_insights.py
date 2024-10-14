# src/modules/analytics/user_behavior_insights.py

import logging
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

import pandas as pd
from dotenv import load_dotenv

# Import DataStorage from data_management module
from src.modules.data_management.data_storage import DataStorage

# Configure logging with RotatingFileHandler to prevent log files from becoming too large
from logging.handlers import RotatingFileHandler

# Load environment variables
load_dotenv()

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = RotatingFileHandler(
    os.path.join('logs', 'user_behavior_insights.log'),
    maxBytes=5 * 1024 * 1024,  # 5 MB
    backupCount=5
)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)


class UserBehaviorInsights:
    """
    Analyzes user behavior patterns to provide actionable insights.
    Identifies behavioral trends, preferences, and anomalies in user activity.
    Helps in predicting future user needs and improving user experience by
    offering personalized recommendations or adjustments based on observed behaviors.
    """

    def __init__(self):
        """
        Initializes the UserBehaviorInsights module with necessary configurations.
        """
        try:
            # Initialize DataStorage instance for accessing user interaction data
            self.data_storage = DataStorage()
            logger.info("UserBehaviorInsights initialized successfully.")
        except Exception as e:
            logger.exception(f"Failed to initialize UserBehaviorInsights: {e}")
            raise e

    def get_user_interactions(self, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """
        Retrieves user interaction data within the specified date range.

        :param start_date: Start date for data retrieval.
        :param end_date: End date for data retrieval.
        :return: DataFrame containing user interactions or None if failed.
        """
        try:
            query = """
                SELECT *
                FROM user_interactions
                WHERE timestamp BETWEEN :start AND :end;
            """
            params = {'start': start_date.isoformat(), 'end': end_date.isoformat()}
            results = self.data_storage.query_data(query, params)
            if not results:
                logger.warning(f"No user interactions found between {start_date} and {end_date}.")
                return None
            df = pd.DataFrame(results)
            logger.info(f"Retrieved {len(df)} user interactions between {start_date} and {end_date}.")
            return df
        except Exception as e:
            logger.error(f"Failed to retrieve user interactions: {e}")
            return None

    def identify_trends(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Identifies trends in feature usage over time.

        :param df: DataFrame containing user interactions.
        :return: DataFrame with trend analysis or None if failed.
        """
        try:
            # Filter for feature usage events
            feature_usage_df = df[df['event_type'] == 'feature_usage']
            if feature_usage_df.empty:
                logger.warning("No feature usage data available for trend analysis.")
                return None

            # Convert timestamp to datetime if not already
            feature_usage_df['timestamp'] = pd.to_datetime(feature_usage_df['timestamp'])

            # Extract date for aggregation
            feature_usage_df['date'] = feature_usage_df['timestamp'].dt.date

            # Group by feature and date to count usage
            trend_df = feature_usage_df.groupby(['feature_name', 'date']).size().reset_index(name='usage_count')

            logger.info("Trend analysis completed successfully.")
            return trend_df
        except Exception as e:
            logger.error(f"Failed to identify trends: {e}")
            return None

    def identify_preferences(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Identifies user preferences based on feature usage frequency.

        :param df: DataFrame containing user interactions.
        :return: DataFrame with user preferences or None if failed.
        """
        try:
            # Filter for feature usage events
            feature_usage_df = df[df['event_type'] == 'feature_usage']
            if feature_usage_df.empty:
                logger.warning("No feature usage data available for preference analysis.")
                return None

            # Group by feature to count total usage
            preference_df = feature_usage_df.groupby('feature_name').size().reset_index(name='total_usage')

            # Calculate usage percentage
            total_usage = preference_df['total_usage'].sum()
            preference_df['usage_percentage'] = (preference_df['total_usage'] / total_usage) * 100

            # Sort by usage percentage descending
            preference_df = preference_df.sort_values(by='usage_percentage', ascending=False)

            logger.info("Preference analysis completed successfully.")
            return preference_df
        except Exception as e:
            logger.error(f"Failed to identify preferences: {e}")
            return None

    def detect_behavioral_anomalies(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Detects anomalies in user behavior, such as sudden spikes or drops in feature usage.

        :param df: DataFrame containing user interactions.
        :return: DataFrame with detected anomalies or None if failed.
        """
        try:
            trend_df = self.identify_trends(df)
            if trend_df is None:
                logger.warning("No trend data available for anomaly detection.")
                return None

            # Pivot the trend data for time series analysis
            pivot_df = trend_df.pivot(index='date', columns='feature_name', values='usage_count').fillna(0)

            # Detect anomalies using rolling mean and standard deviation
            window_size = 7  # 7-day window
            anomalies = []

            for feature in pivot_df.columns:
                rolling_mean = pivot_df[feature].rolling(window=window_size).mean()
                rolling_std = pivot_df[feature].rolling(window=window_size).std()
                current = pivot_df[feature]
                deviation = (current - rolling_mean) / rolling_std

                # Define anomaly as deviation greater than 2 standard deviations
                anomaly_dates = pivot_df.index[(deviation > 2) | (deviation < -2)]
                for date in anomaly_dates:
                    anomalies.append({
                        'feature_name': feature,
                        'date': date,
                        'usage_count': pivot_df.at[date, feature],
                        'deviation': deviation.at[date, feature]
                    })

            if anomalies:
                anomaly_df = pd.DataFrame(anomalies)
                logger.info(f"Detected {len(anomaly_df)} behavioral anomalies.")
                return anomaly_df
            else:
                logger.info("No behavioral anomalies detected.")
                return None
        except Exception as e:
            logger.error(f"Failed to detect behavioral anomalies: {e}")
            return None

    def generate_insights_report(self, start_date: datetime, end_date: datetime) -> Optional[Dict[str, Any]]:
        """
        Generates a comprehensive insights report based on user interactions within the specified date range.

        :param start_date: Start date for the report.
        :param end_date: End date for the report.
        :return: Dictionary containing insights or None if failed.
        """
        try:
            df = self.get_user_interactions(start_date, end_date)
            if df is None:
                logger.warning("No data available to generate insights report.")
                return None

            trends = self.identify_trends(df)
            preferences = self.identify_preferences(df)
            anomalies = self.detect_behavioral_anomalies(df)

            report = {
                'report_generated_at': datetime.utcnow().isoformat(),
                'time_period': {
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat()
                },
                'trends': trends.to_dict(orient='records') if trends is not None else [],
                'preferences': preferences.to_dict(orient='records') if preferences is not None else [],
                'anomalies': anomalies.to_dict(orient='records') if anomalies is not None else []
            }

            logger.info("Insights report generated successfully.")
            return report
        except Exception as e:
            logger.error(f"Failed to generate insights report: {e}")
            return None

    def provide_personalized_recommendations(self, user_id: str, report: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Provides personalized recommendations for a user based on the insights report.

        :param user_id: Unique identifier of the user.
        :param report: Dictionary containing the insights report.
        :return: Dictionary containing recommendations or None if failed.
        """
        try:
            if not report:
                logger.warning("No insights report provided for generating recommendations.")
                return None

            # Example Recommendation Logic:
            # Recommend features that are trending or preferred by similar users
            # Suggest improvements on features where user has encountered difficulties

            # Extract user-specific interactions
            user_interactions = self.data_storage.query_data(
                """
                SELECT *
                FROM user_interactions
                WHERE user_id = :user_id
                AND timestamp BETWEEN :start AND :end;
                """,
                {
                    'user_id': user_id,
                    'start': report['time_period']['start_date'],
                    'end': report['time_period']['end_date']
                }
            )
            user_df = pd.DataFrame(user_interactions)
            if user_df.empty:
                logger.warning(f"No interactions found for user {user_id} in the report period.")
                return None

            # Analyze user's preferred features
            user_pref_df = user_df[user_df['event_type'] == 'feature_usage']
            user_pref_counts = user_pref_df['feature_name'].value_counts().reset_index()
            user_pref_counts.columns = ['feature_name', 'usage_count']

            # Recommend new features that are popular but not used by the user
            popular_features = pd.DataFrame(report['preferences'])
            user_used_features = set(user_pref_counts['feature_name'])
            recommendations = popular_features[~popular_features['feature_name'].isin(user_used_features)].head(3)
            recommended_features = recommendations['feature_name'].tolist()

            # Suggest improvements on features where user encountered difficulties
            difficulty_df = user_df[user_df['event_type'] == 'difficulty_encountered']
            if not difficulty_df.empty:
                difficult_features = difficulty_df['feature_name'].value_counts().head(3).index.tolist()
            else:
                difficult_features = []

            recommendations_dict = {
                'recommended_features': recommended_features,
                'features_needing_improvement': difficult_features
            }

            logger.info(f"Generated recommendations for user {user_id}: {recommendations_dict}")
            return recommendations_dict
        except Exception as e:
            logger.error(f"Failed to provide personalized recommendations: {e}")
            return None

    # --------------------- Example Usage --------------------- #

    def example_usage(self):
        """
        Demonstrates example usage of the UserBehaviorInsights class.
        """
        try:
            insights = UserBehaviorInsights()
            start_date = datetime.utcnow() - timedelta(days=30)  # Last 30 days
            end_date = datetime.utcnow()
            report = insights.generate_insights_report(start_date, end_date)
            if report:
                print("Insights Report:")
                print(report)

                # Example: Provide recommendations for a specific user
                user_id = 'user123'
                recommendations = insights.provide_personalized_recommendations(user_id, report)
                if recommendations:
                    print(f"Personalized Recommendations for {user_id}:")
                    print(recommendations)
        except Exception as e:
            logger.exception(f"Error in example usage: {e}")


# --------------------- Main Execution --------------------- #

if __name__ == "__main__":
    # Run the user behavior insights example
    example_usage()
