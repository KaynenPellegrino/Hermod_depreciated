# src/modules/analytics/user_behavior_analytics.py

import logging
import os
from datetime import datetime
from typing import Optional, Dict, Any

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
    os.path.join('logs', 'user_behavior_analytics.log'),
    maxBytes=5 * 1024 * 1024,  # 5 MB
    backupCount=5
)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)


class UserBehaviorAnalytics:
    """
    Tracks and analyzes user interactions within the Hermod system.
    Collects data on feature usage, navigation paths, and areas of difficulty
    to provide insights for system improvements.
    """

    def __init__(self):
        """
        Initializes the UserBehaviorAnalytics module with necessary configurations.
        """
        try:
            # Initialize DataStorage instance for storing analytics data
            self.data_storage = DataStorage()
            logger.info("UserBehaviorAnalytics initialized successfully.")
        except Exception as e:
            logger.exception(f"Failed to initialize UserBehaviorAnalytics: {e}")
            raise e

    def log_feature_usage(self, user_id: str, feature_name: str, timestamp: Optional[datetime] = None) -> bool:
        """
        Logs the usage of a specific feature by a user.

        :param user_id: Unique identifier of the user.
        :param feature_name: Name of the feature used.
        :param timestamp: Time of the feature usage. If None, current time is used.
        :return: True if logging is successful, False otherwise.
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        data = {
            'user_id': user_id,
            'event_type': 'feature_usage',
            'feature_name': feature_name,
            'timestamp': timestamp.isoformat()
        }

        return self._save_event(data)

    def log_navigation(self, user_id: str, from_page: str, to_page: str, timestamp: Optional[datetime] = None) -> bool:
        """
        Logs the navigation path of a user within the system.

        :param user_id: Unique identifier of the user.
        :param from_page: The page the user navigated from.
        :param to_page: The page the user navigated to.
        :param timestamp: Time of the navigation. If None, current time is used.
        :return: True if logging is successful, False otherwise.
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        data = {
            'user_id': user_id,
            'event_type': 'navigation',
            'from_page': from_page,
            'to_page': to_page,
            'timestamp': timestamp.isoformat()
        }

        return self._save_event(data)

    def log_difficulty(self, user_id: str, feature_name: str, difficulty_level: int, timestamp: Optional[datetime] = None) -> bool:
        """
        Logs instances where a user encounters difficulty using a specific feature.

        :param user_id: Unique identifier of the user.
        :param feature_name: Name of the feature where difficulty was encountered.
        :param difficulty_level: Level of difficulty (e.g., on a scale of 1-5).
        :param timestamp: Time of the difficulty encounter. If None, current time is used.
        :return: True if logging is successful, False otherwise.
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        if not (1 <= difficulty_level <= 5):
            logger.warning(f"Invalid difficulty_level: {difficulty_level}. Must be between 1 and 5.")
            return False

        data = {
            'user_id': user_id,
            'event_type': 'difficulty_encountered',
            'feature_name': feature_name,
            'difficulty_level': difficulty_level,
            'timestamp': timestamp.isoformat()
        }

        return self._save_event(data)

    def _save_event(self, data: Dict[str, Any]) -> bool:
        """
        Saves an event to the analytics storage.

        :param data: Dictionary containing event data.
        :return: True if saving is successful, False otherwise.
        """
        try:
            self.data_storage.save_data(table='user_interactions', data=data)
            logger.info(f"Event logged successfully: {data}")
            return True
        except Exception as e:
            logger.error(f"Failed to log event: {data}. Error: {e}")
            return False

    def get_feature_usage_stats(self, start_date: datetime, end_date: datetime) -> Optional[Dict[str, int]]:
        """
        Retrieves statistics on feature usage within a specified date range.

        :param start_date: Start date for the statistics.
        :param end_date: End date for the statistics.
        :return: Dictionary with feature names as keys and usage counts as values, or None if failed.
        """
        try:
            query = """
                SELECT feature_name, COUNT(*) as usage_count
                FROM user_interactions
                WHERE event_type = 'feature_usage'
                AND timestamp BETWEEN :start AND :end
                GROUP BY feature_name
                ORDER BY usage_count DESC;
            """
            params = {'start': start_date.isoformat(), 'end': end_date.isoformat()}
            results = self.data_storage.query_data(query, params)
            stats = {row['feature_name']: row['usage_count'] for row in results}
            logger.info(f"Feature usage stats from {start_date} to {end_date}: {stats}")
            return stats
        except Exception as e:
            logger.error(f"Failed to retrieve feature usage stats: {e}")
            return None

    def get_navigation_paths(self, user_id: str, start_date: datetime, end_date: datetime) -> Optional[list]:
        """
        Retrieves the navigation paths of a specific user within a date range.

        :param user_id: Unique identifier of the user.
        :param start_date: Start date for the navigation data.
        :param end_date: End date for the navigation data.
        :return: List of navigation events ordered by timestamp, or None if failed.
        """
        try:
            query = """
                SELECT from_page, to_page, timestamp
                FROM user_interactions
                WHERE event_type = 'navigation'
                AND user_id = :user_id
                AND timestamp BETWEEN :start AND :end
                ORDER BY timestamp ASC;
            """
            params = {
                'user_id': user_id,
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            }
            results = self.data_storage.query_data(query, params)
            navigation_paths = [{'from_page': row['from_page'], 'to_page': row['to_page'], 'timestamp': row['timestamp']} for row in results]
            logger.info(f"Navigation paths for user {user_id} from {start_date} to {end_date}: {navigation_paths}")
            return navigation_paths
        except Exception as e:
            logger.error(f"Failed to retrieve navigation paths for user {user_id}: {e}")
            return None

    def get_difficulty_encounters(self, feature_name: str, start_date: datetime, end_date: datetime) -> Optional[Dict[str, int]]:
        """
        Retrieves statistics on difficulty encounters for a specific feature within a date range.

        :param feature_name: Name of the feature.
        :param start_date: Start date for the statistics.
        :param end_date: End date for the statistics.
        :return: Dictionary with difficulty levels as keys and encounter counts as values, or None if failed.
        """
        try:
            query = """
                SELECT difficulty_level, COUNT(*) as encounter_count
                FROM user_interactions
                WHERE event_type = 'difficulty_encountered'
                AND feature_name = :feature_name
                AND timestamp BETWEEN :start AND :end
                GROUP BY difficulty_level
                ORDER BY difficulty_level ASC;
            """
            params = {
                'feature_name': feature_name,
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            }
            results = self.data_storage.query_data(query, params)
            difficulty_stats = {str(row['difficulty_level']): row['encounter_count'] for row in results}
            logger.info(f"Difficulty encounters for feature '{feature_name}' from {start_date} to {end_date}: {difficulty_stats}")
            return difficulty_stats
        except Exception as e:
            logger.error(f"Failed to retrieve difficulty encounters for feature '{feature_name}': {e}")
            return None
