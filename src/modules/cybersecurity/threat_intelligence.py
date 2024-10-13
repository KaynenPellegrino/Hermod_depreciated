# src/modules/cybersecurity/threat_intelligence.py

import logging
import os
import json
import time
from datetime import datetime
from typing import Any, Dict, Optional, List

import requests
from dotenv import load_dotenv
import feedparser
import pandas as pd

# Import MetadataStorage from data_management module
from src.modules.data_management.metadata_storage import MetadataStorage

# Import NotificationManager from notifications module
from src.modules.notifications.notification_manager import NotificationManager

# Load environment variables from .env file
load_dotenv()

# Configure logging with RotatingFileHandler to prevent log files from becoming too large
from logging.handlers import RotatingFileHandler

from src.utils.logger import get_logger

logger = get_logger(__name__, 'logs/threat_intelligence.log')


class ThreatIntelligenceFetcher:
    """
    Fetches threat intelligence data from various sources.
    """

    def __init__(self):
        """
        Initializes the ThreatIntelligenceFetcher with necessary configurations.
        """
        # Initialize Metadata Storage
        self.metadata_storage = MetadataStorage()

        # Initialize Notification Manager
        self.notification_manager = NotificationManager()

        # Configuration parameters
        self.fetch_interval = int(os.getenv('THREAT_INTELLIGENCE_FETCH_INTERVAL', '3600'))  # in seconds
        self.sources = self.load_sources()

        logger.info("ThreatIntelligenceFetcher initialized successfully.")

    def load_sources(self) -> List[Dict[str, Any]]:
        """
        Loads threat intelligence sources from environment variables or configuration files.

        :return: List of source dictionaries
        """
        # Example: Load sources from a JSON file specified in environment variables
        sources_file = os.getenv('THREAT_INTELLIGENCE_SOURCES_FILE', 'threat_intelligence_sources.json')
        if not os.path.exists(sources_file):
            logger.error(f"Sources file '{sources_file}' does not exist.")
            return []

        try:
            with open(sources_file, 'r') as f:
                sources = json.load(f).get('sources', [])
            logger.info(f"Loaded {len(sources)} threat intelligence sources from '{sources_file}'.")
            return sources
        except Exception as e:
            logger.error(f"Failed to load sources from '{sources_file}': {e}")
            return []

    def fetch_from_api(self, source: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Fetches threat intelligence data from an API source.

        :param source: Dictionary containing source details
        :return: Parsed threat data or None if failed
        """
        api_url = source.get('api_url')
        headers = source.get('headers', {})
        params = source.get('params', {})
        auth = None

        if source.get('auth_type') == 'api_key':
            auth = (source.get('auth_username'), source.get('auth_password'))
        elif source.get('auth_type') == 'bearer':
            headers['Authorization'] = f"Bearer {source.get('auth_token')}"

        try:
            response = requests.get(api_url, headers=headers, params=params, auth=auth, timeout=30)
            response.raise_for_status()
            data = response.json()
            logger.info(f"Fetched data from API source '{source.get('name')}'.")
            return data
        except Exception as e:
            logger.error(f"Failed to fetch data from API source '{source.get('name')}': {e}")
            return None

    def fetch_from_feed(self, source: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Fetches threat intelligence data from an RSS/Atom feed source.

        :param source: Dictionary containing source details
        :return: Parsed threat data or None if failed
        """
        feed_url = source.get('feed_url')
        try:
            feed = feedparser.parse(feed_url)
            if feed.bozo:
                raise feed.bozo_exception
            entries = feed.entries
            logger.info(f"Fetched {len(entries)} entries from feed source '{source.get('name')}'.")
            return {'entries': entries}
        except Exception as e:
            logger.error(f"Failed to fetch data from feed source '{source.get('name')}': {e}")
            return None

    def parse_mitre_attack_data(self, data: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        Parses MITRE ATT&CK data.

        :param data: Raw data fetched from MITRE ATT&CK API
        :return: DataFrame containing parsed threat data or None if failed
        """
        try:
            from attackcti import attack_client
            client = attack_client.AttackClient()
            techniques = client.get_techniques()
            df = pd.json_normalize(techniques)
            logger.info("Parsed MITRE ATT&CK data successfully.")
            return df
        except Exception as e:
            logger.error(f"Failed to parse MITRE ATT&CK data: {e}")
            return None

    def parse_cisa_feed(self, data: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        Parses CISA RSS feed data.

        :param data: Raw data fetched from CISA RSS feed
        :return: DataFrame containing parsed threat data or None if failed
        """
        try:
            entries = data.get('entries', [])
            df = pd.DataFrame([{
                'title': entry.get('title'),
                'link': entry.get('link'),
                'published': entry.get('published'),
                'summary': entry.get('summary')
            } for entry in entries])
            logger.info("Parsed CISA feed data successfully.")
            return df
        except Exception as e:
            logger.error(f"Failed to parse CISA feed data: {e}")
            return None

    def process_source(self, source: Dict[str, Any]):
        """
        Processes a single threat intelligence source.

        :param source: Dictionary containing source details
        """
        source_type = source.get('type')
        data = None

        if source_type == 'api':
            data = self.fetch_from_api(source)
            if data:
                # Implement source-specific parsing
                if source.get('name') == 'MITRE ATT&CK':
                    parsed_data = self.parse_mitre_attack_data(data)
                else:
                    parsed_data = pd.json_normalize(data)  # Generic parsing
        elif source_type == 'feed':
            data = self.fetch_from_feed(source)
            if data:
                # Implement source-specific parsing
                if source.get('name') == 'CISA Alerts':
                    parsed_data = self.parse_cisa_feed(data)
                else:
                    parsed_data = pd.json_normalize(data)  # Generic parsing
        else:
            logger.warning(f"Unknown source type '{source_type}' for source '{source.get('name')}'. Skipping.")
            return

        if data and parsed_data is not None:
            # Save parsed data to Metadata Storage
            report = {
                'source_name': source.get('name'),
                'fetched_at': datetime.utcnow().isoformat(),
                'data': parsed_data.to_dict(orient='records')
            }
            self.metadata_storage.save_metadata(report, storage_type='threat_intelligence')

            # Optionally, perform additional analysis or trigger alerts
            self.analyze_and_alert(report, source)

    def analyze_and_alert(self, report: Dict[str, Any], source: Dict[str, Any]):
        """
        Analyzes the fetched threat intelligence data and sends alerts if necessary.

        :param report: Dictionary containing fetched data
        :param source: Dictionary containing source details
        """
        # Placeholder for analysis logic
        # Example: Check for high-severity vulnerabilities
        try:
            if source.get('name') == 'MITRE ATT&CK':
                # Example: Identify techniques related to a specific threat actor
                techniques = report.get('data', [])
                critical_techniques = [tech for tech in techniques if tech.get('kill_chain_phases')]

                if critical_techniques:
                    subject = f"Threat Intelligence Alert from {source.get('name')}"
                    message = f"Detected {len(critical_techniques)} critical techniques.\n\nDetails:\n{json.dumps(critical_techniques, indent=2)}"
                    self.notification_manager.send_email(subject, message)
                    logger.info(f"Sent threat intelligence alert based on data from '{source.get('name')}'.")
        except Exception as e:
            logger.error(f"Failed to analyze threat intelligence data from '{source.get('name')}': {e}")

    def run(self):
        """
        Starts the threat intelligence fetching process, running at configured intervals.
        """
        logger.info("Starting ThreatIntelligenceFetcher.")
        while True:
            for source in self.sources:
                self.process_source(source)
                # Optional: Delay between processing sources to manage API rate limits
                time.sleep(2)
            logger.info(f"Sleeping for {self.fetch_interval} seconds before next fetch cycle.")
            time.sleep(self.fetch_interval)


if __name__ == "__main__":
    try:
        fetcher = ThreatIntelligenceFetcher()
        fetcher.run()
    except KeyboardInterrupt:
        logger.info("ThreatIntelligenceFetcher stopped manually.")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
