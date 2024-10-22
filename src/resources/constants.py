# src/resources/constants.py

"""
constants.py

Function: Global Constants
Purpose: Defines constants used across the application, such as configuration keys,
         default values, fixed parameters, and other immutable settings.
"""

# ----------------------------
# Configuration Keys
# ----------------------------

# Database Configuration Keys
DB_DIALECT = 'dialect'
DB_USERNAME = 'username'
DB_PASSWORD = 'password'
DB_HOST = 'host'
DB_PORT = 'port'
DB_NAME = 'database'

# Dashboard Configuration Keys
DASHBOARD_HOST = 'DASHBOARD_HOST'
DASHBOARD_PORT = 'DASHBOARD_PORT'
DASHBOARD_TITLE = 'DASHBOARD_TITLE'
NOTIFICATION_RECIPIENTS = 'NOTIFICATION_RECIPIENTS'
ALERTS_FILE = 'ALERTS_FILE'
AVAILABLE_TOOLS = 'AVAILABLE_TOOLS'

# Export Configuration
EXPORT_DEFAULT_PROJECTS_FILE = 'project_statistics.csv'
EXPORT_DEFAULT_SYSTEM_FILE = 'system_metrics.csv'
EXPORT_DEFAULT_ALERTS_FILE = 'alerts.csv'

# ----------------------------
# Default Values
# ----------------------------

# Database Defaults
DEFAULT_DB_DIALECT = 'postgresql'
DEFAULT_DB_HOST = 'localhost'
DEFAULT_DB_PORT = 5432
DEFAULT_DB_NAME = 'hermod_ai_assistant'

# Dashboard Defaults
DEFAULT_DASHBOARD_HOST = '0.0.0.0'
DEFAULT_DASHBOARD_PORT = 5000
DEFAULT_DASHBOARD_TITLE = 'Hermod Dashboard'
DEFAULT_REFRESH_INTERVAL_MS = 5000  # 5 seconds
DEFAULT_NOTIFICATION_RECIPIENTS = []
DEFAULT_ALERTS_FILE = 'data/alerts.json'
DEFAULT_AVAILABLE_TOOLS = []

# ----------------------------
# Application Parameters
# ----------------------------

# Logger Configuration
LOG_DIR = 'logs'
DASHBOARD_LOG_FILE = 'dashboard_builder.log'
INTERACTIVE_DASHBOARD_LOG_FILE_PREFIX = 'interactive_dashboard_'

# ----------------------------
# UI Parameters
# ----------------------------

# Font Settings
FONT_FAMILY_TITLE = 'Arial'
FONT_SIZE_TITLE = 12
FONT_WEIGHT_TITLE = 'Bold'
FONT_SIZE_LABEL = 12

# Table Settings
TABLE_EDIT_TRIGGERS = 'NoEditTriggers'

# ----------------------------
# Notification Types
# ----------------------------

NOTIFICATION_INFO = 'info'
NOTIFICATION_WARNING = 'warning'
NOTIFICATION_ERROR = 'error'

# ----------------------------
# Export File Types
# ----------------------------

EXPORT_FILE_TYPE_CSV = 'CSV Files (*.csv)'
EXPORT_FILE_TYPE_ALL = 'All Files (*)'

# ----------------------------
# Other Constants
# ----------------------------

MAX_ALERTS_DISPLAY = 100  # Maximum number of alerts to display
