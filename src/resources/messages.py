# src/resources/messages.py

"""
messages.py

Function: User and System Messages
Purpose: Stores predefined messages displayed to users or logged by the system,
         supporting internationalization and localization if needed.
"""

from enum import Enum, auto


class UserMessages(Enum):
    """
    Enum for user-facing messages displayed in the GUI.
    """
    DASHBOARD_TITLE = "Hermod AI Assistant - Dashboard"
    PROJECT_STATS_TAB = "Project Statistics"
    SYSTEM_PERFORMANCE_TAB = "System Performance"
    ALERTS_TAB = "Alerts"

    EXPORT_SUCCESS = "Data exported successfully to {file_path}."
    EXPORT_FAILURE = "Failed to export data to {file_path}."
    EXPORT_NO_DATA = "No data available to export."

    NOTIFICATION_INFO_TITLE = "Information"
    NOTIFICATION_WARNING_TITLE = "Warning"
    NOTIFICATION_ERROR_TITLE = "Error"

    # Add more user-facing messages as needed


class SystemMessages(Enum):
    """
    Enum for system-facing messages used in logging or notifications.
    """
    INIT_SUCCESS = "Dashboard initialized successfully."
    INIT_FAILURE = "Failed to initialize the dashboard."

    UPDATE_SUCCESS = "Dashboard data updated successfully."
    UPDATE_FAILURE = "Failed to update dashboard data."

    EXPORT_STARTED = "Exporting {data_type} data to {file_path}."
    EXPORT_COMPLETED = "Exported {data_type} data to {file_path} successfully."
    EXPORT_FAILED = "Exporting {data_type} data to {file_path} failed."

    NOTIFICATION_SENT = "Notification sent successfully to recipients."
    NOTIFICATION_FAILED = "Failed to send notification to recipients."

    # Add more system-facing messages as needed
