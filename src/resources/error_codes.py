# src/resources/error_codes.py

"""
error_codes.py

Function: Error Code Definitions
Purpose: Enumerates error codes and messages used for exception handling, enabling consistent error reporting and handling.
"""

from enum import Enum


class ErrorCode(Enum):
    """
    Enum representing various error codes and their corresponding messages.
    """
    # General Errors
    UNKNOWN_ERROR = (1000, "An unknown error has occurred.")
    INVALID_INPUT = (1001, "Invalid input provided.")
    DATABASE_CONNECTION_FAILED = (1002, "Failed to connect to the database.")
    CONFIGURATION_LOAD_FAILED = (1003, "Failed to load configuration.")

    # Dashboard Errors
    DASHBOARD_INIT_FAILED = (2000, "Failed to initialize the dashboard.")
    DASHBOARD_UPDATE_FAILED = (2001, "Failed to update the dashboard data.")
    DASHBOARD_EXPORT_FAILED = (2002, "Failed to export dashboard data.")

    # Module-Specific Errors
    NOTIFICATION_SEND_FAILED = (3000, "Failed to send notification.")
    METRICS_COLLECTION_FAILED = (3001, "Failed to collect system metrics.")
    REALTIME_UPDATER_FAILED = (3002, "Real-time updater encountered an issue.")

    # File Handling Errors
    FILE_NOT_FOUND = (4000, "Requested file was not found.")
    FILE_READ_FAILED = (4001, "Failed to read the file.")
    FILE_WRITE_FAILED = (4002, "Failed to write to the file.")

    # Network Errors
    NETWORK_TIMEOUT = (5000, "Network operation timed out.")
    NETWORK_UNREACHABLE = (5001, "Network is unreachable.")
    API_REQUEST_FAILED = (5002, "API request failed.")

    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message

    def __str__(self):
        return f"[Error {self.code}] {self.message}"
