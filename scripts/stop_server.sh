#!/bin/bash

# =====================================================
# Server Shutdown Script for Hermod AI Assistant Framework
# =====================================================
#
# Function: Automates the graceful termination of the application server,
#           ensuring all processes are terminated correctly, environment
#           variables are unset, and services are stopped via systemd.
#
# Usage: ./stop_server.sh [options]
#
# Options:
#   -e, --environment     Specify the environment (development | staging | production). Required.
#   -h, --help            Display help information.
#
# Examples:
#   ./stop_server.sh --environment production
#
# =====================================================

# ----------------------------
# Configuration Variables
# ----------------------------

# Default values
ENVIRONMENT=""
HELP=false

# Directories
APP_DIR="/var/www/hermod-ai-assistant"       # Root directory of your application
VENV_DIR="$APP_DIR/venv"                     # Python virtual environment directory

# Systemd Service
SERVICE_NAME="hermod-ai-assistant.service"    # Replace with your systemd service name

# Log File
LOG_FILE="$APP_DIR/stop_server.log"

# ----------------------------
# Function Definitions
# ----------------------------

# Function to display help
display_help() {
    echo "Usage: ./stop_server.sh [options]

Options:
  -e, --environment     Specify the environment (development | staging | production). Required.
  -h, --help            Display this help message.

Examples:
  ./stop_server.sh --environment production
"
}

# Function to log messages
log() {
    echo "[$(date +"%F %T")] $1" | tee -a "$LOG_FILE"
}

# Function to parse arguments
parse_arguments() {
    while [[ "$#" -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -h|--help)
                HELP=true
                shift
                ;;
            *)
                echo "Unknown parameter passed: $1"
                display_help
                exit 1
                ;;
        esac
    done

    if $HELP; then
        display_help
        exit 0
    fi

    if [ -z "$ENVIRONMENT" ]; then
        echo "Error: --environment is required."
        display_help
        exit 1
    fi

    if [[ "$ENVIRONMENT" != "development" && "$ENVIRONMENT" != "staging" && "$ENVIRONMENT" != "production" ]]; then
        echo "Error: Invalid environment specified. Choose from: development | staging | production."
        display_help
        exit 1
    fi
}

# Function to deactivate virtual environment
deactivate_virtualenv() {
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        deactivate
        log "Virtual environment deactivated."
    fi
}

# Function to unset environment variables
unset_environment_variables() {
    ENV_FILE="$APP_DIR/.env.$ENVIRONMENT"

    if [ -f "$ENV_FILE" ]; then
        log "Unsetting environment variables from $ENV_FILE..."
        # Unset all variables defined in the .env file
        export $(grep -v '^#' "$ENV_FILE" | xargs -d '\n' | sed 's/^\([^=]*\)=.*$/\1/')
        for var in $(grep -v '^#' "$ENV_FILE" | cut -d '=' -f 1); do
            unset "$var"
            log "Unset variable: $var"
        done
    else
        log "Environment file $ENV_FILE not found. Skipping environment variable unset."
    fi
}

# Function to stop the application service using systemd
stop_systemd_service() {
    log "Stopping the application service ($SERVICE_NAME) using systemd..."
    sudo systemctl stop "$SERVICE_NAME"
    if [ $? -ne 0 ]; then
        log "Failed to stop the service ($SERVICE_NAME)."
        exit 1
    fi
    log "Service ($SERVICE_NAME) stopped successfully."

    # Disable the service from starting on boot
    sudo systemctl disable "$SERVICE_NAME"
    if [ $? -ne 0 ]; then
        log "Failed to disable the service ($SERVICE_NAME) from starting on boot."
        exit 1
    fi
    log "Service ($SERVICE_NAME) disabled from starting on boot."
}

# Function to stop the application server manually (alternative to systemd)
stop_manually() {
    log "Stopping the application server manually..."

    # Find the Gunicorn process
    GUNICORN_PID=$(pgrep -f "gunicorn.*main:app")
    if [ -z "$GUNICORN_PID" ]; then
        log "Gunicorn process not found. Is the application running?"
    else
        kill "$GUNICORN_PID"
        if [ $? -ne 0 ]; then
            log "Failed to terminate Gunicorn process (PID: $GUNICORN_PID)."
            exit 1
        fi
        log "Gunicorn process (PID: $GUNICORN_PID) terminated successfully."
    fi
}

# Function to check service status
check_service_status() {
    log "Checking the status of the service ($SERVICE_NAME)..."
    sudo systemctl status "$SERVICE_NAME" --no-pager
    if [ $? -eq 0 ]; then
        log "Service ($SERVICE_NAME) is still running."
        exit 1
    fi
    log "Service ($SERVICE_NAME) is inactive and stopped."
}

# Function to send shutdown success notification (Optional)
send_success_notification() {
    # Uncomment and configure if email notifications are set up
    # SUBJECT="Server Shutdown Successful: $ENVIRONMENT"
    # MESSAGE="The server for '$ENVIRONMENT' environment has shut down successfully at $(date +"%F %T")."
    # echo "$MESSAGE" | mail -s "$SUBJECT" your_email@example.com
    :
}

# Function to send shutdown failure notification (Optional)
send_failure_notification() {
    # Uncomment and configure if email notifications are set up
    # SUBJECT="Server Shutdown Failed: $ENVIRONMENT"
    # MESSAGE="The server for '$ENVIRONMENT' environment failed to shut down at $(date +"%F %T"). Check the stop_server.log for details."
    # echo "$MESSAGE" | mail -s "$SUBJECT" your_email@example.com
    :
}

# Function to clean up
cleanup() {
    log "Cleaning up..."
    deactivate_virtualenv
    unset_environment_variables
    log "Cleanup completed."
}

# ----------------------------
# Main Execution Flow
# ----------------------------

# Parse command-line arguments
parse_arguments "$@"

# Redirect all output to log file
exec >> "$LOG_FILE" 2>&1

log "========================================="
log "Starting server shutdown for '$ENVIRONMENT' environment."
log "========================================="

# Step 1: Stop the application service using systemd
stop_systemd_service

# Alternatively, to stop the server manually without systemd, uncomment the following line:
# stop_manually

# Step 2: Check service status
check_service_status

# Step 3: Send success notification
send_success_notification

# Step 4: Cleanup
cleanup

log "Server shutdown for '$ENVIRONMENT' environment completed successfully."

exit 0
