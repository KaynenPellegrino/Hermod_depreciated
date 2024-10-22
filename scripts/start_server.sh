#!/bin/bash

# =====================================================
# Server Startup Script for Hermod AI Assistant Framework
# =====================================================
#
# Function: Automates the initiation of the application server,
#           setting necessary environment variables, activating
#           the Python virtual environment, and starting the
#           application using Gunicorn with Uvicorn workers.
#
# Usage: ./start_server.sh [options]
#
# Options:
#   -e, --environment     Specify the environment (development | staging | production). Required.
#   -h, --help            Display help information.
#
# Examples:
#   ./start_server.sh --environment production
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
LOG_FILE="$APP_DIR/start_server.log"

# ----------------------------
# Function Definitions
# ----------------------------

# Function to display help
display_help() {
    echo "Usage: ./start_server.sh [options]

Options:
  -e, --environment     Specify the environment (development | staging | production). Required.
  -h, --help            Display this help message.

Examples:
  ./start_server.sh --environment production
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

# Function to activate virtual environment
activate_virtualenv() {
    if [ ! -d "$VENV_DIR" ]; then
        log "Virtual environment not found at $VENV_DIR. Creating one..."
        python3 -m venv "$VENV_DIR"
        if [ $? -ne 0 ]; then
            log "Failed to create virtual environment."
            exit 1
        fi
    fi

    log "Activating virtual environment..."
    source "$VENV_DIR/bin/activate"
    if [ $? -ne 0 ]; then
        log "Failed to activate virtual environment."
        exit 1
    fi
    log "Virtual environment activated."
}

# Function to set environment variables
set_environment_variables() {
    ENV_FILE="$APP_DIR/.env.$ENVIRONMENT"

    if [ ! -f "$ENV_FILE" ]; then
        log "Environment file $ENV_FILE not found."
        deactivate_virtualenv
        exit 1
    fi

    log "Loading environment variables from $ENV_FILE..."
    export $(grep -v '^#' "$ENV_FILE" | xargs)
    if [ $? -ne 0 ]; then
        log "Failed to load environment variables."
        deactivate_virtualenv
        exit 1
    fi
    log "Environment variables loaded."
}

# Function to deactivate virtual environment
deactivate_virtualenv() {
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        deactivate
        log "Virtual environment deactivated."
    fi
}

# Function to start the application server using systemd
start_systemd_service() {
    log "Starting the application service ($SERVICE_NAME) using systemd..."
    sudo systemctl start "$SERVICE_NAME"
    if [ $? -ne 0 ]; then
        log "Failed to start the service ($SERVICE_NAME)."
        exit 1
    fi
    log "Service ($SERVICE_NAME) started successfully."

    # Enable the service to start on boot
    sudo systemctl enable "$SERVICE_NAME"
    if [ $? -ne 0 ]; then
        log "Failed to enable the service ($SERVICE_NAME) to start on boot."
        exit 1
    fi
    log "Service ($SERVICE_NAME) enabled to start on boot."
}

# Function to start the application server manually (alternative to systemd)
start_manually() {
    log "Starting the application server manually..."

    # Navigate to the application directory
    cd "$APP_DIR" || { log "Application directory not found."; exit 1; }

    # Start Gunicorn with Uvicorn workers
    gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:8000 --daemon
    if [ $? -ne 0 ]; then
        log "Failed to start the application server."
        deactivate_virtualenv
        exit 1
    fi
    log "Application server started successfully on port 8000."
}

# Function to check service status
check_service_status() {
    log "Checking the status of the service ($SERVICE_NAME)..."
    sudo systemctl status "$SERVICE_NAME" --no-pager
    if [ $? -ne 0 ]; then
        log "Service ($SERVICE_NAME) is not running correctly."
        exit 1
    fi
    log "Service ($SERVICE_NAME) is active and running."
}

# Function to send startup success notification (Optional)
send_success_notification() {
    # Uncomment and configure if email notifications are set up
    # SUBJECT="Server Startup Successful: $ENVIRONMENT"
    # MESSAGE="The server for '$ENVIRONMENT' environment has started successfully at $(date +"%F %T")."
    # echo "$MESSAGE" | mail -s "$SUBJECT" your_email@example.com
    :
}

# Function to send startup failure notification (Optional)
send_failure_notification() {
    # Uncomment and configure if email notifications are set up
    # SUBJECT="Server Startup Failed: $ENVIRONMENT"
    # MESSAGE="The server for '$ENVIRONMENT' environment failed to start at $(date +"%F %T"). Check the start_server.log for details."
    # echo "$MESSAGE" | mail -s "$SUBJECT" your_email@example.com
    :
}

# Function to clean up
cleanup() {
    log "Cleaning up..."
    deactivate_virtualenv
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
log "Starting server for '$ENVIRONMENT' environment."
log "========================================="

# Step 1: Activate virtual environment
activate_virtualenv

# Step 2: Set environment variables
set_environment_variables

# Step 3: Start the application server using systemd
start_systemd_service

# Alternatively, to start the server manually without systemd, uncomment the following line:
# start_manually

# Step 4: Check service status
check_service_status

# Step 5: Send success notification
send_success_notification

# Step 6: Cleanup
cleanup

log "Server startup for '$ENVIRONMENT' environment completed successfully."

exit 0
