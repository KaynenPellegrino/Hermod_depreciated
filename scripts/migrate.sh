#!/bin/bash

# =====================================================
# Database Migration Script for Hermod AI Assistant Framework
# =====================================================
#
# Function: Automates the execution of database migrations
#           to update schema changes, ensuring the database
#           is in sync with the application's models.
#
# Usage: ./migrate.sh [environment]
#        environment: development | staging | production
#
# Example:
#        ./migrate.sh production
#
# =====================================================

# ----------------------------
# Configuration Variables
# ----------------------------

# Git repository URL (if migrations are managed via Git)
REPO_URL="https://github.com/yourusername/hermod-ai-assistant.git"  # Replace with your repository URL

# Deployment directories
APP_DIR="/var/www/hermod-ai-assistant"       # Root directory of your application
VENV_DIR="$APP_DIR/venv"                     # Python virtual environment directory

# Alembic directory
ALEMBIC_DIR="$APP_DIR/alembic"

# Service name (systemd) - if needed to restart after migrations
SERVICE_NAME="hermod-ai-assistant"           # Replace with your systemd service name

# Git branch to deploy/migrate
BRANCH=$1                                     # Environment passed as first argument

# Default branch if not provided
if [ -z "$BRANCH" ]; then
    echo "No environment specified. Please provide one of the following: development | staging | production."
    exit 1
fi

# Environment configurations
case "$BRANCH" in
    development)
        ENV_FILE="$APP_DIR/.env.development"
        ;;
    staging)
        ENV_FILE="$APP_DIR/.env.staging"
        ;;
    production)
        ENV_FILE="$APP_DIR/.env.production"
        ;;
    *)
        echo "Invalid environment specified. Choose from: development | staging | production."
        exit 1
        ;;
esac

# Log file
LOG_FILE="$APP_DIR/migrate.log"

# ----------------------------
# Function Definitions
# ----------------------------

# Function to log messages
log() {
    echo "[$(date +"%F %T")] $1" | tee -a "$LOG_FILE"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to set up the virtual environment
setup_virtualenv() {
    if [ ! -d "$VENV_DIR" ]; then
        log "Creating virtual environment..."
        python3 -m venv "$VENV_DIR"
        if [ $? -ne 0 ]; then
            log "Failed to create virtual environment."
            exit 1
        fi
    else
        log "Virtual environment already exists."
    fi
}

# Function to activate the virtual environment and install dependencies
install_dependencies() {
    log "Activating virtual environment..."
    source "$VENV_DIR/bin/activate"
    if [ $? -ne 0 ]; then
        log "Failed to activate virtual environment."
        exit 1
    fi

    log "Installing dependencies..."
    pip install --upgrade pip
    pip install -r "$APP_DIR/requirements.txt"
    if [ $? -ne 0 ]; then
        log "Failed to install dependencies."
        deactivate
        exit 1
    fi

    deactivate
}

# Function to pull the latest code from the repository
pull_latest_code() {
    log "Pulling latest code from repository..."
    if [ ! -d "$APP_DIR/.git" ]; then
        log "Cloning repository..."
        git clone "$REPO_URL" "$APP_DIR"
        if [ $? -ne 0 ]; then
            log "Failed to clone repository."
            exit 1
        fi
    fi

    cd "$APP_DIR" || exit
    git fetch --all
    git reset --hard "origin/$BRANCH"
    git pull origin "$BRANCH"
    if [ $? -ne 0 ]; then
        log "Failed to pull latest code."
        exit 1
    fi
}

# Function to set environment variables
set_environment_variables() {
    log "Setting environment variables from $ENV_FILE..."
    if [ ! -f "$ENV_FILE" ]; then
        log "Environment file $ENV_FILE not found."
        exit 1
    fi
    export $(grep -v '^#' "$ENV_FILE" | xargs)
}

# Function to run Alembic migrations
run_migrations() {
    log "Running Alembic migrations..."
    source "$VENV_DIR/bin/activate"
    if [ $? -ne 0 ]; then
        log "Failed to activate virtual environment."
        exit 1
    fi

    alembic -c "$ALEMBIC_DIR/alembic.ini" upgrade head
    if [ $? -ne 0 ]; then
        log "Alembic migrations failed."
        deactivate
        exit 1
    fi

    deactivate
    log "Alembic migrations completed successfully."
}

# Function to restart the application service (if necessary)
restart_service() {
    if [ -n "$SERVICE_NAME" ]; then
        log "Restarting the application service ($SERVICE_NAME)..."
        sudo systemctl restart "$SERVICE_NAME"
        if [ $? -ne 0 ]; then
            log "Failed to restart the service."
            exit 1
        fi
        log "Service ($SERVICE_NAME) restarted successfully."
    fi
}

# Function to check service status
check_service_status() {
    if [ -n "$SERVICE_NAME" ]; then
        log "Checking the status of the service ($SERVICE_NAME)..."
        sudo systemctl status "$SERVICE_NAME" --no-pager
        if [ $? -ne 0 ]; then
            log "Service is not running correctly."
            exit 1
        fi
    fi
}

# Function to send migration success notification (Optional)
send_success_notification() {
    # Uncomment and configure if email notifications are set up
    # SUBJECT="Database Migration Successful: $BRANCH"
    # MESSAGE="Database migrations to '$BRANCH' environment completed successfully at $(date +"%F %T")."
    # echo "$MESSAGE" | mail -s "$SUBJECT" your_email@example.com
    :
}

# Function to send migration failure notification (Optional)
send_failure_notification() {
    # Uncomment and configure if email notifications are set up
    # SUBJECT="Database Migration Failed: $BRANCH"
    # MESSAGE="Database migrations to '$BRANCH' environment failed at $(date +"%F %T"). Check the migrate.log for details."
    # echo "$MESSAGE" | mail -s "$SUBJECT" your_email@example.com
    :
}

# ----------------------------
# Main Execution Flow
# ----------------------------

# Redirect all output to log file
exec >> "$LOG_FILE" 2>&1

log "========================================="
log "Starting database migration to '$BRANCH' environment."
log "========================================="

# Step 1: Pull latest code
pull_latest_code

# Step 2: Set environment variables
set_environment_variables

# Step 3: Set up virtual environment
setup_virtualenv

# Step 4: Install dependencies
install_dependencies

# Step 5: Run Alembic migrations
run_migrations

# Step 6: Restart the application service (if applicable)
restart_service

# Step 7: Check service status (if applicable)
check_service_status

# Step 8: Send success notification
send_success_notification

log "Database migration to '$BRANCH' environment completed successfully."

exit 0
