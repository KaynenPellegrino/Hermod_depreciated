#!/bin/bash

# =====================================================
# Data Restoration Script for Hermod AI Assistant Framework
# =====================================================
#
# Function: Automates the restoration of data from backups,
#           including databases, configuration files, and
#           project files. Facilitates recovery scenarios
#           and setting up new environments.
#
# Usage: ./restore.sh [options]
#
# Options:
#   -d, --database        Restore the PostgreSQL database from backup.
#   -c, --configurations  Restore configuration files from backup.
#   -f, --files           Restore project files from backup.
#   -a, --all             Restore all components (database, configurations, files).
#   -e, --environment     Specify the environment (development | staging | production). Required for restoring.
#   -b, --backup-dir      Specify the backup directory. Defaults to "/var/www/hermod-ai-assistant/backups".
#   -h, --help            Display help information.
#
# Examples:
#   ./restore.sh --database --environment production
#   ./restore.sh --all --environment staging --backup-dir /path/to/backups
#
# =====================================================

# ----------------------------
# Configuration Variables
# ----------------------------

# Default values
RESTORE_DATABASE=false
RESTORE_CONFIG=false
RESTORE_FILES=false
RESTORE_ALL=false
ENVIRONMENT=""
BACKUP_DIR="/var/www/hermod-ai-assistant/backups"
HELP=false

# Directories
APP_DIR="/var/www/hermod-ai-assistant"       # Root directory of your application
VENV_DIR="$APP_DIR/venv"                     # Python virtual environment directory

# Database Variables
DB_NAME=""
DB_USER=""
DB_HOST="localhost"
DB_PORT="5432"

# Encryption Variables (if backups are encrypted)
ENCRYPTION_PASSWORD=""
ENCRYPTION_ALGORITHM="aes-256-cbc"

# Log File
LOG_FILE="$APP_DIR/restore.log"

# ----------------------------
# Function Definitions
# ----------------------------

# Function to display help
display_help() {
    echo "Usage: ./restore.sh [options]

Options:
  -d, --database        Restore the PostgreSQL database from backup.
  -c, --configurations  Restore configuration files from backup.
  -f, --files           Restore project files from backup.
  -a, --all             Restore all components (database, configurations, files).
  -e, --environment     Specify the environment (development | staging | production). Required for restoring.
  -b, --backup-dir      Specify the backup directory. Defaults to \"/var/www/hermod-ai-assistant/backups\".
  -h, --help            Display this help message.

Examples:
  ./restore.sh --database --environment production
  ./restore.sh --all --environment staging --backup-dir /path/to/backups
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
            -d|--database)
                RESTORE_DATABASE=true
                shift
                ;;
            -c|--configurations)
                RESTORE_CONFIG=true
                shift
                ;;
            -f|--files)
                RESTORE_FILES=true
                shift
                ;;
            -a|--all)
                RESTORE_ALL=true
                shift
                ;;
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -b|--backup-dir)
                BACKUP_DIR="$2"
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

    if $RESTORE_ALL; then
        RESTORE_DATABASE=true
        RESTORE_CONFIG=true
        RESTORE_FILES=true
    fi

    if ! $RESTORE_DATABASE && ! $RESTORE_CONFIG && ! $RESTORE_FILES; then
        echo "Error: At least one restore option must be specified (--database, --configurations, --files, --all)."
        display_help
        exit 1
    fi
}

# Function to setup virtual environment
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

# Function to activate virtual environment and install dependencies
install_dependencies() {
    log "Activating virtual environment and installing dependencies..."
    source "$VENV_DIR/bin/activate"
    if [ $? -ne 0 ]; then
        log "Failed to activate virtual environment."
        exit 1
    fi

    pip install --upgrade pip
    pip install -r "$APP_DIR/requirements.txt"
    if [ $? -ne 0 ]; then
        log "Failed to install dependencies."
        deactivate
        exit 1
    fi

    deactivate
}

# Function to set environment-specific variables
set_environment_variables() {
    log "Setting environment variables for '$ENVIRONMENT' environment..."
    case "$ENVIRONMENT" in
        development)
            ENV_FILE="$APP_DIR/.env.development"
            DB_NAME="hermod_dev_db"
            DB_USER="hermod_dev_user"
            DB_HOST="localhost"
            DB_PORT="5432"
            ;;
        staging)
            ENV_FILE="$APP_DIR/.env.staging"
            DB_NAME="hermod_staging_db"
            DB_USER="hermod_staging_user"
            DB_HOST="localhost"
            DB_PORT="5432"
            ;;
        production)
            ENV_FILE="$APP_DIR/.env.production"
            DB_NAME="hermod_prod_db"
            DB_USER="hermod_prod_user"
            DB_HOST="localhost"
            DB_PORT="5432"
            ;;
        *)
            log "Invalid environment specified: $ENVIRONMENT"
            exit 1
            ;;
    esac

    if [ ! -f "$ENV_FILE" ]; then
        log "Environment file $ENV_FILE not found."
        exit 1
    fi

    export $(grep -v '^#' "$ENV_FILE" | xargs)
}

# Function to restore the database from backup
restore_database() {
    log "Starting database restoration..."

    # Define the database backup file
    DB_BACKUP_FILE=$(find "$BACKUP_DIR/database" -type f -name "${DB_NAME}_*.sql.gz.enc" | sort | tail -n 1)

    if [ -z "$DB_BACKUP_FILE" ]; then
        log "No database backup file found in $BACKUP_DIR/database."
        exit 1
    fi

    log "Found database backup file: $DB_BACKUP_FILE"

    # Decrypt and decompress the backup
    DECRYPTED_BACKUP_FILE="${DB_BACKUP_FILE%.enc}"
    openssl enc -d "$ENCRYPTION_ALGORITHM" -salt -k "$ENCRYPTION_PASSWORD" -in "$DB_BACKUP_FILE" | gunzip > "$DECRYPTED_BACKUP_FILE"
    if [ $? -ne 0 ]; then
        log "Failed to decrypt and decompress the database backup."
        exit 1
    fi
    log "Decrypted and decompressed database backup to $DECRYPTED_BACKUP_FILE"

    # Restore the database
    log "Restoring the PostgreSQL database..."
    psql -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -d "$DB_NAME" -f "$DECRYPTED_BACKUP_FILE"
    if [ $? -ne 0 ]; then
        log "Failed to restore the PostgreSQL database."
        exit 1
    fi
    log "PostgreSQL database restored successfully."

    # Remove the decrypted backup file
    rm "$DECRYPTED_BACKUP_FILE"
    log "Removed decrypted backup file: $DECRYPTED_BACKUP_FILE"
}

# Function to restore configuration files from backup
restore_configurations() {
    log "Starting configuration files restoration..."

    # Define the configuration backup file
    CONFIG_BACKUP_FILE=$(find "$BACKUP_DIR/configurations" -type f -name "configs_*.tar.gz" | sort | tail -n 1)

    if [ -z "$CONFIG_BACKUP_FILE" ]; then
        log "No configuration backup file found in $BACKUP_DIR/configurations."
        exit 1
    fi

    log "Found configuration backup file: $CONFIG_BACKUP_FILE"

    # Extract the configuration files
    tar -xzf "$CONFIG_BACKUP_FILE" -C "$APP_DIR"
    if [ $? -ne 0 ]; then
        log "Failed to extract configuration backup."
        exit 1
    fi
    log "Configuration files restored successfully from $CONFIG_BACKUP_FILE"
}

# Function to restore project files from backup
restore_files() {
    log "Starting project files restoration..."

    # Define the project files backup file
    FILES_BACKUP_FILE=$(find "$BACKUP_DIR/files" -type f -name "files_*.tar.gz" | sort | tail -n 1)

    if [ -z "$FILES_BACKUP_FILE" ]; then
        log "No project files backup file found in $BACKUP_DIR/files."
        exit 1
    fi

    log "Found project files backup file: $FILES_BACKUP_FILE"

    # Extract the project files
    tar -xzf "$FILES_BACKUP_FILE" -C "$APP_DIR"
    if [ $? -ne 0 ]; then
        log "Failed to extract project files backup."
        exit 1
    fi
    log "Project files restored successfully from $FILES_BACKUP_FILE"
}

# Function to restart the application service (if applicable)
restart_service() {
    SERVICE_NAME="hermod-ai-assistant"           # Replace with your systemd service name
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
    SERVICE_NAME="hermod-ai-assistant"           # Replace with your systemd service name
    if [ -n "$SERVICE_NAME" ]; then
        log "Checking the status of the service ($SERVICE_NAME)..."
        sudo systemctl status "$SERVICE_NAME" --no-pager
        if [ $? -ne 0 ]; then
            log "Service is not running correctly."
            exit 1
        fi
    fi
}

# Function to send restoration success notification (Optional)
send_success_notification() {
    # Uncomment and configure if email notifications are set up
    # SUBJECT="Data Restoration Successful: $ENVIRONMENT"
    # MESSAGE="Data restoration for '$ENVIRONMENT' environment completed successfully at $(date +"%F %T")."
    # echo "$MESSAGE" | mail -s "$SUBJECT" your_email@example.com
    :
}

# Function to send restoration failure notification (Optional)
send_failure_notification() {
    # Uncomment and configure if email notifications are set up
    # SUBJECT="Data Restoration Failed: $ENVIRONMENT"
    # MESSAGE="Data restoration for '$ENVIRONMENT' environment failed at $(date +"%F %T"). Check the restore.log for details."
    # echo "$MESSAGE" | mail -s "$SUBJECT" your_email@example.com
    :
}

# Function to clean up
cleanup() {
    log "Cleaning up temporary files..."
    # Add any cleanup commands if necessary
}

# ----------------------------
# Main Execution Flow
# ----------------------------

# Parse command-line arguments
parse_arguments "$@"

# Redirect all output to log file
exec >> "$LOG_FILE" 2>&1

log "========================================="
log "Starting data restoration for '$ENVIRONMENT' environment."
log "========================================="

# Step 1: Pull latest code (if applicable)
# Depending on your restoration strategy, you might want to pull the latest code
# Uncomment the following lines if necessary
# pull_latest_code

# Step 2: Setup virtual environment
setup_virtualenv

# Step 3: Install dependencies
install_dependencies

# Step 4: Set environment variables
set_environment_variables

# Step 5: Restore database (if requested)
if $RESTORE_DATABASE; then
    restore_database
fi

# Step 6: Restore configurations (if requested)
if $RESTORE_CONFIG; then
    restore_configurations
fi

# Step 7: Restore project files (if requested)
if $RESTORE_FILES; then
    restore_files
fi

# Step 8: Restart the application service (if applicable)
if $RESTORE_DATABASE || $RESTORE_CONFIG || $RESTORE_FILES; then
    restart_service
    check_service_status
fi

# Step 9: Send success notification
send_success_notification

# Step 10: Cleanup
cleanup

log "Data restoration for '$ENVIRONMENT' environment completed successfully."

exit 0
