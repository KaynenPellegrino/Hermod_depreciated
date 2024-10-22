#!/bin/bash

# =====================================================
# Project Packaging Script for Hermod AI Assistant Framework
# =====================================================
#
# Function: Automates the packaging of the Hermod AI Assistant Framework
#           into Docker images and compressed archives (.tar.gz, .zip)
#           for deployment and distribution.
#
# Usage: ./package_project.sh [options]
#
# Options:
#   -d, --docker          Package the project into a Docker image.
#   -a, --archive         Package the project into compressed archives.
#   -e, --environment     Specify the environment (development | staging | production). Required for Docker packaging.
#   -h, --help            Display help information.
#
# Examples:
#   ./package_project.sh --docker --environment production
#   ./package_project.sh --archive
#
# =====================================================

# ----------------------------
# Configuration Variables
# ----------------------------

# Default values
PACKAGE_DOCKER=false
PACKAGE_ARCHIVE=false
ENVIRONMENT=""
HELP=false

# Directories
APP_DIR="/var/www/hermod-ai-assistant"       # Root directory of your application
DIST_DIR="$APP_DIR/dist"                     # Directory to store packaged artifacts
VENV_DIR="$APP_DIR/venv"                     # Python virtual environment directory

# Docker Variables
DOCKER_IMAGE_NAME="hermod-ai-assistant"
DOCKER_TAG="latest"

# Archive Variables
ARCHIVE_FORMAT="tar.gz"                       # Options: tar.gz, zip

# Log File
LOG_FILE="$APP_DIR/package_project.log"

# ----------------------------
# Function Definitions
# ----------------------------

# Function to display help
display_help() {
    echo "Usage: ./package_project.sh [options]

Options:
  -d, --docker          Package the project into a Docker image.
  -a, --archive         Package the project into compressed archives.
  -e, --environment     Specify the environment (development | staging | production). Required for Docker packaging.
  -h, --help            Display this help message.

Examples:
  ./package_project.sh --docker --environment production
  ./package_project.sh --archive
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
            -d|--docker)
                PACKAGE_DOCKER=true
                shift
                ;;
            -a|--archive)
                PACKAGE_ARCHIVE=true
                shift
                ;;
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

    if $PACKAGE_DOCKER && [[ -z "$ENVIRONMENT" ]]; then
        echo "Error: --environment is required when packaging Docker images."
        display_help
        exit 1
    fi

    if ! $PACKAGE_DOCKER && ! $PACKAGE_ARCHIVE; then
        echo "Error: At least one packaging option must be specified (--docker or --archive)."
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

# Function to pull latest code
pull_latest_code() {
    log "Pulling latest code from repository..."
    cd "$APP_DIR" || { log "Application directory not found."; exit 1; }

    if [ -d ".git" ]; then
        git fetch --all
        git reset --hard "origin/$ENVIRONMENT"
        git pull origin "$ENVIRONMENT"
        if [ $? -ne 0 ]; then
            log "Failed to pull latest code."
            exit 1
        fi
    else
        log "Git repository not found in $APP_DIR. Skipping code pull."
    fi
}

# Function to create distribution directory
create_dist_dir() {
    if [ ! -d "$DIST_DIR" ]; then
        mkdir -p "$DIST_DIR"
        if [ $? -ne 0 ]; then
            log "Failed to create distribution directory at $DIST_DIR."
            exit 1
        fi
    fi
}

# Function to package as Docker image
package_docker() {
    log "Packaging project into Docker image..."
    cd "$APP_DIR" || { log "Application directory not found."; exit 1; }

    # Tag Docker image based on environment
    case "$ENVIRONMENT" in
        development)
            DOCKER_TAG="dev"
            ;;
        staging)
            DOCKER_TAG="staging"
            ;;
        production)
            DOCKER_TAG="latest"
            ;;
        *)
            log "Unknown environment: $ENVIRONMENT. Using 'latest' tag."
            DOCKER_TAG="latest"
            ;;
    esac

    # Build Docker image
    docker build -t "$DOCKER_IMAGE_NAME:$DOCKER_TAG" .
    if [ $? -ne 0 ]; then
        log "Docker image build failed."
        exit 1
    fi

    # Save Docker image as tar.gz archive
    docker save "$DOCKER_IMAGE_NAME:$DOCKER_TAG" | gzip > "$DIST_DIR/${DOCKER_IMAGE_NAME}_${DOCKER_TAG}_$(date +"%F_%T").tar.gz"
    if [ $? -ne 0 ]; then
        log "Failed to save Docker image as archive."
        exit 1
    fi

    log "Docker image packaged successfully: $DIST_DIR/${DOCKER_IMAGE_NAME}_${DOCKER_TAG}_$(date +"%F_%T").tar.gz"
}

# Function to package as compressed archive
package_archive() {
    log "Packaging project into compressed archive ($ARCHIVE_FORMAT)..."

    # Define archive name
    ARCHIVE_NAME="hermod-ai-assistant_$(date +"%F_%T").$ARCHIVE_FORMAT"

    case "$ARCHIVE_FORMAT" in
        tar.gz)
            tar -czf "$DIST_DIR/$ARCHIVE_NAME" -C "$APP_DIR" .
            if [ $? -ne 0 ]; then
                log "Failed to create tar.gz archive."
                exit 1
            fi
            ;;
        zip)
            zip -r "$DIST_DIR/$ARCHIVE_NAME" "$APP_DIR" -x "*.git*" "*.env*" "dist/*" "*.log"
            if [ $? -ne 0 ]; then
                log "Failed to create zip archive."
                exit 1
            fi
            ;;
        *)
            log "Unsupported archive format: $ARCHIVE_FORMAT. Supported formats: tar.gz, zip."
            exit 1
            ;;
    esac

    log "Archive packaged successfully: $DIST_DIR/$ARCHIVE_NAME"
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
log "Starting project packaging process."
log "========================================="

# Step 1: Pull latest code
if $PACKAGE_DOCKER || $PACKAGE_ARCHIVE; then
    pull_latest_code
fi

# Step 2: Setup virtual environment
if $PACKAGE_DOCKER || $PACKAGE_ARCHIVE; then
    setup_virtualenv
fi

# Step 3: Install dependencies
if $PACKAGE_DOCKER || $PACKAGE_ARCHIVE; then
    install_dependencies
fi

# Step 4: Create distribution directory
create_dist_dir

# Step 5: Package into Docker image (if requested)
if $PACKAGE_DOCKER; then
    package_docker
fi

# Step 6: Package into compressed archive (if requested)
if $PACKAGE_ARCHIVE; then
    package_archive
fi

# Step 7: Cleanup
cleanup

log "Project packaging process completed successfully."

exit 0
