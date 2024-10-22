#!/bin/bash

# =====================================================
# Backup Script for Hermod AI Assistant Framework
# =====================================================
#
# Function: Automates the backup of the PostgreSQL database,
#           configuration files, and important project files.
#
# Usage: ./backup.sh
#
# Schedule: Can be scheduled via cron for regular backups.
#
# =====================================================

# ----------------------------
# Configuration Variables
# ----------------------------

# Timestamp for backup files
TIMESTAMP=$(date +"%F_%T")

# Backup directories
BACKUP_DIR="/path/to/your/backups"  # Change to your desired backup directory
DB_BACKUP_DIR="$BACKUP_DIR/database"
CONFIG_BACKUP_DIR="$BACKUP_DIR/configurations"
FILES_BACKUP_DIR="$BACKUP_DIR/files"

# PostgreSQL Database Credentials
DB_NAME="your_database_name"        # Replace with your database name
DB_USER="your_database_user"        # Replace with your database user
DB_HOST="localhost"                 # Database host
DB_PORT="5432"                      # Database port (default for PostgreSQL)

# Paths to Configuration and Important Files
CONFIG_DIR="/path/to/your/configs"  # Directory containing configuration files
PROJECT_DIR="/path/to/your/project" # Directory containing important project files

# Retention Policy
RETENTION_DAYS=7  # Number of days to keep backups

# Log File
LOG_FILE="$BACKUP_DIR/backup.log"

# ----------------------------
# Function Definitions
# ----------------------------

# Function to create directories if they don't exist
create_directories() {
    mkdir -p "$DB_BACKUP_DIR"
    mkdir -p "$CONFIG_BACKUP_DIR"
    mkdir -p "$FILES_BACKUP_DIR"
}

# Function to backup PostgreSQL database
backup_database() {
    echo "[$(date +"%F %T")] Starting database backup..." | tee -a "$LOG_FILE"
    pg_dump -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" "$DB_NAME" > "$DB_BACKUP_DIR/${DB_NAME}_$TIMESTAMP.sql"

    if [ $? -eq 0 ]; then
        echo "[$(date +"%F %T")] Database backup successful." | tee -a "$LOG_FILE"
    else
        echo "[$(date +"%F %T")] Database backup failed!" | tee -a "$LOG_FILE"
        exit 1
    fi
}

# Function to backup configuration files
backup_configurations() {
    echo "[$(date +"%F %T")] Starting configuration files backup..." | tee -a "$LOG_FILE"
    tar -czf "$CONFIG_BACKUP_DIR/configs_$TIMESTAMP.tar.gz" -C "$CONFIG_DIR" .

    if [ $? -eq 0 ]; then
        echo "[$(date +"%F %T")] Configuration files backup successful." | tee -a "$LOG_FILE"
    else
        echo "[$(date +"%F %T")] Configuration files backup failed!" | tee -a "$LOG_FILE"
        exit 1
    fi
}

# Function to backup important project files
backup_files() {
    echo "[$(date +"%F %T")] Starting project files backup..." | tee -a "$LOG_FILE"
    tar -czf "$FILES_BACKUP_DIR/files_$TIMESTAMP.tar.gz" -C "$PROJECT_DIR" .

    if [ $? -eq 0 ]; then
        echo "[$(date +"%F %T")] Project files backup successful." | tee -a "$LOG_FILE"
    else
        echo "[$(date +"%F %T")] Project files backup failed!" | tee -a "$LOG_FILE"
        exit 1
    fi
}

# Function to enforce retention policy
enforce_retention_policy() {
    echo "[$(date +"%F %T")] Enforcing retention policy: Keeping backups for the last $RETENTION_DAYS days." | tee -a "$LOG_FILE"
    find "$BACKUP_DIR" -type f -mtime +$RETENTION_DAYS -exec rm {} \;

    if [ $? -eq 0 ]; then
        echo "[$(date +"%F %T")] Retention policy enforced successfully." | tee -a "$LOG_FILE"
    else
        echo "[$(date +"%F %T")] Retention policy enforcement failed!" | tee -a "$LOG_FILE"
        exit 1
    fi
}

# Function to log completion
log_completion() {
    echo "[$(date +"%F %T")] Backup process completed successfully." | tee -a "$LOG_FILE"
}

# ----------------------------
# Main Execution Flow
# ----------------------------

# Step 1: Create necessary directories
create_directories

# Step 2: Backup database
backup_database

# Step 3: Backup configuration files
backup_configurations

# Step 4: Backup important project files
backup_files

# Step 5: Enforce retention policy
enforce_retention_policy

# Step 6: Log completion
log_completion

exit 0
