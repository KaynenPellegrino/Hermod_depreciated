# gcp/cloud_functions/function1/main.py

import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from flask import Flask, request, jsonify

app = Flask(__name__)

# Database connection parameters from environment variables
DB_USER = os.getenv('DB_USER')  # e.g., 'hermod_user'
DB_PASSWORD = os.getenv('DB_PASSWORD')  # e.g., 'your_secure_password'
DB_NAME = os.getenv('DB_NAME')  # e.g., 'hermod_db'
DB_HOST = os.getenv('DB_HOST')  # e.g., '35.223.123.456' or 'localhost' if using Unix socket
DB_PORT = os.getenv('DB_PORT', '5432')  # Default PostgreSQL port

# Establish a connection pool or a single connection
def get_db_connection():
    try:
        conn = psycopg2.connect(
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME
        )
        return conn
    except Exception as e:
        app.logger.error(f"Database connection failed: {e}")
        return None

@app.route('/hermod', methods=['POST'])
def hermod_handler():
    """
    HTTP Cloud Function that handles incoming requests to create new projects.
    Expects a JSON payload with project details.
    """
    try:
        # Parse JSON request
        request_json = request.get_json()
        if not request_json:
            return jsonify({"error": "Invalid JSON payload"}), 400

        # Extract required fields
        username = request_json.get('username')
        email = request_json.get('email')
        project_name = request_json.get('project_name')
        description = request_json.get('description')
        programming_language = request_json.get('programming_language')

        # Validate required fields
        if not all([username, email, project_name, programming_language]):
            return jsonify({"error": "Missing required fields"}), 400

        # Connect to the database
        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Database connection failed"}), 500

        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Check if user exists; if not, create user
        cursor.execute("""
            SELECT user_id FROM users WHERE email = %s
        """, (email,))
        user = cursor.fetchone()

        if not user:
            # Insert new user
            cursor.execute("""
                INSERT INTO users (username, email, password_hash, role)
                VALUES (%s, %s, %s, %s)
                RETURNING user_id
            """, (username, email, 'hashed_password_placeholder', 'user'))  # Replace 'hashed_password_placeholder' as needed
            user_id = cursor.fetchone()['user_id']
            conn.commit()
            app.logger.info(f"Created new user with ID: {user_id}")
        else:
            user_id = user['user_id']
            app.logger.info(f"Existing user found with ID: {user_id}")

        # Insert new project
        cursor.execute("""
            INSERT INTO projects (user_id, project_name, description, programming_language, status)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING project_id, status, created_at
        """, (user_id, project_name, description, programming_language, 'pending'))
        project = cursor.fetchone()
        conn.commit()

        # Close database connection
        cursor.close()
        conn.close()

        # Return success response
        return jsonify({
            "message": "Project created successfully",
            "project_id": project['project_id'],
            "status": project['status'],
            "created_at": project['created_at'].isoformat()
        }), 201

    except Exception as e:
        app.logger.error(f"Error handling request: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

# For local testing purposes only
if __name__ == '__main__':
    app.run(debug=True)
