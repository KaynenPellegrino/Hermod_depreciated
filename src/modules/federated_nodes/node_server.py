# federated_nodes/node_server.py

from flask import Flask, request, jsonify
import joblib
import json
import threading
import time
import os
import logging
from logging.handlers import RotatingFileHandler

app = Flask(__name__)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler = RotatingFileHandler('logs/node_server.log', maxBytes=10**6, backupCount=5)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)

# Load or initialize the local model
MODEL_PATH = 'model/local_model.joblib'
lock = threading.Lock()

if os.path.exists(MODEL_PATH):
    local_model = joblib.load(MODEL_PATH)
    logger.info("Local model loaded successfully.")
else:
    # Initialize a simple model if none exists
    from sklearn.ensemble import RandomForestClassifier
    local_model = RandomForestClassifier()
    # Assume some initial training here
    # X_initial, y_initial = load_local_data()
    # local_model.fit(X_initial, y_initial)
    joblib.dump(local_model, MODEL_PATH)
    logger.info("Initialized and saved new local model.")


@app.route('/tasks/train_model', methods=['POST'])
def train_model():
    data = request.get_json()
    logger.info(f"Received 'train_model' task with data: {data}")
    try:
        # Implement local training logic here
        # Example: Load local data, train the model, and save updates
        # X_train, y_train = load_local_data()
        # local_model.fit(X_train, y_train)

        # Simulate training time
        time.sleep(30)  # Replace with actual training code

        # Save the updated model
        with lock:
            joblib.dump(local_model, MODEL_PATH)
        logger.info("Model training completed and saved successfully.")
        return jsonify({'status': 'success'}), 200
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        return jsonify({'status': 'failure', 'error': str(e)}), 500


@app.route('/model/update', methods=['GET'])
def model_update():
    logger.info("Received request for model update.")
    try:
        with lock:
            model = joblib.load(MODEL_PATH)
            # Convert model parameters to JSON serializable format
            model_params = model.get_params()
        return jsonify({'model_parameters': model_params}), 200
    except Exception as e:
        logger.error(f"Error retrieving model update: {e}")
        return jsonify({'status': 'failure', 'error': str(e)}), 500


@app.route('/tasks/update_model', methods=['POST'])
def update_model():
    data = request.get_json()
    logger.info(f"Received 'update_model' task with data: {data}")
    try:
        new_params = data.get('model_parameters')
        if not new_params:
            logger.error("No model parameters provided for update.")
            return jsonify({'status': 'failure', 'error': 'No model parameters provided.'}), 400

        with lock:
            model = joblib.load(MODEL_PATH)
            model.set_params(**new_params)
            joblib.dump(model, MODEL_PATH)

        logger.info("Model updated successfully with new parameters.")
        return jsonify({'status': 'success'}), 200
    except Exception as e:
        logger.error(f"Error updating model: {e}")
        return jsonify({'status': 'failure', 'error': str(e)}), 500


@app.route('/status', methods=['GET'])
def status():
    logger.info("Received status check request.")
    try:
        return jsonify({'status': 'online', 'model_path': MODEL_PATH}), 200
    except Exception as e:
        logger.error(f"Error retrieving status: {e}")
        return jsonify({'status': 'failure', 'error': str(e)}), 500


if __name__ == '__main__':
    # Run the node server on a specified port
    import argparse

    parser = argparse.ArgumentParser(description='Federated Node Server')
    parser.add_argument('--port', type=int, default=5001, help='Port to run the node server on')
    args = parser.parse_args()

    # Ensure the model directory exists
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    app.run(host='0.0.0.0', port=args.port)
