import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class LearningModel:
    def __init__(self):
        """Initialize the model."""
        self.model = RandomForestClassifier()
        self.performance_data = []

    def log_performance(self, success_rate):
        """Log performance data for training."""
        self.performance_data.append(success_rate)

    def suggest_optimization(self):
        """Suggest optimizations based on performance data."""
        if len(self.performance_data) > 10:
            return 'Refactor code.'
        return 'No optimization needed.'

    def train_model(self, data, labels, model_path='ml_model.pkl'):
        """Train the model and save it."""
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)
        self.model.fit(X_train, y_train)
        accuracy = self.model.score(X_test, y_test)
        print(f"Model trained with accuracy: {accuracy}")
        self.save_model(model_path)
        return accuracy

    def save_model(self, model_path):
        """Save the trained model to a file."""
        joblib.dump(self.model, model_path)
        print(f"Model saved at {model_path}")

    def load_model(self, model_path):
        """Load a pre-trained model."""
        self.model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")

    def predict(self, input_data):
        """Make a prediction using the trained model."""
        return self.model.predict(np.array([input_data]))
