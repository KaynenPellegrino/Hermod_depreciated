import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from hermod.utils.logger import setup_logger
import random

# Initialize logger
logger = setup_logger()

class AIRefactorSystem:
    """
    AI-based system to predict whether refactoring will improve performance and manage Q-learning for decision making.
    """
    def __init__(self, model_path='models/refactor_ai_model.pkl', actions=['refactor', 'do_nothing'], epsilon=1.0, learning_rate=0.1, discount_factor=0.95, decay=0.99):
        self.model_path = model_path
        self.model = self.load_or_initialize_model()
        self.q_table = {}
        self.actions = actions
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.decay = decay

    def load_or_initialize_model(self):
        if os.path.exists(self.model_path):
            logger.info("Loaded existing AI model for refactoring.")
            return joblib.load(self.model_path)
        else:
            logger.info("Initialized a new AI model for refactoring.")
            return RandomForestClassifier()

    def choose_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = {action: 0 for action in self.actions}
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        return max(self.q_table[state], key=self.q_table[state].get)

    def update_q_table(self, state, action, reward, next_state):
        if next_state not in self.q_table:
            self.q_table[next_state] = {action: 0 for action in self.actions}
        old_q_value = self.q_table[state][action]
        future_optimal_value = max(self.q_table[next_state].values())
        new_q_value = old_q_value + self.learning_rate * (reward + self.discount_factor * future_optimal_value - old_q_value)
        self.q_table[state][action] = new_q_value

    def give_reward(self, test_passed, performance_improved):
        reward = 0
        if test_passed:
            reward += 10
        else:
            reward -= 20
        if performance_improved:
            reward += 5
        else:
            reward -= 5
        return reward

    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        joblib.dump(self.model, self.model_path)

    def predict_refactor(self, module_features):
        return self.model.predict(module_features)
