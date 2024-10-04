from sklearn.ensemble import RandomForestClassifier
import numpy as np


class LocalModel:

    def __init__(self):
        self.model = RandomForestClassifier()
        self.data = []
        self.labels = []

    def train(self, data, labels):
        self.model.fit(data, labels)

    def predict(self, input_data):
        return self.model.predict(np.array([input_data]))


def refine_gpt_prompt(current_prompt, feedback):
    if 'slow performance' in feedback:
        return current_prompt + ' Optimize for performance.'
    return current_prompt
