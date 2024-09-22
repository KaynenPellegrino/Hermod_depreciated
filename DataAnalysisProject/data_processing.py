import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self, data):
        self.data = data

    def clean_data(self):
        self.data.dropna(inplace=True)
        self.data = self.data.reset_index(drop=True)

    def normalize_data(self):
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        self.data[numeric_cols] = (self.data[numeric_cols] - self.data[numeric_cols].mean()) / self.data[numeric_cols].std()

    def filter_data(self, condition):
        self.data = self.data.query(condition)

    def get_summary(self):
        return self.data.describe()

    def save_to_csv(self, filename):
        self.data.to_csv(filename, index=False)

# Example usage:
# df = pd.read_csv('data.csv')
# processor = DataProcessor(df)
# processor.clean_data()
# processor.normalize_data()
# processor.filter_data('column_name > value')
# summary = processor.get_summary()
# processor.save_to_csv('processed_data.csv')