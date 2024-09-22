import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self, data):
        self.data = data

    def clean_data(self):
        self.data.dropna(inplace=True)
        self.data = self.data.reset_index(drop=True)

    def normalize_data(self, columns):
        for column in columns:
            self.data[column] = (self.data[column] - self.data[column].mean()) / self.data[column].std()

    def filter_data(self, column, threshold):
        self.data = self.data[self.data[column] > threshold]

    def aggregate_data(self, group_by_column, agg_column, agg_func):
        return self.data.groupby(group_by_column)[agg_column].agg(agg_func).reset_index()

    def save_to_csv(self, filename):
        self.data.to_csv(filename, index=False)

# Example usage:
# df = pd.read_csv('data.csv')
# processor = DataProcessor(df)
# processor.clean_data()
# processor.normalize_data(['column1', 'column2'])
# processor.filter_data('column3', 10)
# aggregated = processor.aggregate_data('column4', 'column5', 'sum')
# processor.save_to_csv('processed_data.csv')