import pandas as pd

class DataCleaner:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def drop_missing(self, threshold=0.5):
        self.dataframe = self.dataframe.dropna(thresh=int(threshold * len(self.dataframe)))

    def fill_missing(self, method='mean'):
        for column in self.dataframe.select_dtypes(include=['float64', 'int64']).columns:
            if method == 'mean':
                self.dataframe[column].fillna(self.dataframe[column].mean(), inplace=True)
            elif method == 'median':
                self.dataframe[column].fillna(self.dataframe[column].median(), inplace=True)
            elif method == 'mode':
                self.dataframe[column].fillna(self.dataframe[column].mode()[0], inplace=True)

    def remove_duplicates(self):
        self.dataframe = self.dataframe.drop_duplicates()

    def convert_dtypes(self, conversions):
        for column, dtype in conversions.items():
            self.dataframe[column] = self.dataframe[column].astype(dtype)

    def normalize(self, columns):
        for column in columns:
            self.dataframe[column] = (self.dataframe[column] - self.dataframe[column].min()) / (self.dataframe[column].max() - self.dataframe[column].min())

    def get_cleaned_data(self):
        return self.dataframe