import pandas as pd
import numpy as np
import os
import json

class DataLoader:
    def __init__(self, source_type, source_path):
        self.source_type = source_type
        self.source_path = source_path

    def load_data(self):
        if self.source_type == 'csv':
            return pd.read_csv(self.source_path)
        elif self.source_type == 'excel':
            return pd.read_excel(self.source_path)
        elif self.source_type == 'json':
            with open(self.source_path) as f:
                return pd.json_normalize(json.load(f))
        elif self.source_type == 'sql':
            from sqlalchemy import create_engine
            engine = create_engine(self.source_path)
            return pd.read_sql('SELECT * FROM your_table', con=engine)
        else:
            raise ValueError("Unsupported source type")

    def preprocess_data(self, df, drop_columns=None, fill_na=None, normalize=False):
        if drop_columns:
            df = df.drop(columns=drop_columns)
        if fill_na:
            df = df.fillna(fill_na)
        if normalize:
            df = (df - df.mean()) / df.std()
        return df

    def load_and_preprocess(self, drop_columns=None, fill_na=None, normalize=False):
        df = self.load_data()
        return self.preprocess_data(df, drop_columns, fill_na, normalize)