import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class StatisticalAnalysis:
    def __init__(self, data):
        self.data = data

    def descriptive_statistics(self):
        return self.data.describe()

    def correlation_matrix(self):
        return self.data.corr()

    def plot_correlation_matrix(self):
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.data.corr(), annot=True, fmt=".2f", cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.show()

class LinearModel:
    def __init__(self, data, target_variable):
        self.data = data
        self.target_variable = target_variable
        self.model = None

    def prepare_data(self):
        X = self.data.drop(columns=[self.target_variable])
        y = self.data[self.target_variable]
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def fit_model(self):
        X_train, X_test, y_train, y_test = self.prepare_data()
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        return mse, r2

    def summary(self):
        X = sm.add_constant(self.data.drop(columns=[self.target_variable]))
        y = self.data[self.target_variable]
        model = sm.OLS(y, X).fit()
        return model.summary()

# Example usage:
# df = pd.read_csv('your_data.csv')
# analysis = StatisticalAnalysis(df)
# print(analysis.descriptive_statistics())
# analysis.plot_correlation_matrix()
# model = LinearModel(df, 'target_column')
# mse, r2 = model.fit_model()
# print(f'MSE: {mse}, R2: {r2}')
# print(model.summary())