import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class EDA:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def summary_statistics(self):
        return self.dataframe.describe()

    def missing_values(self):
        return self.dataframe.isnull().sum()

    def data_types(self):
        return self.dataframe.dtypes

    def correlation_matrix(self):
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.dataframe.corr(), annot=True, fmt=".2f", cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.show()

    def univariate_analysis(self, column):
        plt.figure(figsize=(10, 5))
        if self.dataframe[column].dtype == 'object':
            sns.countplot(y=column, data=self.dataframe)
        else:
            sns.histplot(self.dataframe[column], bins=30, kde=True)
        plt.title(f'Univariate Analysis of {column}')
        plt.show()

    def bivariate_analysis(self, column1, column2):
        plt.figure(figsize=(10, 5))
        sns.scatterplot(x=column1, y=column2, data=self.dataframe)
        plt.title(f'Bivariate Analysis between {column1} and {column2}')
        plt.show()

    def pairplot(self):
        sns.pairplot(self.dataframe)
        plt.title('Pairplot of the DataFrame')
        plt.show()

    def boxplot(self, column):
        plt.figure(figsize=(10, 5))
        sns.boxplot(y=self.dataframe[column])
        plt.title(f'Boxplot of {column}')
        plt.show()

    def categorical_analysis(self, categorical_column, numerical_column):
        plt.figure(figsize=(10, 5))
        sns.boxplot(x=categorical_column, y=numerical_column, data=self.dataframe)
        plt.title(f'Boxplot of {numerical_column} by {categorical_column}')
        plt.show()