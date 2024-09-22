import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class DataVisualizer:
    def __init__(self, data):
        self.data = data

    def line_plot(self, x, y, title='Line Plot', xlabel='X-axis', ylabel='Y-axis'):
        plt.figure(figsize=(10, 6))
        plt.plot(self.data[x], self.data[y], marker='o')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid()
        plt.show()

    def bar_plot(self, x, y, title='Bar Plot', xlabel='X-axis', ylabel='Y-axis'):
        plt.figure(figsize=(10, 6))
        plt.bar(self.data[x], self.data[y])
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(rotation=45)
        plt.show()

    def scatter_plot(self, x, y, title='Scatter Plot', xlabel='X-axis', ylabel='Y-axis'):
        plt.figure(figsize=(10, 6))
        plt.scatter(self.data[x], self.data[y])
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid()
        plt.show()

    def histogram(self, column, title='Histogram', xlabel='Value', ylabel='Frequency', bins=10):
        plt.figure(figsize=(10, 6))
        plt.hist(self.data[column], bins=bins, edgecolor='black')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid()
        plt.show()

    def box_plot(self, column, title='Box Plot', xlabel='Category', ylabel='Value'):
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.data[column])
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid()
        plt.show()

    def heatmap(self, correlation_matrix, title='Heatmap'):
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
        plt.title(title)
        plt.show()