import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    return pd.read_csv(file_path)

def clean_data(df):
    df.dropna(inplace=True)
    return df

def analyze_data(df):
    summary = df.describe()
    correlation = df.corr()
    return summary, correlation

def visualize_data(df):
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()

def main(file_path):
    data = load_data(file_path)
    clean_data = clean_data(data)
    summary, correlation = analyze_data(clean_data)
    print("Data Summary:\n", summary)
    print("Correlation Matrix:\n", correlation)
    visualize_data(clean_data)

if __name__ == "__main__":
    main('data.csv')