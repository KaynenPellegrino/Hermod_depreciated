import pandas as pd

class ReportGenerator:
    def __init__(self, data):
        self.data = data

    def generate_summary(self):
        summary = {
            'mean': self.data.mean(),
            'median': self.data.median(),
            'std_dev': self.data.std(),
            'min': self.data.min(),
            'max': self.data.max(),
            'count': self.data.count()
        }
        return summary

    def generate_report(self, filename):
        summary = self.generate_summary()
        with open(filename, 'w') as f:
            f.write("Analysis Report\n")
            f.write("================\n")
            for key, value in summary.items():
                f.write(f"{key.capitalize()}: {value}\n")

# Example usage
if __name__ == "__main__":
    data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    report_generator = ReportGenerator(data)
    report_generator.generate_report("analysis_report.txt")