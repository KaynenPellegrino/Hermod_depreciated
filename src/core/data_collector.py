import json
import requests
from bs4 import BeautifulSoup

class DataCollector:
    def __init__(self):
        self.log_file = "hermod_data_log.json"
        self.sources = ["https://github.com/", "https://stackoverflow.com/"]

    def log_data(self, event, result):
        """Log data into a file."""
        log_entry = {"event": event, "result": result}
        with open(self.log_file, "a") as f:
            json.dump(log_entry, f)
            f.write("\n")
        print(f"Logged event: {event}")

    def collect_data_online(self, keyword):
        """Scrape web data based on the keyword."""
        collected_data = []
        for source in self.sources:
            response = requests.get(source)
            soup = BeautifulSoup(response.text, 'html.parser')
            data = soup.find_all(text=lambda t: keyword in t)
            collected_data.extend(data)
        return collected_data
