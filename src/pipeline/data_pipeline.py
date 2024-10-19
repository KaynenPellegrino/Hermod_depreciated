# src/pipelines/data_pipeline.py

import logging
from src.modules.data_management.data_ingestor import DataIngestor
from src.modules.data_management.data_preprocessor import DataPreprocessor
from src.modules.data_management.data_storage import DataStorage

logger = logging.getLogger(__name__)

class DataPipeline:
    def __init__(self):
        self.ingestor = DataIngestor()
        self.preprocessor = DataPreprocessor()
        self.storage = DataStorage()

    def run(self, source: str, destination_table: str):
        logger.info("Starting Data Pipeline")
        raw_data = self.ingestor.ingest(source)
        processed_data = self.preprocessor.preprocess(raw_data)
        self.storage.save_data(table=destination_table, data=processed_data)
        logger.info("Data Pipeline completed successfully")
