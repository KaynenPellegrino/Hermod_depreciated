#!/usr/bin/env python3
"""
user_feedback_processor.py

Function: User Feedback Processing
Purpose: Processes raw user feedback, extracting meaningful information and preparing it for analysis.
         Utilizes Natural Language Processing (NLP) techniques to interpret and categorize textual feedback.
"""

import os
import sys
import yaml
import logging
from datetime import datetime
import time
import pandas as pd
from sqlalchemy import create_engine, text
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()


# ----------------------------
# Configuration and Logging
# ----------------------------

def load_config(config_path='config.yaml'):
    """
    Load configuration from a YAML file.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logging.error(f"Failed to load configuration file: {e}")
        sys.exit(1)


def setup_logging(log_dir='logs'):
    """
    Setup logging configuration.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'user_feedback_processor_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


# ----------------------------
# Database Setup
# ----------------------------

class Database:
    """
    Database connection handler using SQLAlchemy.
    """

    def __init__(self, db_config):
        self.db_config = db_config
        self.engine = self.create_db_engine()

    def create_db_engine(self):
        """
        Create a SQLAlchemy engine based on configuration.
        """
        try:
            dialect = self.db_config['dialect']
            username = self.db_config['username']
            password = self.db_config['password']
            host = self.db_config['host']
            port = self.db_config['port']
            database = self.db_config['database']
            db_url = f"{dialect}://{username}:{password}@{host}:{port}/{database}"
            engine = create_engine(db_url, pool_pre_ping=True)
            logging.info("Database engine created successfully.")
            return engine
        except Exception as e:
            logging.error(f"Failed to create database engine: {e}")
            sys.exit(1)

    def fetch_raw_feedback(self, table_name, last_processed_id):
        """
        Fetch raw user feedback from the database table where id > last_processed_id.
        """
        try:
            query = f"SELECT * FROM {table_name} WHERE id > :last_id ORDER BY id ASC"
            df = pd.read_sql_query(text(query), self.engine, params={"last_id": last_processed_id})
            logging.info(f"Fetched {len(df)} new feedback records.")
            return df
        except Exception as e:
            logging.error(f"Error fetching raw feedback: {e}")
            return pd.DataFrame()

    def insert_processed_feedback(self, table_name, df):
        """
        Insert processed feedback into the designated table.
        """
        try:
            df.to_sql(table_name, con=self.engine, if_exists='append', index=False)
            logging.info(f"Inserted {len(df)} processed feedback records into {table_name}.")
        except Exception as e:
            logging.error(f"Error inserting processed feedback: {e}")

    def get_last_processed_id(self, table_name):
        """
        Retrieve the last processed feedback ID from the processed_feedback table.
        """
        try:
            query = f"SELECT MAX(id) as last_id FROM {table_name}"
            df = pd.read_sql_query(text(query), self.engine)
            last_id = df['last_id'].iloc[0] if not df.empty and not pd.isna(df['last_id'].iloc[0]) else 0
            logging.info(f"Last processed feedback ID: {last_id}")
            return last_id
        except Exception as e:
            logging.error(f"Error retrieving last processed ID: {e}")
            return 0


# ----------------------------
# NLP Processing Class
# ----------------------------

class NLPProcessor:
    """
    Handles Natural Language Processing tasks for user feedback.
    """

    def __init__(self):
        # Use Hugging Face pipelines
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.tokenizer = pipeline("token-classification")

        # Initialize TF-IDF Vectorizer and LDA for topic modeling
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.lda = LatentDirichletAllocation(n_components=5, random_state=42)

    def preprocess_text(self, text):
        """
        Preprocess the text by tokenizing using Hugging Face tokenizer.
        """
        tokens = self.tokenizer.tokenizer(text)
        return ' '.join(tokens)

    def extract_sentiment(self, text):
        """
        Extract sentiment scores from the text using Hugging Face's sentiment analysis.
        """
        sentiment = self.sentiment_analyzer(text)
        return sentiment[0]['score']  # Overall sentiment score

    def perform_topic_modeling(self, texts):
        """
        Perform topic modeling using TF-IDF and LDA.
        """
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.lda.fit(tfidf_matrix)
        topics = self.lda.transform(tfidf_matrix)
        return topics

    def get_topic_keywords(self, num_keywords=10):
        """
        Get the top keywords for each topic.
        """
        keywords = []
        feature_names = self.vectorizer.get_feature_names_out()
        for topic_idx, topic in enumerate(self.lda.components_):
            top_features = topic.argsort()[:-num_keywords - 1:-1]
            topic_keywords = [feature_names[i] for i in top_features]
            keywords.append(', '.join(topic_keywords))
        return keywords


# ----------------------------
# User Feedback Processor Class
# ----------------------------

class UserFeedbackProcessor:
    """
    Processes raw user feedback, extracts meaningful information, and prepares it for analysis.
    """

    def __init__(self, config, db):
        self.config = config
        self.db = db
        self.raw_table = self.config.get('raw_feedback_table', 'raw_feedback')
        self.processed_table = self.config.get('processed_feedback_table', 'processed_feedback')
        self.nlp_processor = NLPProcessor()

    def process_feedback(self):
        """
        Main method to process user feedback.
        """
        last_id = self.db.get_last_processed_id(self.processed_table)
        raw_feedback_df = self.db.fetch_raw_feedback(self.raw_table, last_id)

        if raw_feedback_df.empty:
            logging.info("No new feedback to process.")
            return

        # Preprocess text
        raw_feedback_df['processed_text'] = raw_feedback_df['feedback_text'].apply(self.nlp_processor.preprocess_text)

        # Extract sentiment
        raw_feedback_df['sentiment_score'] = raw_feedback_df['feedback_text'].apply(
            self.nlp_processor.extract_sentiment)

        # Perform topic modeling
        topics = self.nlp_processor.perform_topic_modeling(raw_feedback_df['processed_text'])
        topic_keywords = self.nlp_processor.get_topic_keywords()
        raw_feedback_df['topic'] = topics.argmax(axis=1).apply(lambda x: f"Topic {x + 1}: {topic_keywords[x]}")

        # Select and rename columns for processed feedback
        processed_feedback_df = raw_feedback_df[['id', 'user_id', 'feedback_text', 'processed_text',
                                                 'sentiment_score', 'topic', 'timestamp']].copy()

        # Insert processed feedback into the database
        self.db.insert_processed_feedback(self.processed_table, processed_feedback_df)

    def run(self):
        """
        Runs the feedback processing in a loop based on the configured interval.
        """
        logging.info("Starting User Feedback Processing.")
        interval = self.config.get('processing_interval', 300)  # default: 5 minutes
        while True:
            try:
                self.process_feedback()
            except Exception as e:
                logging.error(f"Error during feedback processing: {e}")
            logging.info(f"Sleeping for {interval} seconds before next processing cycle.")
            time.sleep(interval)


# ----------------------------
# Main Function
# ----------------------------

def main():
    """
    Entry point for the User Feedback Processing module.
    """
    # Load configuration
    config = load_config()

    # Setup logging
    setup_logging(config.get('log_dir', 'logs'))

    logging.info("Initializing User Feedback Processor.")

    # Initialize Database connection
    db_config = config.get('database')
    if not db_config:
        logging.error("Database configuration not found in config.yaml.")
        sys.exit(1)

    db = Database(db_config)

    # Initialize UserFeedbackProcessor
    processor = UserFeedbackProcessor(config, db)

    # Start processing
    try:
        processor.run()
    except KeyboardInterrupt:
        logging.info("User Feedback Processing stopped by user.")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
