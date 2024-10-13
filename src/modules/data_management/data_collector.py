# data_management/data_collector.py

import logging
import os
import requests
from typing import Dict, Any, List, Optional, Union
from abc import ABC, abstractmethod
from github import Github, GithubException, Repository
from bs4 import BeautifulSoup
import scrapy
from scrapy.crawler import CrawlerProcess
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException
from webdriver_manager.chrome import ChromeDriverManager
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text, Boolean
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError
import json
import time
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    filename='hermod_data_collector.log',
    level=logging.DEBUG,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# SQLAlchemy setup
Base = declarative_base()


class DataSource(Base):
    __tablename__ = 'data_sources'

    id = Column(Integer, primary_key=True)
    url = Column(String, unique=True, nullable=False)
    source_type = Column(String, nullable=False)  # e.g., 'github', 'web', 'api'
    metadata = Column(Text)  # JSON string
    discovered_at = Column(DateTime, nullable=False)
    is_relevant = Column(Boolean, default=False)


class DatabaseManager:
    """
    Manages PostgreSQL database connections and operations.
    """

    def __init__(self):
        self.db_host = os.getenv('POSTGRES_HOST')
        self.db_port = os.getenv('POSTGRES_PORT', '5432')
        self.db_name = os.getenv('POSTGRES_DB')
        self.db_user = os.getenv('POSTGRES_USER')
        self.db_password = os.getenv('POSTGRES_PASSWORD')

        if not all([self.db_host, self.db_port, self.db_name, self.db_user, self.db_password]):
            logging.error("Database credentials are not fully set in the environment variables.")
            raise ValueError("Missing database configuration in environment variables.")

        self.engine = create_engine(
            f"postgresql+psycopg2://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
        )
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

        # Create tables if they don't exist
        Base.metadata.create_all(self.engine)
        logging.info("DatabaseManager initialized and tables created if not existing.")

    def add_source(self, url: str, source_type: str, metadata: Dict[str, Any], is_relevant: bool = False) -> bool:
        """
        Adds a new data source to the database.

        :param url: URL of the data source
        :param source_type: Type of the data source ('github', 'web', 'api', etc.)
        :param metadata: Additional metadata as a dictionary
        :param is_relevant: Boolean indicating if the source is relevant
        :return: True if added, False if already exists
        """
        try:
            existing = self.session.query(DataSource).filter_by(url=url).first()
            if existing:
                logging.debug(f"Data source already exists: {url}")
                return False
            new_source = DataSource(
                url=url,
                source_type=source_type,
                metadata=json.dumps(metadata),
                discovered_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                is_relevant=is_relevant
            )
            self.session.add(new_source)
            self.session.commit()
            logging.info(f"Added new data source: {url}")
            return True
        except SQLAlchemyError as e:
            self.session.rollback()
            logging.error(f"Database error while adding source '{url}': {e}")
            return False

    def get_all_sources(self) -> List[DataSource]:
        """
        Retrieves all data sources from the database.

        :return: List of DataSource objects
        """
        try:
            sources = self.session.query(DataSource).all()
            logging.debug(f"Retrieved {len(sources)} data sources from the database.")
            return sources
        except SQLAlchemyError as e:
            logging.error(f"Database error while retrieving sources: {e}")
            return []

    def update_source_relevance(self, url: str, is_relevant: bool):
        """
        Updates the relevance of a data source.

        :param url: URL of the data source
        :param is_relevant: Boolean indicating if the source is relevant
        """
        try:
            source = self.session.query(DataSource).filter_by(url=url).first()
            if source:
                source.is_relevant = is_relevant
                self.session.commit()
                logging.info(f"Updated relevance for source '{url}' to {is_relevant}.")
        except SQLAlchemyError as e:
            self.session.rollback()
            logging.error(f"Database error while updating relevance for '{url}': {e}")


# Interfaces for data collection
class DataCollectorInterface(ABC):
    """
    Interface for Data Collection.
    Defines methods for collecting data from various sources.
    """

    @abstractmethod
    def collect_data(self, source: str, params: Dict[str, Any]) -> Any:
        pass


# GitHub Data Collector with enhanced retry mechanism
class GitHubDataCollector(DataCollectorInterface):
    """
    Collector for GitHub repositories using GitHub API.
    """

    def __init__(self, access_token: Optional[str] = None):
        """
        Initializes the GitHubDataCollector with an optional access token.

        :param access_token: GitHub personal access token for authenticated requests
        """
        self.access_token = os.getenv('GITHUB_ACCESS_TOKEN') if access_token is None else access_token
        if not self.access_token:
            logging.warning("No GitHub access token provided. Proceeding with unauthenticated requests (limited rate).")
            self.github = Github()
        else:
            self.github = Github(self.access_token)
        logging.info("GitHubDataCollector initialized.")

    def collect_data(self, source: str, params: Dict[str, Any] = {}) -> Any:
        """
        Collects data from a specified GitHub repository with retry logic.

        :param source: Full name of the repository (e.g., 'octocat/Hello-World')
        :param params: Parameters dict to specify what data to collect
        :return: Repository data
        """
        logging.info(f"Collecting data from GitHub repository: {source}")
        max_retries = 5
        backoff_factor = 0.3
        for retry in range(max_retries):
            try:
                repo: Repository = self.github.get_repo(source)
                repo_data = {
                    "name": repo.name,
                    "full_name": repo.full_name,
                    "description": repo.description,
                    "language": repo.language,
                    "stargazers_count": repo.stargazers_count,
                    "forks_count": repo.forks_count,
                    "open_issues_count": repo.open_issues_count,
                    "watchers_count": repo.watchers_count,
                    "created_at": repo.created_at.isoformat(),
                    "updated_at": repo.updated_at.isoformat(),
                    "pushed_at": repo.pushed_at.isoformat(),
                    "topics": repo.get_topics(),
                    "license": repo.license.name if repo.license else None,
                    "url": repo.html_url
                }
                logging.debug(f"Repository data: {repo_data}")
                return repo_data
            except GithubException as e:
                if e.status in [403, 502, 503, 504]:
                    sleep_time = backoff_factor * (2 ** retry)
                    logging.warning(f"GitHub API error {e.status}. Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    logging.error(f"GitHub API error for repository '{source}': {e}")
                    raise e
            except Exception as e:
                logging.error(f"Unexpected error while collecting GitHub data: {e}")
                raise e
        logging.error(f"Failed to collect GitHub data for '{source}' after {max_retries} retries.")
        raise Exception(f"GitHub API rate limit exceeded or persistent error for repository '{source}'.")

    def list_repositories(self, user: str, params: Dict[str, Any] = {}) -> List[Dict[str, Any]]:
        """
        Lists repositories for a specified user with retry logic.

        :param user: GitHub username
        :param params: Additional parameters like visibility, affiliation
        :return: List of repository data
        """
        logging.info(f"Listing repositories for user: {user}")
        max_retries = 5
        backoff_factor = 0.3
        for retry in range(max_retries):
            try:
                user_obj = self.github.get_user(user)
                repos = user_obj.get_repos()
                repo_list = []
                for repo in repos:
                    repo_data = {
                        "name": repo.name,
                        "full_name": repo.full_name,
                        "description": repo.description,
                        "language": repo.language,
                        "stargazers_count": repo.stargazers_count,
                        "forks_count": repo.forks_count,
                        "open_issues_count": repo.open_issues_count,
                        "watchers_count": repo.watchers_count,
                        "created_at": repo.created_at.isoformat(),
                        "updated_at": repo.updated_at.isoformat(),
                        "pushed_at": repo.pushed_at.isoformat(),
                        "topics": repo.get_topics(),
                        "license": repo.license.name if repo.license else None,
                        "url": repo.html_url
                    }
                    repo_list.append(repo_data)
                logging.debug(f"Total repositories fetched: {len(repo_list)}")
                return repo_list
            except GithubException as e:
                if e.status in [403, 502, 503, 504]:
                    sleep_time = backoff_factor * (2 ** retry)
                    logging.warning(f"GitHub API error {e.status}. Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    logging.error(f"GitHub API error while listing repositories for user '{user}': {e}")
                    raise e
            except Exception as e:
                logging.error(f"Unexpected error while listing GitHub repositories: {e}")
                raise e
        logging.error(f"Failed to list GitHub repositories for user '{user}' after {max_retries} retries.")
        raise Exception(f"GitHub API rate limit exceeded or persistent error for user '{user}'.")


# Web Scraper using Scrapy and Selenium for dynamic content
class WebScraper(DataCollectorInterface):
    """
    Scrapes data from public websites using Scrapy and Selenium for dynamic content.
    """

    def __init__(self, user_agent: str = 'HermodBot/1.0'):
        """
        Initializes the WebScraper with a specified user agent.

        :param user_agent: User-Agent header for HTTP requests
        """
        self.user_agent = user_agent
        logging.info("WebScraper initialized.")

    def collect_data(self, source: str, params: Dict[str, Any] = {}) -> Any:
        """
        Scrapes data from the specified URL based on parameters.

        :param source: URL of the website to scrape
        :param params: Parameters dict to specify what data to extract
        :return: Extracted data
        """
        logging.info(f"Starting web scraping for URL: {source}")
        try:
            if params.get('dynamic_content'):
                return self.scrape_dynamic_content(source, params)
            else:
                return self.scrape_static_content(source, params)
        except Exception as e:
            logging.error(f"Web scraping failed for URL '{source}': {e}")
            raise e

    def scrape_static_content(self, source: str, params: Dict[str, Any] = {}) -> Any:
        """
        Scrapes static content from a webpage using Scrapy.

        :param source: URL to scrape data from
        :param params: Parameters dict to specify what data to extract
        :return: Extracted data
        """
        logging.info(f"Scraping static content from '{source}'")
        process = CrawlerProcess(settings={
            'USER_AGENT': self.user_agent,
            'LOG_ENABLED': False
        })

        extracted_data = []

        class StaticScraperSpider(scrapy.Spider):
            name = 'static_scraper_spider'
            start_urls = [source]

            def parse(self, response):
                if params.get('extract_paragraphs'):
                    paragraphs = response.css('p::text').getall()
                    extracted_data.extend(paragraphs)
                if params.get('extract_links'):
                    links = response.css('a::attr(href)').getall()
                    extracted_data.extend(links)
                # Add more extraction rules based on params

        try:
            process.crawl(StaticScraperSpider)
            process.start()
            logging.debug(f"Extracted static data from {source}: {extracted_data}")
            return extracted_data
        except Exception as e:
            logging.error(f"Static web scraping failed for URL '{source}': {e}")
            raise e

    def scrape_dynamic_content(self, source: str, params: Dict[str, Any] = {}) -> Any:
        """
        Scrapes dynamic content from a webpage using Selenium.

        :param source: URL to scrape data from
        :param params: Parameters dict to specify what data to extract
        :return: Extracted data
        """
        logging.info(f"Scraping dynamic content from '{source}' using Selenium.")
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument(f'user-agent={self.user_agent}')
            driver = webdriver.Chrome(ChromeDriverManager().install(), options=chrome_options)
            driver.get(source)
            time.sleep(3)  # Wait for JavaScript to load content

            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')
            data = []
            if params.get('extract_paragraphs'):
                data.extend([p.get_text() for p in soup.find_all('p')])
            if params.get('extract_links'):
                data.extend([a.get('href') for a in soup.find_all('a', href=True)])
            # Add more extraction logic based on params

            logging.debug(f"Extracted dynamic data from {source}: {data}")
            driver.quit()
            return data
        except WebDriverException as e:
            logging.error(f"Selenium WebDriver error while scraping '{source}': {e}")
            raise e
        except Exception as e:
            logging.error(f"Dynamic web scraping failed for URL '{source}': {e}")
            raise e


# API Data Collector with enhanced retry mechanism
class APICollector(DataCollectorInterface):
    """
    Collector for fetching data from external APIs with retry logic.
    """

    def __init__(self, base_url: str, headers: Optional[Dict[str, str]] = None):
        """
        Initializes the APICollector with base URL and optional headers.

        :param base_url: Base URL of the API
        :param headers: Optional HTTP headers for requests
        """
        self.base_url = base_url
        self.headers = headers or {}
        logging.info(f"APICollector initialized with base URL: {self.base_url}")

    def collect_data(self, endpoint: str, params: Dict[str, Any] = {}) -> Any:
        """
        Fetches data from the specified API endpoint with retry logic.

        :param endpoint: API endpoint to fetch data from
        :param params: Query parameters for the API request
        :return: Fetched data
        """
        url = os.path.join(self.base_url, endpoint)
        logging.info(f"Fetching data from API endpoint: {url} with params: {params}")
        max_retries = 5
        backoff_factor = 0.3
        for retry in range(max_retries):
            try:
                response = requests.get(url, headers=self.headers, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                logging.debug(f"Received data from API: {json.dumps(data, indent=2)[:500]}...")
                return data
            except requests.exceptions.HTTPError as e:
                if response.status_code in [429, 500, 502, 503, 504]:
                    sleep_time = backoff_factor * (2 ** retry)
                    logging.warning(f"API request error {response.status_code}. Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    logging.error(f"API request failed for endpoint '{endpoint}': {e}")
                    raise e
            except requests.exceptions.RequestException as e:
                sleep_time = backoff_factor * (2 ** retry)
                logging.warning(f"API request exception: {e}. Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            except ValueError as e:
                logging.error(f"Invalid JSON response from endpoint '{endpoint}': {e}")
                raise e
        logging.error(f"Failed to fetch data from API endpoint '{endpoint}' after {max_retries} retries.")
        raise Exception(f"API request failed for endpoint '{endpoint}' after {max_retries} retries.")


# Autonomous Source Discovery using Web Crawling
class SourceDiscovery:
    """
    Enables Hermod to discover new data sources by crawling the internet.
    """

    def __init__(self, start_urls: List[str], max_depth: int = 2, user_agent: str = 'HermodBot/1.0',
                 db_manager: Optional[DatabaseManager] = None):
        """
        Initializes the SourceDiscovery with starting URLs and crawl depth.

        :param start_urls: List of URLs to start crawling from
        :param max_depth: Maximum depth to crawl
        :param user_agent: User-Agent header for HTTP requests
        :param db_manager: Instance of DatabaseManager for storing discovered sources
        """
        self.start_urls = start_urls
        self.max_depth = max_depth
        self.user_agent = user_agent
        self.db_manager = db_manager
        logging.info("SourceDiscovery initialized.")

    def discover_sources(self) -> List[str]:
        """
        Discovers new data sources by crawling the internet.

        :return: List of discovered source URLs
        """
        logging.info("Starting source discovery process.")
        process = CrawlerProcess(settings={
            'USER_AGENT': self.user_agent,
            'DEPTH_LIMIT': self.max_depth,
            'LOG_ENABLED': False
        })

        discovered_sources = []

        class SourceSpider(scrapy.Spider):
            name = 'source_spider'
            start_urls = self.start_urls

            def parse(self, response):
                links = response.css('a::attr(href)').getall()
                for link in links:
                    if link.startswith('http'):
                        discovered_sources.append(link)
                        yield scrapy.Request(url=link, callback=self.parse)

        try:
            process.crawl(SourceSpider)
            process.start()
            unique_sources = list(set(discovered_sources))
            logging.debug(f"Discovered {len(unique_sources)} unique sources.")

            # Store discovered sources in the database
            for url in unique_sources:
                # Placeholder for metadata; can be expanded as needed
                metadata = {"discovered_from": self.start_urls}
                self.db_manager.add_source(url=url, source_type='web', metadata=metadata)

            return unique_sources
        except Exception as e:
            logging.error(f"Source discovery failed: {e}")
            raise e


# Machine Learning for Source Relevance
class SourceClassifier:
    """
    Classifies the relevance of data sources using a pre-trained machine learning model.
    """

    def __init__(self, model_path: str = 'models/source_classifier.joblib'):
        """
        Initializes the SourceClassifier with a pre-trained model.

        :param model_path: Path to the trained machine learning model
        """
        if not os.path.exists(model_path):
            logging.error(f"Machine learning model not found at '{model_path}'.")
            raise FileNotFoundError(f"Model file '{model_path}' does not exist.")

        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load('models/vectorizer.joblib')
        logging.info("SourceClassifier initialized with pre-trained model.")

    def classify(self, text: str) -> bool:
        """
        Classifies the source relevance based on the input text.

        :param text: Text content of the source (e.g., webpage content)
        :return: Boolean indicating if the source is relevant
        """
        try:
            X = self.vectorizer.transform([text])
            prediction = self.model.predict(X)[0]
            logging.debug(f"Source relevance classification result: {prediction}")
            return bool(prediction)
        except Exception as e:
            logging.error(f"Error during source classification: {e}")
            return False


# Enhanced Web Scraper with Selenium
class EnhancedWebScraper(WebScraper):
    """
    Scrapes dynamic content from public websites using Selenium for JavaScript-rendered pages.
    """

    def __init__(self, user_agent: str = 'HermodBot/1.0'):
        """
        Initializes the EnhancedWebScraper with Selenium WebDriver.

        :param user_agent: User-Agent header for HTTP requests
        """
        self.user_agent = user_agent
        logging.info("EnhancedWebScraper initialized with Selenium.")

    def collect_data(self, source: str, params: Dict[str, Any] = {}) -> Any:
        """
        Scrapes data from the specified URL using Selenium for dynamic content.

        :param source: URL of the website to scrape
        :param params: Parameters dict to specify what data to extract
        :return: Extracted data
        """
        logging.info(f"Starting enhanced web scraping for URL: {source} using Selenium.")
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument(f'user-agent={self.user_agent}')
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--no-sandbox")
            driver = webdriver.Chrome(ChromeDriverManager().install(), options=chrome_options)
            driver.get(source)
            time.sleep(3)  # Wait for JavaScript to render content

            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')
            data = []
            if params.get('extract_paragraphs'):
                data.extend([p.get_text() for p in soup.find_all('p')])
            if params.get('extract_links'):
                data.extend([a.get('href') for a in soup.find_all('a', href=True)])
            # Add more extraction logic based on params

            logging.debug(f"Extracted enhanced data from {source}: {data}")
            driver.quit()
            return data
        except WebDriverException as e:
            logging.error(f"Selenium WebDriver error while scraping '{source}': {e}")
            raise e
        except Exception as e:
            logging.error(f"Enhanced web scraping failed for URL '{source}': {e}")
            raise e


# Composite DataCollector orchestrating different collectors with ML and DB integration
class DataCollector:
    """
    Orchestrates data collection from various sources, integrates with database and machine learning for source relevance.
    """

    def __init__(self,
                 github_collector: GitHubDataCollector,
                 web_scraper: WebScraper,
                 api_collector: APICollector,
                 source_discovery: Optional[SourceDiscovery] = None,
                 db_manager: Optional[DatabaseManager] = None,
                 source_classifier: Optional[SourceClassifier] = None):
        """
        Initializes the DataCollector with specific data collectors, database manager, and classifier.

        :param github_collector: Instance of GitHubDataCollector
        :param web_scraper: Instance of WebScraper (can be Generic or Enhanced)
        :param api_collector: Instance of APICollector
        :param source_discovery: Optional instance of SourceDiscovery for discovering new sources
        :param db_manager: Instance of DatabaseManager for database interactions
        :param source_classifier: Instance of SourceClassifier for classifying source relevance
        """
        self.github_collector = github_collector
        self.web_scraper = web_scraper
        self.api_collector = api_collector
        self.source_discovery = source_discovery
        self.db_manager = db_manager
        self.source_classifier = source_classifier
        logging.info("DataCollector initialized with all components.")

    def collect_github_repo(self, repo_full_name: str) -> Dict[str, Any]:
        """
        Collects data from a specific GitHub repository and stores it in the database.

        :param repo_full_name: Full name of the repository (e.g., 'octocat/Hello-World')
        :return: Repository data
        """
        logging.info(f"Collecting GitHub repository data for '{repo_full_name}'.")
        repo_data = self.github_collector.collect_data(repo_full_name)

        # Add to database
        self.db_manager.add_source(
            url=repo_data['url'],
            source_type='github',
            metadata=repo_data,
            is_relevant=True  # Assuming GitHub repos are relevant by default
        )
        return repo_data

    def collect_web_data(self, url: str, params: Dict[str, Any] = {}) -> Any:
        """
        Collects data from a specified web URL, classifies its relevance, and stores it in the database.

        :param url: URL to scrape data from
        :param params: Parameters dict to specify what data to extract
        :return: Extracted data
        """
        logging.info(f"Collecting web data from '{url}'.")
        data = self.web_scraper.collect_data(url, params)

        # Combine all extracted text for classification
        combined_text = ' '.join([str(item) for item in data if isinstance(item, str)])
        is_relevant = self.source_classifier.classify(combined_text) if self.source_classifier else True

        # Add to database
        self.db_manager.add_source(
            url=url,
            source_type='web',
            metadata={"extracted_data": data},
            is_relevant=is_relevant
        )
        return data

    def collect_api_data(self, endpoint: str, params: Dict[str, Any] = {}) -> Any:
        """
        Collects data from a specified API endpoint and stores it in the database.

        :param endpoint: API endpoint to fetch data from
        :param params: Query parameters for the API request
        :return: Fetched data
        """
        logging.info(f"Collecting API data from endpoint '{endpoint}'.")
        api_data = self.api_collector.collect_data(endpoint, params)

        # Assuming API data contains URLs or identifiers to classify relevance
        # Placeholder: Combine all text data for classification
        if isinstance(api_data, list):
            combined_text = ' '.join([json.dumps(item) for item in api_data if isinstance(item, dict)])
        elif isinstance(api_data, dict):
            combined_text = json.dumps(api_data)
        else:
            combined_text = str(api_data)

        is_relevant = self.source_classifier.classify(combined_text) if self.source_classifier else True

        # Add to database
        self.db_manager.add_source(
            url=f"{self.api_collector.base_url}/{endpoint}",
            source_type='api',
            metadata={"api_data": api_data},
            is_relevant=is_relevant
        )
        return api_data

    def discover_new_sources(self, start_urls: List[str] = ['https://www.example.com']) -> List[str]:
        """
        Discovers new data sources by crawling the internet and stores them in the database.

        :param start_urls: Starting URLs for crawling
        :return: List of discovered source URLs
        """
        if not self.source_discovery:
            logging.warning("SourceDiscovery instance not provided. Skipping source discovery.")
            return []
        logging.info("Discovering new data sources.")
        new_sources = self.source_discovery.discover_sources()
        logging.info(f"Discovered {len(new_sources)} new sources.")
        return new_sources


# Machine Learning Model Training (if needed)
def train_source_classifier(model_path: str = 'models/source_classifier.joblib',
                            vectorizer_path: str = 'models/vectorizer.joblib'):
    """
    Trains a simple machine learning model for source relevance classification.
    This is a placeholder and should be replaced with actual training data and process.

    :param model_path: Path to save the trained model
    :param vectorizer_path: Path to save the trained vectorizer
    """
    logging.info("Starting training of SourceClassifier model.")
    # Example training data
    texts = [
        "This is a reputable source with high-quality data.",
        "This website contains spam and irrelevant information.",
        "Official documentation and tutorials.",
        "Random blog with no useful content."
    ]
    labels = [1, 0, 1, 0]  # 1: Relevant, 0: Not relevant

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)

    model = LogisticRegression()
    model.fit(X, labels)

    # Save the model and vectorizer
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    logging.info(f"SourceClassifier model trained and saved to '{model_path}' and '{vectorizer_path}'.")


# Uncomment the following line to train the model (only run once)
# train_source_classifier()

# Example usage and test cases
if __name__ == "__main__":
    # Initialize DatabaseManager
    try:
        db_manager = DatabaseManager()
    except Exception as e:
        print(f"Database initialization failed: {e}")
        exit(1)

    # Initialize GitHubDataCollector
    github_collector = GitHubDataCollector()

    # Initialize EnhancedWebScraper (Selenium)
    web_scraper = EnhancedWebScraper(user_agent='HermodBot/1.0')

    # Initialize APICollector (Example with JSONPlaceholder API)
    api_collector = APICollector(base_url='https://jsonplaceholder.typicode.com')

    # Initialize SourceDiscovery with some starting URLs
    source_discovery = SourceDiscovery(
        start_urls=['https://www.example.com'],
        max_depth=1,  # Limiting depth for demonstration; increase as needed
        user_agent='HermodBot/1.0',
        db_manager=db_manager
    )

    # Initialize SourceClassifier
    try:
        source_classifier = SourceClassifier(model_path='models/source_classifier.joblib')
    except FileNotFoundError:
        print(
            "SourceClassifier model not found. Please train the model first by uncommenting the 'train_source_classifier()' line and running the script.")
        source_classifier = None

    # Initialize DataCollector
    data_collector = DataCollector(
        github_collector=github_collector,
        web_scraper=web_scraper,
        api_collector=api_collector,
        source_discovery=source_discovery,
        db_manager=db_manager,
        source_classifier=source_classifier
    )

    # Example 1: Collect data from a GitHub repository
    try:
        repo_full_name = 'octocat/Hello-World'  # Replace with desired repository
        repo_data = data_collector.collect_github_repo(repo_full_name)
        print("GitHub Repository Data:")
        print(json.dumps(repo_data, indent=2))
    except Exception as e:
        print(f"Failed to collect GitHub data: {e}")

    # Example 2: Collect data from a dynamic web page
    try:
        url = 'https://www.example.com'  # Replace with desired URL
        params = {'extract_paragraphs': True, 'extract_links': True, 'dynamic_content': True}
        web_data = data_collector.collect_web_data(url, params)
        print("\nWeb Scraped Data (Paragraphs and Links):")
        print(web_data)
    except Exception as e:
        print(f"Failed to scrape web data: {e}")

    # Example 3: Collect data from an API
    try:
        endpoint = 'posts'
        api_params = {'userId': 1}
        api_data = data_collector.collect_api_data(endpoint, api_params)
        print("\nAPI Data (Posts):")
        print(json.dumps(api_data, indent=2))
    except Exception as e:
        print(f"Failed to collect API data: {e}")

    # Example 4: Discover new data sources
    try:
        new_sources = data_collector.discover_new_sources()
        print("\nDiscovered New Sources:")
        for source in new_sources:
            print(source)
    except Exception as e:
        print(f"Failed to discover new sources: {e}")
