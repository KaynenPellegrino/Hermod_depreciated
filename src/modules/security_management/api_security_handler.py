# src/modules/cybersecurity/api_security_handler.py

import logging
import os
import json
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Callable

import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import RequestException, HTTPError
from urllib3.util.retry import Retry
from dotenv import load_dotenv

# Import MetadataStorage from data_management module
from src.modules.data_management.metadata_storage import MetadataStorage

# Import NotificationManager from notifications module
from src.modules.notifications.notification_manager import NotificationManager

# Load environment variables from .env file
load_dotenv()

# Configure logging with RotatingFileHandler to prevent log files from becoming too large
from logging.handlers import RotatingFileHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = RotatingFileHandler('logs/api_security_handler.log', maxBytes=5*1024*1024, backupCount=5)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class APISecurityHandler:
    """
    Manages security protocols for external API integrations.
    Ensures secure, encrypted API calls with token-based authentication and adheres to security best practices.
    """

    def __init__(self, api_name: str, base_url: str, auth_type: str = 'none',
                 auth_credentials: Optional[Dict[str, Any]] = None,
                 headers: Optional[Dict[str, str]] = None):
        """
        Initializes the APISecurityHandler with necessary configurations.

        :param api_name: Name identifier for the API.
        :param base_url: Base URL for the API endpoints.
        :param auth_type: Type of authentication ('none', 'api_key', 'oauth2').
        :param auth_credentials: Dictionary containing authentication credentials.
        :param headers: Default headers to include in every API request.
        """
        self.api_name = api_name
        self.base_url = base_url.rstrip('/')
        self.auth_type = auth_type.lower()
        self.auth_credentials = auth_credentials or {}
        self.default_headers = headers or {}

        # Initialize Metadata Storage
        self.metadata_storage = MetadataStorage()

        # Initialize Notification Manager
        self.notification_manager = NotificationManager()

        # Initialize session with retry strategy
        self.session = requests.Session()
        retries = Retry(total=5,
                        backoff_factor=0.3,
                        status_forcelist=[500, 502, 503, 504],
                        allowed_methods=["HEAD", "GET", "OPTIONS", "POST", "PUT", "DELETE"])
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount('https://', adapter)
        self.session.mount('http://', adapter)

        # Apply default headers
        self.session.headers.update(self.default_headers)

        # Handle authentication
        if self.auth_type == 'api_key':
            self._setup_api_key_auth()
        elif self.auth_type == 'oauth2':
            self._setup_oauth2_auth()

        logger.info(f"APISecurityHandler initialized for API '{self.api_name}' with base URL '{self.base_url}'.")

    def _setup_api_key_auth(self):
        """
        Sets up API Key based authentication by adding the API key to headers or query parameters.
        """
        api_key = self.auth_credentials.get('api_key')
        key_name = self.auth_credentials.get('key_name', 'Authorization')

        if not api_key:
            logger.error(f"API Key not provided for API '{self.api_name}'.")
            raise ValueError("API Key must be provided for API Key authentication.")

        # Depending on API specification, API key can be sent in headers or as query params
        in_header = self.auth_credentials.get('in_header', True)
        if in_header:
            self.session.headers.update({key_name: api_key})
            logger.info(f"API Key added to headers for API '{self.api_name}'.")
        else:
            self.api_key_param = self.auth_credentials.get('api_key_param', 'api_key')
            logger.info(f"API Key will be sent as query parameter '{self.api_key_param}' for API '{self.api_name}'.")

    def _setup_oauth2_auth(self):
        """
        Sets up OAuth2.0 based authentication, handling token retrieval and refresh.
        """
        token_url = self.auth_credentials.get('token_url')
        client_id = self.auth_credentials.get('client_id')
        client_secret = self.auth_credentials.get('client_secret')
        scope = self.auth_credentials.get('scope', '')
        grant_type = self.auth_credentials.get('grant_type', 'client_credentials')

        if not all([token_url, client_id, client_secret]):
            logger.error(f"OAuth2 credentials incomplete for API '{self.api_name}'.")
            raise ValueError("OAuth2 credentials must include 'token_url', 'client_id', and 'client_secret'.")

        self.token_url = token_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.scope = scope
        self.grant_type = grant_type
        self.token = None
        self.token_expiry = None

        self._fetch_oauth2_token()

    def _fetch_oauth2_token(self):
        """
        Fetches a new OAuth2 token using client credentials grant.
        """
        data = {
            'grant_type': self.grant_type,
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'scope': self.scope
        }

        try:
            response = self.session.post(self.token_url, data=data)
            response.raise_for_status()
            token_data = response.json()
            self.token = token_data.get('access_token')
            expires_in = token_data.get('expires_in', 3600)  # default to 1 hour
            self.token_expiry = datetime.utcnow() + timedelta(seconds=expires_in - 60)  # refresh 1 min before expiry

            if self.token:
                self.session.headers.update({'Authorization': f'Bearer {self.token}'})
                logger.info(f"OAuth2 token acquired for API '{self.api_name}'. Expires in {expires_in} seconds.")
            else:
                logger.error(f"Failed to acquire OAuth2 token for API '{self.api_name}'.")
                raise ValueError("OAuth2 token not found in response.")

        except RequestException as e:
            logger.error(f"Error fetching OAuth2 token for API '{self.api_name}': {e}")
            raise

    def _refresh_oauth2_token_if_needed(self):
        """
        Refreshes the OAuth2 token if it's expired or about to expire.
        """
        if self.token_expiry and datetime.utcnow() >= self.token_expiry:
            logger.info(f"OAuth2 token expired or near expiry for API '{self.api_name}'. Refreshing token.")
            self._fetch_oauth2_token()

    def make_request(self, method: str, endpoint: str, params: Optional[Dict[str, Any]] = None,
                    data: Optional[Any] = None, json_data: Optional[Dict[str, Any]] = None,
                    headers: Optional[Dict[str, str]] = None,
                    timeout: int = 30,
                    callback: Optional[Callable[[requests.Response], Any]] = None) -> Optional[requests.Response]:
        """
        Makes a secure API request with proper authentication and error handling.

        :param method: HTTP method ('GET', 'POST', 'PUT', 'DELETE', etc.).
        :param endpoint: API endpoint relative to the base URL.
        :param params: Query parameters.
        :param data: Form data.
        :param json_data: JSON payload.
        :param headers: Additional headers.
        :param timeout: Request timeout in seconds.
        :param callback: Optional callback function to process the response.
        :return: Response object if successful, None otherwise.
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        request_headers = headers or {}

        # Handle OAuth2 token refresh
        if self.auth_type == 'oauth2':
            self._refresh_oauth2_token_if_needed()

        # If API key is sent as a query parameter
        if self.auth_type == 'api_key' and not self.auth_credentials.get('in_header', True):
            api_key_param = self.auth_credentials.get('api_key_param', 'api_key')
            if params:
                params[api_key_param] = self.auth_credentials.get('api_key')
            else:
                params = {api_key_param: self.auth_credentials.get('api_key')}
            logger.debug(f"API Key added as query parameter '{api_key_param}' for API '{self.api_name}'.")

        try:
            response = self.session.request(method=method.upper(),
                                            url=url,
                                            params=params,
                                            data=data,
                                            json=json_data,
                                            headers=request_headers,
                                            timeout=timeout)
            response.raise_for_status()

            logger.info(f"{method.upper()} request to '{url}' succeeded with status code {response.status_code}.")

            # Callback processing if provided
            if callback:
                callback(response)

            # Log response metadata without sensitive data
            self._log_api_interaction(method, url, params, data, json_data, response)

            return response

        except HTTPError as http_err:
            logger.error(f"HTTP error occurred for API '{self.api_name}': {http_err} - Response: {http_err.response.text}")
            self._handle_api_error(http_err.response)
        except RequestException as req_err:
            logger.error(f"Request exception occurred for API '{self.api_name}': {req_err}")
            self._handle_api_error(None)
        except Exception as e:
            logger.error(f"Unexpected error during API '{self.api_name}' request: {e}")
            self._handle_api_error(None)

        return None

    def _log_api_interaction(self, method: str, url: str, params: Optional[Dict[str, Any]],
                             data: Optional[Any], json_data: Optional[Dict[str, Any]],
                             response: requests.Response):
        """
        Logs API interactions for auditing purposes without exposing sensitive information.

        :param method: HTTP method used.
        :param url: Full URL of the request.
        :param params: Query parameters.
        :param data: Form data.
        :param json_data: JSON payload.
        :param response: Response object.
        """
        interaction = {
            'api_name': self.api_name,
            'method': method.upper(),
            'url': url,
            'params': params,
            'data': data,
            'json_data': json_data,
            'response_status': response.status_code,
            'response_time': response.elapsed.total_seconds(),
            'timestamp': datetime.utcnow().isoformat()
        }
        self.metadata_storage.save_metadata(interaction, storage_type='api_interaction')
        logger.debug(f"API interaction logged for '{self.api_name}': {interaction}")

    def _handle_api_error(self, response: Optional[requests.Response]):
        """
        Handles API errors by logging and optionally sending notifications.

        :param response: Response object that caused the error.
        """
        error_details = {
            'api_name': self.api_name,
            'error_time': datetime.utcnow().isoformat(),
            'error_response': response.text if response else 'No response',
            'error_status_code': response.status_code if response else 'No status code'
        }
        self.metadata_storage.save_metadata(error_details, storage_type='api_error')

        # Optionally, send a notification for critical errors
        if response and response.status_code >= 500:
            subject = f"Critical API Error in '{self.api_name}'"
            message = f"An error occurred while interacting with API '{self.api_name}':\n" \
                      f"Status Code: {error_details['error_status_code']}\n" \
                      f"Response: {error_details['error_response']}"
            self.notification_manager.send_notification(subject, message, channel='email')

    def add_default_header(self, key: str, value: str):
        """
        Adds a default header to be included in every API request.

        :param key: Header name.
        :param value: Header value.
        """
        self.session.headers.update({key: value})
        logger.info(f"Added default header '{key}: {value}' for API '{self.api_name}'.")

    def remove_default_header(self, key: str):
        """
        Removes a default header from the session.

        :param key: Header name to remove.
        """
        if key in self.session.headers:
            del self.session.headers[key]
            logger.info(f"Removed default header '{key}' from API '{self.api_name}'.")
        else:
            logger.warning(f"Header '{key}' not found in API '{self.api_name}' session headers.")

    def close_session(self):
        """
        Closes the session to release resources.
        """
        self.session.close()
        logger.info(f"Session closed for API '{self.api_name}'.")


# Example usage:
if __name__ == "__main__":
    # Example configuration for an API with API Key authentication
    api_handler = APISecurityHandler(
        api_name='ExampleAPI',
        base_url='https://api.example.com',
        auth_type='api_key',
        auth_credentials={
            'api_key': os.getenv('EXAMPLE_API_KEY'),
            'key_name': 'X-API-KEY',  # Custom header name if needed
            'in_header': True
        },
        headers={
            'Content-Type': 'application/json'
        }
    )

    # Making a GET request
    response = api_handler.make_request(
        method='GET',
        endpoint='/v1/resources',
        params={'param1': 'value1'},
        callback=lambda resp: logger.info(f"Callback processed response with status {resp.status_code}")
    )

    if response:
        data = response.json()
        print(data)

    # Close the session when done
    api_handler.close_session()
