# src/modules/notifications/notification_manager.py

import logging
import os
import json
import smtplib
from datetime import datetime
from typing import List, Optional, Dict, Any
from string import Template
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from dotenv import load_dotenv
from twilio.rest import Client

# Import MetadataStorage from data_management module
from src.modules.data_management.metadata_storage import MetadataStorage

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('logs/notification_manager.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class NotificationManager:
    """
    Manages sending notifications to users or administrators via various channels such as email, SMS, Slack, or in-app notifications.
    """

    def __init__(self):
        """
        Initializes the NotificationManager with necessary configurations.
        """
        # Initialize Metadata Storage
        self.metadata_storage = MetadataStorage()

        # Email configuration
        self.smtp_server = os.getenv('ALERT_SMTP_SERVER')
        self.smtp_port = int(os.getenv('ALERT_SMTP_PORT', '587'))
        self.smtp_user = os.getenv('ALERT_SMTP_USER')
        self.smtp_password = os.getenv('ALERT_SMTP_PASSWORD')
        self.email_sender = os.getenv('ALERT_EMAIL_SENDER', self.smtp_user)
        self.email_recipient = os.getenv('ALERT_RECIPIENT')

        # Slack configuration
        self.slack_webhook_url = os.getenv('ALERT_SLACK_WEBHOOK_URL')

        # SMS configuration (Twilio)
        self.twilio_account_sid = os.getenv('ALERT_TWILIO_ACCOUNT_SID')
        self.twilio_auth_token = os.getenv('ALERT_TWILIO_AUTH_TOKEN')
        self.sms_recipients = [num.strip() for num in os.getenv('ALERT_SMS_RECIPIENTS', '').split(',') if num.strip()]
        self.sms_from_number = os.getenv('ALERT_SMS_FROM_NUMBER')

        # Push notification configuration
        self.push_webhook_url = os.getenv('ALERT_PUSH_WEBHOOK_URL')

        # Initialize Twilio Client if SMS is configured
        if self.twilio_account_sid and self.twilio_auth_token and self.sms_recipients and self.sms_from_number:
            try:
                self.twilio_client = Client(self.twilio_account_sid, self.twilio_auth_token)
                logger.info("Twilio client initialized successfully.")
            except Exception as e:
                self.twilio_client = None
                logger.error(f"Failed to initialize Twilio client: {e}")
        else:
            self.twilio_client = None
            if self.sms_recipients:
                logger.warning(
                    "Twilio credentials or SMS configuration incomplete. SMS notifications will be disabled.")

        logger.info("NotificationManager initialized successfully.")

    def send_templated_email(self, subject_template: str, message_template: str, context: Dict[str, Any],
                             recipients: Optional[List[str]] = None) -> bool:
        """
        Sends a templated email notification by substituting placeholders with actual context.

        :param subject_template: Template for the email subject.
        :param message_template: Template for the email body.
        :param context: Dictionary containing values to substitute in the templates.
        :param recipients: List of recipient email addresses. If None, uses default.
        :return: True if sent successfully, False otherwise.
        """
        subject = Template(subject_template).safe_substitute(context)
        message = Template(message_template).safe_substitute(context)
        return self.send_email(subject, message, recipients)

    def send_email(self, subject: str, message: str, recipients: Optional[List[str]] = None) -> bool:
        """
        Sends an email notification.

        :param subject: Subject of the email.
        :param message: Body of the email.
        :param recipients: List of recipient email addresses. If None, uses default.
        :return: True if sent successfully, False otherwise.
        """
        recipients = recipients or [self.email_recipient]
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_sender
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = subject

            msg.attach(MIMEText(message, 'plain'))

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)

            logger.info(f"Email sent to {recipients} with subject '{subject}'.")
            return True
        except Exception as e:
            logger.error(f"Failed to send email to {recipients}: {e}")
            return False

    def send_sms(self, message: str) -> bool:
        """
        Sends an SMS notification via Twilio.

        :param message: Message to be sent.
        :return: True if sent successfully, False otherwise.
        """
        if not self.twilio_client:
            logger.error("Twilio client is not initialized. Cannot send SMS.")
            return False

        if not self.sms_recipients:
            logger.warning("No SMS recipients configured. Skipping SMS notification.")
            return False

        success = True
        for to_number in self.sms_recipients:
            try:
                msg = self.twilio_client.messages.create(
                    body=message,
                    from_=self.sms_from_number,
                    to=to_number
                )
                logger.info(f"SMS sent to {to_number}. SID: {msg.sid}")
            except Exception as e:
                logger.error(f"Failed to send SMS to {to_number}: {e}")
                success = False
        return success

    def send_slack_message(self, message: str) -> bool:
        """
        Sends a Slack message using Incoming Webhooks.

        :param message: Message to be sent.
        :return: True if sent successfully, False otherwise.
        """
        if not self.slack_webhook_url:
            logger.error("Slack webhook URL not configured. Cannot send Slack message.")
            return False

        try:
            payload = {
                "text": message
            }
            response = requests.post(self.slack_webhook_url, json=payload)
            if response.status_code == 200:
                logger.info("Slack message sent successfully.")
                return True
            else:
                logger.error(f"Failed to send Slack message. Status Code: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Exception occurred while sending Slack message: {e}")
            return False

    def send_push_notification(self, title: str, message: str) -> bool:
        """
        Sends a push notification via a configured push service.

        :param title: Title of the push notification.
        :param message: Body of the push notification.
        :return: True if sent successfully, False otherwise.
        """
        if not self.push_webhook_url:
            logger.error("Push webhook URL not configured. Cannot send push notification.")
            return False

        try:
            payload = {
                "title": title,
                "message": message,
                # Additional fields as required by the push service
            }
            response = requests.post(self.push_webhook_url, json=payload)
            if response.status_code == 200:
                logger.info("Push notification sent successfully.")
                return True
            else:
                logger.error(f"Failed to send push notification. Status Code: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Exception occurred while sending push notification: {e}")
            return False

    def send_inapp_notification(self, user_id: str, message: str) -> bool:
        """
        Sends an in-app notification to a user.

        :param user_id: ID of the user to notify.
        :param message: Message to be sent.
        :return: True if sent successfully, False otherwise.
        """
        # Placeholder for in-app notification logic
        # This could involve interacting with a database or an API
        try:
            # Example: Send a POST request to an in-app notification API
            inapp_api_url = os.getenv('ALERT_INAPP_API_URL')
            if not inapp_api_url:
                logger.error("In-app API URL not configured. Cannot send in-app notification.")
                return False

            payload = {
                "user_id": user_id,
                "message": message,
                "timestamp": datetime.utcnow().isoformat()
            }
            response = requests.post(inapp_api_url, json=payload)
            if response.status_code == 200:
                logger.info(f"In-app notification sent to user {user_id}.")
                return True
            else:
                logger.error(
                    f"Failed to send in-app notification to user {user_id}. Status Code: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Exception occurred while sending in-app notification to user {user_id}: {e}")
            return False

    def send_notification(self, subject: str, message: str, channel: str = 'email') -> bool:
        """
        Sends a notification through the specified channel.

        :param subject: Subject or title of the notification.
        :param message: Body of the notification.
        :param channel: Channel to send the notification ('email', 'sms', 'slack', 'push', 'inapp').
        :return: True if sent successfully, False otherwise.
        """
        if channel == 'email':
            return self.send_email(subject, message)
        elif channel == 'sms':
            return self.send_sms(message)
        elif channel == 'slack':
            return self.send_slack_message(message)
        elif channel == 'push':
            return self.send_push_notification(subject, message)
        elif channel == 'inapp':
            # For in-app notifications, you might need additional parameters like user_id
            logger.error("In-app notifications require a user_id. Use the 'notify' method instead.")
            return False
        else:
            logger.warning(f"Unsupported notification channel: {channel}")
            return False

    def notify(self, channel: str, subject: Optional[str] = None, message: str = "",
               recipients: Optional[List[str]] = None, user_id: Optional[str] = None) -> bool:
        """
        General method to send notifications via specified channel.

        :param channel: Notification channel ('email', 'sms', 'slack', 'push', 'inapp').
        :param subject: Subject of the notification (used for email and push).
        :param message: Body of the notification.
        :param recipients: List of recipients (email addresses or phone numbers).
        :param user_id: User ID for in-app notifications.
        :return: True if sent successfully, False otherwise.
        """
        if channel == 'email':
            if not subject:
                logger.warning("Email subject not provided. Skipping email notification.")
                return False
            return self.send_email(subject, message, recipients)
        elif channel == 'sms':
            return self.send_sms(message)
        elif channel == 'slack':
            return self.send_slack_message(message)
        elif channel == 'push':
            if not subject:
                logger.warning("Push notification title not provided. Skipping push notification.")
                return False
            return self.send_push_notification(subject, message)
        elif channel == 'inapp':
            if not user_id:
                logger.warning("User ID not provided for in-app notification. Skipping in-app notification.")
                return False
            return self.send_inapp_notification(user_id, message)
        else:
            logger.error(f"Unsupported notification channel: {channel}")
            return False

    def add_email_recipient(self, email: str) -> bool:
        """
        Adds an email recipient to the default recipients list.

        :param email: Email address to add.
        :return: True if added successfully, False otherwise.
        """
        if not hasattr(self, 'email_recipients'):
            self.email_recipients = [self.email_recipient] if self.email_recipient else []

        if email not in self.email_recipients:
            self.email_recipients.append(email)
            logger.info(f"Added email recipient: {email}")
            return True
        else:
            logger.info(f"Email recipient already exists: {email}")
            return False

    def remove_email_recipient(self, email: str) -> bool:
        """
        Removes an email recipient from the default recipients list.

        :param email: Email address to remove.
        :return: True if removed successfully, False otherwise.
        """
        if hasattr(self, 'email_recipients') and email in self.email_recipients:
            self.email_recipients.remove(email)
            logger.info(f"Removed email recipient: {email}")
            return True
        else:
            logger.info(f"Email recipient not found: {email}")
            return False

    def add_sms_recipient(self, phone_number: str) -> bool:
        """
        Adds an SMS recipient to the default recipients list.

        :param phone_number: Phone number to add (in E.164 format).
        :return: True if added successfully, False otherwise.
        """
        if not hasattr(self, 'sms_recipients'):
            self.sms_recipients = self.sms_recipients or []

        if phone_number not in self.sms_recipients:
            self.sms_recipients.append(phone_number)
            logger.info(f"Added SMS recipient: {phone_number}")
            return True
        else:
            logger.info(f"SMS recipient already exists: {phone_number}")
            return False

    def remove_sms_recipient(self, phone_number: str) -> bool:
        """
        Removes an SMS recipient from the default recipients list.

        :param phone_number: Phone number to remove.
        :return: True if removed successfully, False otherwise.
        """
        if hasattr(self, 'sms_recipients') and phone_number in self.sms_recipients:
            self.sms_recipients.remove(phone_number)
            logger.info(f"Removed SMS recipient: {phone_number}")
            return True
        else:
            logger.info(f"SMS recipient not found: {phone_number}")
            return False
