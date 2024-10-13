# src/modules/notifications/notification_manager.py

import logging
import os
import json
from datetime import datetime
from typing import List, Optional, Dict, Any
from string import Template

from dotenv import load_dotenv
from twilio.rest import Client

# Import MetadataStorage from data_management module
from src.modules.data_management.metadata_storage import MetadataStorage

# Load environment variables from .env file
load_dotenv()

# Configure logging with RotatingFileHandler to prevent log files from becoming too large
from logging.handlers import RotatingFileHandler

from src.utils.logger import get_logger

logger = get_logger(__name__, 'logs/notification_manager.log')


class NotificationManager:
    """
    Manages sending notifications to users or administrators via various channels such as email, SMS, or in-app notifications.
    """

    def __init__(self):
        """
        Initializes the NotificationManager with necessary configurations.
        """
        # Initialize Metadata Storage
        self.metadata_storage = MetadataStorage()

        # Configuration parameters for Email
        self.smtp_server = os.getenv('NOTIFICATION_SMTP_SERVER', 'smtp.example.com')
        self.smtp_port = int(os.getenv('NOTIFICATION_SMTP_PORT', '587'))
        self.smtp_user = os.getenv('NOTIFICATION_SMTP_USER', 'alert@example.com')
        self.smtp_password = os.getenv('NOTIFICATION_SMTP_PASSWORD', 'alertpassword')

        # Configuration parameters for SMS
        self.twilio_account_sid = os.getenv('NOTIFICATION_TWILIO_ACCOUNT_SID', '')
        self.twilio_auth_token = os.getenv('NOTIFICATION_TWILIO_AUTH_TOKEN', '')
        self.twilio_from_number = os.getenv('NOTIFICATION_TWILIO_FROM_NUMBER', '')

        # Default notification recipients
        self.email_recipients: List[str] = json.loads(os.getenv('NOTIFICATION_EMAIL_RECIPIENTS', '[]'))
        self.sms_recipients: List[str] = json.loads(os.getenv('NOTIFICATION_SMS_RECIPIENTS', '[]'))

        # Initialize Twilio Client if SMS is configured
        if self.twilio_account_sid and self.twilio_auth_token:
            self.twilio_client = Client(self.twilio_account_sid, self.twilio_auth_token)
            logger.info("Twilio client initialized successfully.")
        else:
            self.twilio_client = None
            if self.sms_recipients:
                logger.warning("Twilio credentials not found. SMS notifications will be disabled.")

        logger.info("NotificationManager initialized successfully.")

    def send_templated_email(self, subject_template: str, message_template: str, context: Dict[str, Any],
                             recipients: Optional[List[str]] = None):
        subject = Template(subject_template).safe_substitute(context)
        message = Template(message_template).safe_substitute(context)
        self.send_email(subject, message, recipients)

    def send_email(self, subject: str, message: str, recipients: Optional[List[str]] = None):
        """
        Sends an email notification to the specified recipients.

        :param subject: Subject of the email.
        :param message: Body of the email.
        :param recipients: List of email addresses to send the email to. If None, uses default recipients.
        """
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart

        recipients = recipients if recipients is not None else self.email_recipients

        if not recipients:
            logger.warning("No email recipients specified. Skipping email notification.")
            return

        try:
            # Create the email message
            msg = MIMEMultipart()
            msg['From'] = self.smtp_user
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = subject

            msg.attach(MIMEText(message, 'plain'))

            # Connect to the SMTP server and send the email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.smtp_user, self.smtp_password)
            server.sendmail(self.smtp_user, recipients, msg.as_string())
            server.quit()

            logger.info(f"Email sent to {recipients} with subject '{subject}'.")

            # Log the notification in Metadata Storage
            notification_record = {
                'timestamp': datetime.utcnow().isoformat(),
                'channel': 'email',
                'recipients': recipients,
                'subject': subject,
                'message': message
            }
            self.metadata_storage.save_metadata(notification_record, storage_type='notifications')

        except Exception as e:
            logger.error(f"Failed to send email: {e}")

    def send_slack_notification(self, webhook_url: str, message: str):
        import requests
        try:
            payload = {'text': message}
            response = requests.post(webhook_url, json=payload)
            response.raise_for_status()
            logger.info("Slack notification sent successfully.")

            # Log the notification
            notification_record = {
                'timestamp': datetime.utcnow().isoformat(),
                'channel': 'slack',
                'webhook_url': webhook_url,
                'message': message
            }
            self.metadata_storage.save_metadata(notification_record, storage_type='notifications')
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")

    def send_sms(self, message: str, recipients: Optional[List[str]] = None):
        """
        Sends an SMS notification to the specified recipients using Twilio.

        :param message: Body of the SMS.
        :param recipients: List of phone numbers to send the SMS to. If None, uses default recipients.
        """
        recipients = recipients if recipients is not None else self.sms_recipients

        if not recipients:
            logger.warning("No SMS recipients specified. Skipping SMS notification.")
            return

        if not self.twilio_client:
            logger.error("Twilio client not initialized. Cannot send SMS.")
            return

        for recipient in recipients:
            try:
                message_response = self.twilio_client.messages.create(
                    body=message,
                    from_=self.twilio_from_number,
                    to=recipient
                )
                logger.info(f"SMS sent to {recipient}. Message SID: {message_response.sid}")

                # Log the notification in Metadata Storage
                notification_record = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'channel': 'sms',
                    'recipients': [recipient],
                    'message': message,
                    'sid': message_response.sid
                }
                self.metadata_storage.save_metadata(notification_record, storage_type='notifications')

            except Exception as e:
                logger.error(f"Failed to send SMS to {recipient}: {e}")

    def send_inapp_notification(self, user_id: str, message: str):
        """
        Sends an in-app notification to a specific user.

        :param user_id: Identifier of the user to send the notification to.
        :param message: Body of the in-app notification.
        """
        try:
            # Placeholder for in-app notification logic
            # This could involve writing to a database table that the frontend listens to,
            # sending via WebSockets, or integrating with a messaging service.
            # For demonstration, we'll log the notification.

            logger.info(f"In-App notification sent to user '{user_id}': {message}")

            # Log the notification in Metadata Storage
            notification_record = {
                'timestamp': datetime.utcnow().isoformat(),
                'channel': 'inapp',
                'recipients': [user_id],
                'message': message
            }
            self.metadata_storage.save_metadata(notification_record, storage_type='notifications')

        except Exception as e:
            logger.error(f"Failed to send in-app notification to user '{user_id}': {e}")

    def notify(self, channel: str, subject: Optional[str] = None, message: str = "", recipients: Optional[List[str]] = None, user_id: Optional[str] = None):
        """
        General method to send notifications via specified channel.

        :param channel: Notification channel ('email', 'sms', 'inapp').
        :param subject: Subject of the notification (used for email).
        :param message: Body of the notification.
        :param recipients: List of recipients (email addresses or phone numbers).
        :param user_id: User ID for in-app notifications.
        """
        if channel == 'email':
            if not subject:
                logger.warning("Email subject not provided. Skipping email notification.")
                return
            self.send_email(subject, message, recipients)
        elif channel == 'sms':
            self.send_sms(message, recipients)
        elif channel == 'inapp':
            if not user_id:
                logger.warning("User ID not provided for in-app notification. Skipping in-app notification.")
                return
            self.send_inapp_notification(user_id, message)
        else:
            logger.error(f"Unsupported notification channel: {channel}")

    def add_email_recipient(self, email: str):
        """
        Adds an email recipient to the default recipients list.

        :param email: Email address to add.
        """
        if email not in self.email_recipients:
            self.email_recipients.append(email)
            logger.info(f"Added email recipient: {email}")
        else:
            logger.info(f"Email recipient already exists: {email}")

    def remove_email_recipient(self, email: str):
        """
        Removes an email recipient from the default recipients list.

        :param email: Email address to remove.
        """
        if email in self.email_recipients:
            self.email_recipients.remove(email)
            logger.info(f"Removed email recipient: {email}")
        else:
            logger.info(f"Email recipient not found: {email}")

    def add_sms_recipient(self, phone_number: str):
        """
        Adds an SMS recipient to the default recipients list.

        :param phone_number: Phone number to add (in E.164 format).
        """
        if phone_number not in self.sms_recipients:
            self.sms_recipients.append(phone_number)
            logger.info(f"Added SMS recipient: {phone_number}")
        else:
            logger.info(f"SMS recipient already exists: {phone_number}")

    def remove_sms_recipient(self, phone_number: str):
        """
        Removes an SMS recipient from the default recipients list.

        :param phone_number: Phone number to remove.
        """
        if phone_number in self.sms_recipients:
            self.sms_recipients.remove(phone_number)
            logger.info(f"Removed SMS recipient: {phone_number}")
        else:
            logger.info(f"SMS recipient not found: {phone_number}")
