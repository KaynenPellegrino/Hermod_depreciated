# src/utils/helpers.py

import os
import smtplib
import random
import string
import unicodedata
import re
import uuid
from datetime import datetime, timedelta
from email.message import EmailMessage
from typing import Optional
from src.utils.config_loader import ConfigurationManager

# Load configuration
config_manager = ConfigurationManager()
EMAIL_HOST = config_manager.get('email.host', 'smtp.example.com')
EMAIL_PORT = config_manager.get('email.port', 587)
EMAIL_HOST_USER = config_manager.get('email.user', 'noreply@example.com')
EMAIL_HOST_PASSWORD = config_manager.get('email.password', 'password')
EMAIL_USE_TLS = config_manager.get('email.use_tls', True)
EMAIL_USE_SSL = config_manager.get('email.use_ssl', False)
DEFAULT_FROM_EMAIL = config_manager.get('email.default_from', 'noreply@example.com')

def send_email(
    subject: str,
    body: str,
    to: list,
    from_email: Optional[str] = None,
    cc: Optional[list] = None,
    bcc: Optional[list] = None,
    attachments: Optional[list] = None,
) -> bool:
    """
    Sends an email using SMTP.
    """
    from_email = from_email or DEFAULT_FROM_EMAIL
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = from_email
    msg['To'] = ', '.join(to)
    if cc:
        msg['Cc'] = ', '.join(cc)
    if bcc:
        msg['Bcc'] = ', '.join(bcc)
    msg.set_content(body)

    if attachments:
        for attachment in attachments:
            # attachment should be a tuple (filename, content, mimetype)
            filename, content, mimetype = attachment
            msg.add_attachment(content, maintype=mimetype.split('/')[0],
                               subtype=mimetype.split('/')[1], filename=filename)

    try:
        if EMAIL_USE_SSL:
            server = smtplib.SMTP_SSL(EMAIL_HOST, EMAIL_PORT)
        else:
            server = smtplib.SMTP(EMAIL_HOST, EMAIL_PORT)
        if EMAIL_USE_TLS:
            server.starttls()
        server.login(EMAIL_HOST_USER, EMAIL_HOST_PASSWORD)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        # Log the exception
        import logging
        logger = logging.getLogger(__name__)
        logger.exception(f"Failed to send email: {e}")
        return False

def generate_random_password(length: int = 12) -> str:
    """
    Generates a secure random password.
    """
    chars = string.ascii_letters + string.digits + string.punctuation
    password = ''.join(random.SystemRandom().choice(chars) for _ in range(length))
    return password

def slugify(value: str, allow_unicode: bool = False) -> str:
    """
    Converts a string to a URL-friendly slug.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    value = re.sub(r'[-\s]+', '-', value).strip('-_')
    return value

def humanize_time(delta: timedelta) -> str:
    """
    Converts a timedelta to a human-readable string.
    """
    seconds = int(delta.total_seconds())
    periods = [
        ('year', 60*60*24*365),
        ('month', 60*60*24*30),
        ('day', 60*60*24),
        ('hour', 60*60),
        ('minute', 60),
        ('second', 1)
    ]
    strings = []
    for period_name, period_seconds in periods:
        if seconds >= period_seconds:
            period_value, seconds = divmod(seconds, period_seconds)
            if period_value > 0:
                strings.append(f"{period_value} {period_name}{'s' if period_value > 1 else ''}")
    return ', '.join(strings)

def get_current_timestamp() -> int:
    """
    Returns the current UTC timestamp.
    """
    return int(datetime.utcnow().timestamp())

def generate_uuid() -> str:
    """
    Generates a UUID string.
    """
    return str(uuid.uuid4())

def secure_filename(filename: str) -> str:
    """
    Secures a filename before storing it directly on the filesystem.
    """
    filename = os.path.basename(filename)
    filename = filename.replace(os.path.sep, '_')
    return filename

def parse_size(size_str: str) -> int:
    """
    Parses a human-readable file size string (e.g., '10MB') into bytes.
    """
    size_str = size_str.strip().upper()
    units = {'B':1, 'KB':1024, 'MB':1024**2, 'GB':1024**3, 'TB':1024**4}
    match = re.match(r'(\d+(?:\.\d+)?)([KMGTP]?B)', size_str)
    if not match:
        raise ValueError(f"Invalid size format: {size_str}")
    size, unit = match.groups()
    size = float(size) * units[unit]
    return int(size)

def sizeof_fmt(num: float, suffix: str = 'B') -> str:
    """
    Converts a file size to human-readable form.

    Args:
        num (float): The size in bytes.
        suffix (str, optional): The suffix to use. Defaults to 'B'.

    Returns:
        str: Human-readable file size.
    """
    for unit in ['','K','M','G','T','P','E','Z']:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Y{suffix}"