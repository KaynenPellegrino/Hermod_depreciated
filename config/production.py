from base import BaseConfig

class ProductionConfig(BaseConfig):
    DEBUG = False
    # Production-specific configurations
# config/production.py

from .base import *

# Debug mode
DEBUG = False

# Allowed hosts
ALLOWED_HOSTS = os.getenv('ALLOWED_HOSTS', 'hermod.example.com').split(',')

# Email backend (SMTP for production)
EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST = os.getenv('EMAIL_HOST', 'smtp.gmail.com')
EMAIL_PORT = os.getenv('EMAIL_PORT', 587)
EMAIL_USE_TLS = True
EMAIL_HOST_USER = os.getenv('EMAIL_HOST_USER', 'your-email@example.com')
EMAIL_HOST_PASSWORD = os.getenv('EMAIL_HOST_PASSWORD', 'your-email-password')

# Database (production PostgreSQL)
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.getenv('DB_NAME', 'hermod_db'),
        'USER': os.getenv('DB_USER', 'hermod_user'),
        'PASSWORD': os.getenv('DB_PASSWORD', 'your_secure_password'),
        'HOST': os.getenv('DB_HOST', 'db.hermod.example.com'),
        'PORT': os.getenv('DB_PORT', '5432'),
    }
}

# Static files (collect static)
STATIC_ROOT = BASE_DIR / 'staticfiles'

# Security settings
SECURE_SSL_REDIRECT = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
SECURE_HSTS_SECONDS = 31536000  # 1 year
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True

# Logging (production settings)
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': BASE_DIR / 'logs/hermod.log',
        },
        'console': {
            'class': 'logging.StreamHandler',
        },
    },
    'loggers': {
        'django': {
            'handlers': ['file', 'console'],
            'level': 'INFO',
            'propagate': True,
        },
        'hermod_app': {
            'handlers': ['file', 'console'],
            'level': 'DEBUG',
            'propagate': False,
        },
    },
}