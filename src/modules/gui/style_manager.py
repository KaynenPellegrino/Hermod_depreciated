#!/usr/bin/env python3
"""
style_manager.py

Function: GUI Style and Theme Management
Purpose: Defines the appearance of the GUI, including colors, fonts, layouts, and custom themes.
         Allows users to personalize the interface through predefined themes or customizations,
         extending from the customization_handler.py functionality.
"""

import sys
import os
import yaml
import logging
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtGui import QPalette, QColor, QFont
from PyQt5.QtCore import Qt
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
    log_file = os.path.join(log_dir, f'style_manager_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


# Initialize logging
setup_logging()


# ----------------------------
# StyleManager Class
# ----------------------------

class StyleManager:
    """
    Manages GUI styles and themes.
    Defines predefined themes, handles theme switching, and applies styles to the application.
    """

    def __init__(self, app, config, communicate, main_window):
        """
        Initialize the StyleManager with application reference, configuration, communication signals,
        and main window reference.

        :param app: QApplication instance.
        :param config: Configuration dictionary loaded from config.yaml.
        :param communicate: Communicate instance for signal handling.
        :param main_window: Reference to the main GUI window.
        """
        self.app = app
        self.config = config
        self.communicate = communicate
        self.main_window = main_window
        self.themes = self.load_themes()
        self.current_theme = 'Light'  # Default theme
        self.apply_theme(self.current_theme)
        self.setup_connections()

    def load_themes(self):
        """
        Load predefined themes from the configuration.

        :return: Dictionary of themes.
        """
        themes = self.config.get('themes', {})
        if not themes:
            logging.warning("No themes defined in configuration. Using default Light theme.")
            themes = {
                'Light': self.default_light_theme(),
                'Dark': self.default_dark_theme(),
                'High Contrast': self.default_high_contrast_theme()
            }
        logging.info("Themes loaded successfully.")
        return themes

    def default_light_theme(self):
        """
        Define the default Light theme.

        :return: Dictionary containing palette and font settings.
        """
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor('white'))
        palette.setColor(QPalette.WindowText, Qt.black)
        palette.setColor(QPalette.Base, QColor(240, 240, 240))
        palette.setColor(QPalette.AlternateBase, QColor(225, 225, 225))
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.ToolTipText, Qt.black)
        palette.setColor(QPalette.Text, Qt.black)
        palette.setColor(QPalette.Button, QColor(240, 240, 240))
        palette.setColor(QPalette.ButtonText, Qt.black)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, Qt.white)

        font = QFont('Arial', 10)

        return {'palette': palette, 'font': font}

    def default_dark_theme(self):
        """
        Define the default Dark theme.

        :return: Dictionary containing palette and font settings.
        """
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.ToolTipText, Qt.white)
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, Qt.black)

        font = QFont('Arial', 10)

        return {'palette': palette, 'font': font}

    def default_high_contrast_theme(self):
        """
        Define the default High Contrast theme.

        :return: Dictionary containing palette and font settings.
        """
        palette = QPalette()
        palette.setColor(QPalette.Window, Qt.black)
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, Qt.black)
        palette.setColor(QPalette.AlternateBase, Qt.black)
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.ToolTipText, Qt.white)
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, Qt.black)
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, QColor(0, 120, 215))
        palette.setColor(QPalette.Highlight, QColor(0, 120, 215))
        palette.setColor(QPalette.HighlightedText, Qt.black)

        font = QFont('Arial', 12, QFont.Bold)

        return {'palette': palette, 'font': font}

    def apply_theme(self, theme_name):
        """
        Apply the specified theme to the application.

        :param theme_name: Name of the theme to apply.
        """
        theme = self.themes.get(theme_name)
        if not theme:
            logging.error(f"Theme '{theme_name}' not found. Applying default Light theme.")
            theme = self.themes['Light']

        # Apply palette
        self.app.setPalette(theme['palette'])

        # Apply font
        self.app.setFont(theme['font'])

        # Additional style settings can be applied here (e.g., stylesheet)
        stylesheet = self.config.get('stylesheets', {}).get(theme_name, '')
        if stylesheet:
            self.app.setStyleSheet(stylesheet)

        self.current_theme = theme_name
        logging.info(f"Applied theme: {theme_name}")

    def switch_theme(self, theme_name):
        """
        Switch to a different theme.

        :param theme_name: Name of the theme to switch to.
        """
        if theme_name not in self.themes:
            logging.error(f"Attempted to switch to undefined theme: {theme_name}")
            self.show_notification("Theme Error", f"Theme '{theme_name}' is not available.", "error")
            return
        self.apply_theme(theme_name)
        self.show_notification("Theme Changed", f"Theme switched to '{theme_name}'.", "info")

    def create_custom_theme(self, theme_name, palette_colors, font_settings):
        """
        Create and add a custom theme.

        :param theme_name: Name of the custom theme.
        :param palette_colors: Dictionary of palette color settings.
        :param font_settings: Dictionary of font settings.
        """
        palette = QPalette()
        for role, color in palette_colors.items():
            try:
                qcolor = QColor(color)
                if not qcolor.isValid():
                    raise ValueError(f"Invalid color code: {color}")
                palette.setColor(QPalette.Role(role), qcolor)
            except Exception as e:
                logging.error(f"Error setting color for role '{role}': {e}")
                self.show_notification("Theme Error", f"Invalid color for '{role}': {color}", "error")
                return

        font = QFont(font_settings.get('family', 'Arial'),
                     font_settings.get('size', 10),
                     font_settings.get('weight', QFont.Normal))
        font.setItalic(font_settings.get('italic', False))
        font.setUnderline(font_settings.get('underline', False))

        self.themes[theme_name] = {'palette': palette, 'font': font}
        logging.info(f"Custom theme '{theme_name}' created successfully.")
        self.show_notification("Theme Created", f"Custom theme '{theme_name}' has been created.", "info")

    def save_custom_theme(self, theme_name):
        """
        Save the custom theme to the configuration for persistence.

        :param theme_name: Name of the custom theme to save.
        """
        theme = self.themes.get(theme_name)
        if not theme:
            logging.error(f"Attempted to save undefined theme: {theme_name}")
            self.show_notification("Theme Error", f"Theme '{theme_name}' is not defined.", "error")
            return

        # Serialize palette colors
        palette = theme['palette']
        palette_dict = {}
        for role in QPalette.ColorRole:
            color = palette.color(QPalette.ColorRole(role))
            role_name = QPalette.colorRoleToString(role)
            palette_dict[role_name] = color.name()

        # Serialize font settings
        font = theme['font']
        font_dict = {
            'family': font.family(),
            'size': font.pointSize(),
            'weight': font.weight(),
            'italic': font.italic(),
            'underline': font.underline()
        }

        # Update configuration
        self.config['themes'][theme_name] = {
            'palette': palette_dict,
            'font': font_dict
        }

        # Save to config.yaml
        try:
            with open('config.yaml', 'w') as file:
                yaml.dump(self.config, file)
            logging.info(f"Custom theme '{theme_name}' saved to configuration.")
            self.show_notification("Theme Saved", f"Custom theme '{theme_name}' has been saved.", "info")
        except Exception as e:
            logging.error(f"Failed to save custom theme '{theme_name}': {e}")
            self.show_notification("Save Error", f"Failed to save theme '{theme_name}'.", "error")

    def load_saved_themes(self):
        """
        Load saved themes from the configuration.
        """
        self.themes = self.load_themes()
        logging.info("Saved themes loaded successfully.")

    def setup_connections(self):
        """
        Connect signals to their respective handler methods for dynamic theme switching.
        """
        # Example: Connect a signal to switch themes
        self.communicate.switch_theme_signal.connect(self.switch_theme)

        # Example: Connect a signal to create custom themes
        self.communicate.create_custom_theme_signal.connect(self.create_custom_theme)

        # Example: Connect a signal to save custom themes
        self.communicate.save_custom_theme_signal.connect(self.save_custom_theme)

    def show_notification(self, title, message, notification_type="info"):
        """
        Display a pop-up notification to the user.

        :param title: Title of the notification.
        :param message: Message content of the notification.
        :param notification_type: Type of notification ('info', 'warning', 'error').
        """
        if notification_type == "info":
            QMessageBox.information(self.main_window, title, message)
        elif notification_type == "warning":
            QMessageBox.warning(self.main_window, title, message)
        elif notification_type == "error":
            QMessageBox.critical(self.main_window, title, message)
        else:
            QMessageBox.information(self.main_window, title, message)
