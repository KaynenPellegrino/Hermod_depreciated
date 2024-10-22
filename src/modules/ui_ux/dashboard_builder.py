# src/modules/ui_ux/dashboard_builder.py

"""
dashboard_builder.py

Function: Dynamic Dashboard Creation
Purpose: Constructs various sections of the PyQt5-based interactive dashboard,
         including project statistics, system performance, and alerts.
"""

import os
import logging
import json
from typing import Dict, Any, List
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QLabel,
    QPushButton,
    QFileDialog,
    QMessageBox
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
import pandas as pd

# Import constants
from src.resources.constants import (
    FONT_FAMILY_TITLE,
    FONT_SIZE_TITLE,
    FONT_WEIGHT_TITLE,
    FONT_SIZE_LABEL,
    TABLE_EDIT_TRIGGERS,
    NOTIFICATION_INFO,
    NOTIFICATION_WARNING,
    NOTIFICATION_ERROR,
    EXPORT_FILE_TYPE_CSV,
    EXPORT_FILE_TYPE_ALL,
    EXPORT_DEFAULT_PROJECTS_FILE,
    EXPORT_DEFAULT_SYSTEM_FILE,
    EXPORT_DEFAULT_ALERTS_FILE
)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('logs/dashboard_builder.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)


class DashboardBuilder:
    """
    Constructs various sections of the PyQt5-based interactive dashboard.
    """

    def __init__(self):
        """
        Initializes the DashboardBuilder with necessary configurations.
        """
        try:
            # Initialize data containers
            self.project_stats_table = None
            self.system_performance_labels = {}
            self.alerts_table = None
            logger.info("DashboardBuilder initialized successfully.")
        except Exception as e:
            logger.exception(f"Failed to initialize DashboardBuilder: {e}")
            raise e

    def build_project_statistics_tab(self) -> QWidget:
        """
        Builds the Project Statistics tab with a table displaying project data.

        :return: QWidget representing the Project Statistics tab.
        """
        logger.info("Building Project Statistics tab.")
        try:
            widget = QWidget()
            layout = QVBoxLayout()
            widget.setLayout(layout)

            # Project Statistics Table
            self.project_stats_table = QTableWidget()
            self.project_stats_table.setColumnCount(5)
            self.project_stats_table.setHorizontalHeaderLabels([
                'Project Name', 'Status', 'Completion (%)', 'Start Date', 'End Date'
            ])
            self.project_stats_table.horizontalHeader().setStretchLastSection(True)
            self.project_stats_table.setEditTriggers(getattr(QTableWidget, TABLE_EDIT_TRIGGERS))
            layout.addWidget(self.project_stats_table)

            return widget
        except Exception as e:
            logger.error(f"Error building Project Statistics tab: {e}")
            raise e

    def build_system_performance_tab(self) -> QWidget:
        """
        Builds the System Performance tab with labels displaying real-time metrics.

        :return: QWidget representing the System Performance tab.
        """
        logger.info("Building System Performance tab.")
        try:
            widget = QWidget()
            layout = QVBoxLayout()
            widget.setLayout(layout)

            # CPU Usage
            cpu_layout = QHBoxLayout()
            cpu_label_title = QLabel("CPU Usage:")
            cpu_label_title.setFont(QFont(FONT_FAMILY_TITLE, FONT_SIZE_TITLE, QFont.Bold))
            self.system_performance_labels['cpu'] = QLabel("N/A")
            self.system_performance_labels['cpu'].setFont(QFont(FONT_FAMILY_TITLE, FONT_SIZE_LABEL))
            cpu_layout.addWidget(cpu_label_title)
            cpu_layout.addWidget(self.system_performance_labels['cpu'])
            cpu_layout.addStretch()
            layout.addLayout(cpu_layout)

            # Memory Usage
            memory_layout = QHBoxLayout()
            memory_label_title = QLabel("Memory Usage:")
            memory_label_title.setFont(QFont(FONT_FAMILY_TITLE, FONT_SIZE_TITLE, QFont.Bold))
            self.system_performance_labels['memory'] = QLabel("N/A")
            self.system_performance_labels['memory'].setFont(QFont(FONT_FAMILY_TITLE, FONT_SIZE_LABEL))
            memory_layout.addWidget(memory_label_title)
            memory_layout.addWidget(self.system_performance_labels['memory'])
            memory_layout.addStretch()
            layout.addLayout(memory_layout)

            # Disk Usage
            disk_layout = QHBoxLayout()
            disk_label_title = QLabel("Disk Usage:")
            disk_label_title.setFont(QFont(FONT_FAMILY_TITLE, FONT_SIZE_TITLE, QFont.Bold))
            self.system_performance_labels['disk'] = QLabel("N/A")
            self.system_performance_labels['disk'].setFont(QFont(FONT_FAMILY_TITLE, FONT_SIZE_LABEL))
            disk_layout.addWidget(disk_label_title)
            disk_layout.addWidget(self.system_performance_labels['disk'])
            disk_layout.addStretch()
            layout.addLayout(disk_layout)

            return widget
        except Exception as e:
            logger.error(f"Error building System Performance tab: {e}")
            raise e

    def build_alerts_tab(self) -> QWidget:
        """
        Builds the Alerts tab with a table displaying recent alerts.

        :return: QWidget representing the Alerts tab.
        """
        logger.info("Building Alerts tab.")
        try:
            widget = QWidget()
            layout = QVBoxLayout()
            widget.setLayout(layout)

            # Alerts Table
            self.alerts_table = QTableWidget()
            self.alerts_table.setColumnCount(3)
            self.alerts_table.setHorizontalHeaderLabels(['Timestamp', 'Severity', 'Message'])
            self.alerts_table.horizontalHeader().setStretchLastSection(True)
            self.alerts_table.setEditTriggers(getattr(QTableWidget, TABLE_EDIT_TRIGGERS))
            layout.addWidget(self.alerts_table)

            return widget
        except Exception as e:
            logger.error(f"Error building Alerts tab: {e}")
            raise e

    def update_project_statistics(self, project_stats_df: pd.DataFrame):
        """
        Updates the Project Statistics table with new data.

        :param project_stats_df: DataFrame containing project statistics.
        """
        logger.info("Updating Project Statistics table.")
        try:
            if self.project_stats_table is None:
                logger.error("Project Statistics table widget not initialized.")
                return

            self.project_stats_table.setRowCount(len(project_stats_df))
            for row_idx, row in project_stats_df.iterrows():
                self.project_stats_table.setItem(row_idx, 0, QTableWidgetItem(str(row['project_name'])))
                self.project_stats_table.setItem(row_idx, 1, QTableWidgetItem(str(row['status'])))
                self.project_stats_table.setItem(row_idx, 2, QTableWidgetItem(f"{row['completion_percentage']}%"))
                self.project_stats_table.setItem(row_idx, 3, QTableWidgetItem(str(row['start_date'])))
                self.project_stats_table.setItem(row_idx, 4, QTableWidgetItem(str(row['end_date'])))

            self.project_stats_table.resizeColumnsToContents()
            logger.info("Project Statistics table updated successfully.")
        except Exception as e:
            logger.error(f"Error updating Project Statistics table: {e}")

    def update_system_performance(self, system_metrics_df: pd.DataFrame):
        """
        Updates the System Performance labels with new data.

        :param system_metrics_df: DataFrame containing system metrics.
        """
        logger.info("Updating System Performance labels.")
        try:
            if not system_metrics_df.empty:
                latest_metrics = system_metrics_df.iloc[0]
                self.system_performance_labels['cpu'].setText(f"{latest_metrics['cpu_percent']}%")
                self.system_performance_labels['memory'].setText(f"{latest_metrics['memory_percent']}%")
                self.system_performance_labels['disk'].setText(f"{latest_metrics['disk_percent']}%")
                logger.info("System Performance labels updated successfully.")
            else:
                logger.warning("No system metrics data available to update.")
        except Exception as e:
            logger.error(f"Error updating System Performance labels: {e}")

    def update_alerts(self, alerts_df: pd.DataFrame):
        """
        Updates the Alerts table with new data.

        :param alerts_df: DataFrame containing alerts.
        """
        logger.info("Updating Alerts table.")
        try:
            if self.alerts_table is None:
                logger.error("Alerts table widget not initialized.")
                return

            self.alerts_table.setRowCount(len(alerts_df))
            for row_idx, row in alerts_df.iterrows():
                timestamp = row.get('timestamp', 'N/A')
                severity = row.get('severity', 'N/A')
                message = row.get('message', 'N/A')
                self.alerts_table.setItem(row_idx, 0, QTableWidgetItem(str(timestamp)))
                self.alerts_table.setItem(row_idx, 1, QTableWidgetItem(str(severity)))
                self.alerts_table.setItem(row_idx, 2, QTableWidgetItem(str(message)))

            self.alerts_table.resizeColumnsToContents()
            logger.info("Alerts table updated successfully.")
        except Exception as e:
            logger.error(f"Error updating Alerts table: {e}")

    def show_notification(self, title: str, message: str, notification_type: str = NOTIFICATION_INFO):
        """
        Displays a pop-up notification to the user.

        :param title: Title of the notification.
        :param message: Message content of the notification.
        :param notification_type: Type of notification ('info', 'warning', 'error').
        """
        logger.info(f"Showing notification: {title} - {message}")
        try:
            msg_box = QMessageBox()
            msg_box.setWindowTitle(title)
            msg_box.setText(message)
            if notification_type == NOTIFICATION_INFO:
                msg_box.setIcon(QMessageBox.Information)
            elif notification_type == NOTIFICATION_WARNING:
                msg_box.setIcon(QMessageBox.Warning)
            elif notification_type == NOTIFICATION_ERROR:
                msg_box.setIcon(QMessageBox.Critical)
            else:
                msg_box.setIcon(QMessageBox.Information)
            msg_box.exec_()
            logger.info("Notification displayed successfully.")
        except Exception as e:
            logger.error(f"Error displaying notification: {e}")

    def add_export_buttons(self):
        """
        Adds export buttons to each dashboard tab for data export functionality.
        """
        logger.info("Adding export buttons to dashboard tabs.")
        try:
            # Project Statistics Export Button
            export_project_btn = QPushButton("Export Project Statistics")
            export_project_btn.clicked.connect(lambda: self.export_data('projects', EXPORT_DEFAULT_PROJECTS_FILE))
            self.project_stats_table.parent().layout().addWidget(export_project_btn)

            # System Performance Export Button
            export_system_btn = QPushButton("Export System Metrics")
            export_system_btn.clicked.connect(lambda: self.export_data('system_metrics', EXPORT_DEFAULT_SYSTEM_FILE))
            self.system_performance_labels['cpu'].parent().layout().addWidget(export_system_btn)

            # Alerts Export Button
            export_alerts_btn = QPushButton("Export Alerts")
            export_alerts_btn.clicked.connect(lambda: self.export_data('alerts', EXPORT_DEFAULT_ALERTS_FILE))
            self.alerts_table.parent().layout().addWidget(export_alerts_btn)

            logger.info("Export buttons added successfully.")
        except Exception as e:
            logger.error(f"Error adding export buttons: {e}")

    def export_data(self, data_type: str, default_filename: str):
        """
        Exports specified data to a CSV file.

        :param data_type: Type of data to export ('projects', 'system_metrics', 'alerts').
        :param default_filename: Default filename for the exported CSV.
        """
        logger.info(f"Exporting {data_type} data.")
        try:
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getSaveFileName(
                None,
                f"Save {data_type.capitalize()} Data",
                default_filename,
                f"{EXPORT_FILE_TYPE_CSV};;{EXPORT_FILE_TYPE_ALL}",
                options=options
            )
            if file_path:
                if data_type == 'projects':
                    data_df = self.project_stats_table_model()
                elif data_type == 'system_metrics':
                    data_df = self.system_performance_labels_model()
                elif data_type == 'alerts':
                    data_df = self.alerts_table_model()
                else:
                    logger.error(f"Unknown data type for export: {data_type}")
                    self.show_notification("Export Error", f"Unknown data type: {data_type}", NOTIFICATION_ERROR)
                    return

                if not data_df.empty:
                    data_df.to_csv(file_path, index=False)
                    self.show_notification("Export Successful", f"{data_type.capitalize()} data exported to {file_path}.", NOTIFICATION_INFO)
                    logger.info(f"{data_type.capitalize()} data exported successfully to {file_path}.")
                else:
                    self.show_notification("Export Warning", f"No {data_type.capitalize()} data available to export.", NOTIFICATION_WARNING)
                    logger.warning(f"No {data_type.capitalize()} data available to export.")
        except Exception as e:
            logger.error(f"Failed to export {data_type} data: {e}")
            self.show_notification("Export Failed", f"Failed to export {data_type.capitalize()} data.", NOTIFICATION_ERROR)

    def project_stats_table_model(self) -> pd.DataFrame:
        """
        Retrieves data from the Project Statistics table.

        :return: DataFrame containing project statistics.
        """
        logger.info("Retrieving data from Project Statistics table for export.")
        try:
            row_count = self.project_stats_table.rowCount()
            data = []
            for row in range(row_count):
                project_name_item = self.project_stats_table.item(row, 0)
                status_item = self.project_stats_table.item(row, 1)
                completion_item = self.project_stats_table.item(row, 2)
                start_date_item = self.project_stats_table.item(row, 3)
                end_date_item = self.project_stats_table.item(row, 4)

                # Handle possible None items
                project_name = project_name_item.text() if project_name_item else 'N/A'
                status = status_item.text() if status_item else 'N/A'
                completion = completion_item.text().replace('%', '') if completion_item else '0'
                start_date = start_date_item.text() if start_date_item else 'N/A'
                end_date = end_date_item.text() if end_date_item else 'N/A'

                try:
                    completion_percentage = float(completion)
                except ValueError:
                    completion_percentage = 0.0

                data.append({
                    'project_name': project_name,
                    'status': status,
                    'completion_percentage': completion_percentage,
                    'start_date': start_date,
                    'end_date': end_date
                })
            df = pd.DataFrame(data)
            logger.info("Project Statistics table data retrieved successfully.")
            return df
        except Exception as e:
            logger.error(f"Error retrieving Project Statistics table data: {e}")
            return pd.DataFrame()

    def system_performance_labels_model(self) -> pd.DataFrame:
        """
        Retrieves data from the System Performance labels.

        :return: DataFrame containing system performance metrics.
        """
        logger.info("Retrieving data from System Performance labels for export.")
        try:
            cpu_text = self.system_performance_labels['cpu'].text().replace('%', '')
            memory_text = self.system_performance_labels['memory'].text().replace('%', '')
            disk_text = self.system_performance_labels['disk'].text().replace('%', '')

            try:
                cpu_percent = float(cpu_text)
            except ValueError:
                cpu_percent = 0.0

            try:
                memory_percent = float(memory_text)
            except ValueError:
                memory_percent = 0.0

            try:
                disk_percent = float(disk_text)
            except ValueError:
                disk_percent = 0.0

            data = {
                'cpu_percent': [cpu_percent],
                'memory_percent': [memory_percent],
                'disk_percent': [disk_percent]
            }
            df = pd.DataFrame(data)
            logger.info("System Performance labels data retrieved successfully.")
            return df
        except Exception as e:
            logger.error(f"Error retrieving System Performance labels data: {e}")
            return pd.DataFrame()

    def alerts_table_model(self) -> pd.DataFrame:
        """
        Retrieves data from the Alerts table.

        :return: DataFrame containing alerts.
        """
        logger.info("Retrieving data from Alerts table for export.")
        try:
            row_count = self.alerts_table.rowCount()
            data = []
            for row in range(row_count):
                timestamp_item = self.alerts_table.item(row, 0)
                severity_item = self.alerts_table.item(row, 1)
                message_item = self.alerts_table.item(row, 2)

                # Handle possible None items
                timestamp = timestamp_item.text() if timestamp_item else 'N/A'
                severity = severity_item.text() if severity_item else 'N/A'
                message = message_item.text() if message_item else 'N/A'

                data.append({
                    'timestamp': timestamp,
                    'severity': severity,
                    'message': message
                })
            df = pd.DataFrame(data)
            logger.info("Alerts table data retrieved successfully.")
            return df
        except Exception as e:
            logger.error(f"Error retrieving Alerts table data: {e}")
            return pd.DataFrame()
