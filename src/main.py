# src/main.py

import logging
import sys
from config.base import Config
from utils.logger import setup_logging
from modules.analytics.system_health_monitor import SystemHealthMonitor
from modules.analytics.user_behavior_insights import UserBehaviorInsights
from modules.code_generation.code_generator import CodeGenerator
from modules.code_generation.doc_updater import DocUpdater
from modules.code_generation.project_manager import ProjectManager
from modules.code_generation.template_manager import TemplateManager
from modules.data_management.data_collector import DataCollector
from modules.data_management.metadata_storage import MetadataStorage
from modules.feedback_loop.feedback_loop_manager import FeedbackLoopManager
from modules.gui.gui_manager import GUIManager
from modules.self_optimization.self_optimizer import SelfOptimizer
from modules.error_handling.error_logger import ErrorLogger
from modules.notifications.notification_manager import NotificationManager

# Optional security modules based on environment requirements
try:
    from modules.advanced_security.behavioral_authentication import BehavioralAuthentication
except ImportError:
    BehavioralAuthentication = None
    logging.warning("BehavioralAuthentication module not found; continuing without it.")

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)


def initialize_modules(config):
    """
    Initializes and configures the necessary modules for Hermod's operation.
    """
    logger.info("Initializing core modules...")

    # Analytics modules
    health_monitor = SystemHealthMonitor()
    behavior_insights = UserBehaviorInsights()

    # Code generation modules
    project_manager = ProjectManager()
    template_manager = TemplateManager()
    code_generator = CodeGenerator()
    doc_updater = DocUpdater(project_manager, template_manager)

    # Data management modules
    metadata_storage = MetadataStorage()
    data_collector = DataCollector()

    # Feedback loop
    feedback_manager = FeedbackLoopManager()

    # GUI management
    gui_manager = GUIManager()

    # Self-optimization
    self_optimizer = SelfOptimizer()

    # Error handling
    error_logger = ErrorLogger()

    # Notification manager for alerts
    notification_manager = NotificationManager()

    # Optional behavioral authentication if available
    behavioral_auth = BehavioralAuthentication() if BehavioralAuthentication else None

    # Store all initialized modules in a dictionary for easy reference
    modules = {
        "health_monitor": health_monitor,
        "behavior_insights": behavior_insights,
        "project_manager": project_manager,
        "template_manager": template_manager,
        "code_generator": code_generator,
        "doc_updater": doc_updater,
        "metadata_storage": metadata_storage,
        "data_collector": data_collector,
        "feedback_manager": feedback_manager,
        "gui_manager": gui_manager,
        "self_optimizer": self_optimizer,
        "error_logger": error_logger,
        "notification_manager": notification_manager,
        "behavioral_auth": behavioral_auth,
    }

    # Start data collection and metadata storage processes
    logger.info("Starting data collection and metadata initialization...")
    data_collector.start_collection()
    metadata_storage.load_metadata()

    return modules


def start_gui(gui_manager):
    """
    Starts the graphical user interface if GUIManager is available.
    """
    if gui_manager:
        logger.info("Starting GUI...")
        gui_manager.initialize_gui()
    else:
        logger.warning("GUI Manager not initialized; running in headless mode.")


def main():
    """
    Entry point for Hermod application, coordinating module initialization and starting the main application loop.
    """
    logger.info("Launching Hermod AI Assistant...")

    # Load configuration
    config = Config()

    # Initialize all modules
    modules = initialize_modules(config)

    # Start the GUI (if enabled)
    start_gui(modules.get("gui_manager"))

    # Start continuous monitoring, feedback loop, and optimization
    logger.info("Starting continuous monitoring, feedback, and optimization processes...")
    modules["health_monitor"].start_monitoring()
    modules["feedback_manager"].run_feedback_loop()
    modules["self_optimizer"].start_optimization()

    # Main operation loop
    try:
        logger.info("Running main application loop...")
        while True:
            # Periodic checks or tasks can be placed here
            pass
    except KeyboardInterrupt:
        logger.info("Shutting down Hermod...")
    finally:
        logger.info("Hermod has stopped. Cleanup complete.")


if __name__ == "__main__":
    main()
