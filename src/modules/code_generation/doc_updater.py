import logging
from typing import List, Dict, Any
from project_management.project_manager import ProjectManager
from documentation_generator.documentation_generator import DocumentationGenerator
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os
import time
from threading import Event

# Configure logging
logging.basicConfig(
    filename='hermod_doc_updater.log',
    level=logging.DEBUG,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

class FileSystemWatcher:
    """
    FileSystemWatcher class to abstract the file system monitoring functionality.
    """
    def __init__(self):
        self.observer = Observer()

    def start(self, path: str, event_handler: FileSystemEventHandler) -> None:
        self.observer.schedule(event_handler, path, recursive=True)
        self.observer.start()
        logging.debug(f"Started file system watcher on path: {path}")

    def stop(self) -> None:
        self.observer.stop()
        self.observer.join()
        logging.debug("Stopped file system watcher.")

# Custom Event Handler
class CodeChangeHandler(FileSystemEventHandler):
    def __init__(self, callback):
        """
        Initializes the event handler with a callback function.

        :param callback: Function to call when a file change is detected
        """
        super().__init__()
        self.callback = callback

    def on_any_event(self, event):
        """
        Called on any file system event.

        :param event: The event object representing the file system event
        """
        if event.is_directory:
            return
        if event.event_type in ['modified', 'created', 'deleted', 'moved']:
            logging.info(f"Detected {event.event_type} on file: {event.src_path}")
            self.callback(event)

# DocUpdater Class
class DocUpdater:
    def __init__(self,
                 project_manager: ProjectManager,
                 documentation_generator: DocumentationGenerator,
                 fs_watcher: FileSystemWatcher):
        """
        Initializes the DocUpdater with necessary dependencies.

        :param project_manager: Instance of ProjectManager
        :param documentation_generator: Instance of DocumentationGenerator
        :param fs_watcher: Instance of FileSystemWatcher
        """
        self.project_manager = project_manager
        self.documentation_generator = documentation_generator
        self.fs_watcher = fs_watcher
        self.stop_event = Event()
        logging.info("DocUpdater initialized.")

    def start_monitoring(self, project_id: str) -> None:
        """
        Starts monitoring the codebase for the specified project.

        :param project_id: Unique identifier for the project
        """
        logging.info(f"Starting documentation monitoring for project_id='{project_id}'.")
        project_structure = self.project_manager.get_project_structure(project_id)
        if not project_structure:
            logging.error(f"No project structure found for project_id='{project_id}'.")
            return

        # Assuming the base path for generated_projects is in the current directory
        base_path = os.path.join('generated_projects', project_id, 'src')
        if not os.path.exists(base_path):
            logging.error(f"Project path '{base_path}' does not exist.")
            return

        # Create event handler with a callback to handle changes
        event_handler = CodeChangeHandler(lambda event: self.on_change(event, project_id))

        # Start the file system watcher
        self.fs_watcher.start(base_path, event_handler)
        logging.info(f"Monitoring started for project_id='{project_id}' on path='{base_path}'.")

    def stop_monitoring(self) -> None:
        """
        Stops monitoring the codebase.
        """
        logging.info("Stopping documentation monitoring.")
        self.fs_watcher.stop()
        logging.info("Documentation monitoring stopped.")

    def on_change(self, event, project_id: str) -> None:
        """
        Callback function triggered when a change is detected in the codebase.

        :param event: The event object representing the file system event
        :param project_id: Unique identifier for the project
        """
        logging.info(f"Change detected in project_id='{project_id}'. Updating documentation.")
        self.update_documentation(project_id)

    def update_documentation(self, project_id: str) -> None:
        """
        Triggers the documentation regeneration process.

        :param project_id: Unique identifier for the project
        """
        try:
            documentation = self.documentation_generator.generate_all_documentation(project_id)
            if documentation:
                self.documentation_generator.save_documentation(project_id, documentation)
                logging.info(f"Documentation updated successfully for project_id='{project_id}'.")
            else:
                logging.warning(f"No documentation generated for project_id='{project_id}'.")
        except Exception as e:
            logging.error(f"Error updating documentation for project_id='{project_id}': {e}")

# Example usage and test cases
if __name__ == "__main__":
    # Initialize dependencies
    project_manager = ProjectManager()
    documentation_generator = DocumentationGenerator()
    fs_watcher = FileSystemWatcher()

    # Initialize DocUpdater
    doc_updater = DocUpdater(project_manager, documentation_generator, fs_watcher)

    # Define project ID
    project_id = "proj_12345"

    # Start monitoring in a separate thread to allow graceful shutdown
    try:
        doc_updater.start_monitoring(project_id)
        print(f"Started monitoring documentation for project '{project_id}'. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping documentation monitoring.")
        doc_updater.stop_monitoring()
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        doc_updater.stop_monitoring()