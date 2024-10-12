from src.modules.code_generation.project_manager import VersionControlInterface


class VersionControl(VersionControlInterface):
    def initialize_repo(self, project_path: str) -> None:
        try:
            repo = git.Repo.init(project_path)
            logging.debug(f"Initialized new Git repository at '{project_path}'.")
        except Exception as e:
            logging.error(f"Failed to initialize Git repository at '{project_path}': {e}")
            raise e

    def commit_changes(self, project_path: str, commit_message: str) -> None:
        try:
            repo = git.Repo(project_path)
            repo.git.add(A=True)
            repo.index.commit(commit_message)
            logging.debug(f"Committed changes with message: '{commit_message}'.")
        except Exception as e:
            logging.error(f"Failed to commit changes in '{project_path}': {e}")
            raise e
