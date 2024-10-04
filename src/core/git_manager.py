import git
import os


class GitManager:

    def __init__(self, repo_path):
        self.repo_path = repo_path
        self.repo = git.Repo(repo_path)
        self.origin = self.repo.remote(name='origin')

    def commit_and_push(self, commit_message):
        try:
            self.repo.git.add(A=True)
            self.repo.index.commit(commit_message)
            self.origin.push()
            print(f'Changes committed and pushed: {commit_message}')
        except Exception as e:
            print(f'Error committing or pushing changes: {e}')

    def pull_changes(self):
        try:
            self.origin.pull()
            print('Latest changes pulled from the remote repository.')
        except Exception as e:
            print(f'Error pulling changes: {e}')
