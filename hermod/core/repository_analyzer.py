import os
import requests
from radon.complexity import cc_visit
from hermod.utils.logger import setup_logger
import subprocess

# Initialize logger
logger = setup_logger()

def fetch_open_source_repositories(language='Python', stars=500):
    url = f'https://api.github.com/search/repositories?q=language:{language}+stars:>{stars}'
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['items']
    else:
        logger.error(f"Failed to fetch repositories from GitHub: {response.status_code}")
        return []

def calculate_complexity_from_code(code):
    return cc_visit(code)

def clone_and_analyze_repo(repo_url):
    repo_name = repo_url.split("/")[-1]
    os.system(f"git clone {repo_url} {repo_name}")
    for root, _, files in os.walk(repo_name):
        for file in files:
            if file.endswith('.py'):
                with open(os.path.join(root, file), 'r') as f:
                    code = f.read()
                complexity = calculate_complexity_from_code(code)
                logger.info(f"Complexity for {file}: {complexity}")

# Example usage
repositories = fetch_open_source_repositories()
for repo in repositories:
    print(repo['name'], repo['html_url'])
    clone_and_analyze_repo(repo['html_url'])

def find_repos_with_tests(language='Python', stars=500):
    url = f'https://api.github.com/search/repositories?q=language:{language}+stars:>{stars}+has_tests'
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['items']
    else:
        logger.error(f"Failed to fetch repositories with tests from GitHub: {response.status_code}")
        return []

# Example usage for fetching repos with tests
repos_with_tests = find_repos_with_tests()
for repo in repos_with_tests:
    print(repo['name'], repo['html_url'])
    clone_and_analyze_repo(repo['html_url'])

def run_pylint_on_repo(repo_path):
    try:
        result = subprocess.run(['pylint', repo_path], capture_output=True, text=True)
        logger.info(f"Pylint report for {repo_path}: {result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to run Pylint on {repo_path}: {e}")

# Example usage:
run_pylint_on_repo(repo_name)
