import subprocess

def run_sonarqube_scan(project_path):
    """
    Runs a SonarQube scan on the project to evaluate code quality.
    """
    try:
        subprocess.run(['sonar-scanner', '-Dproject.home=' + project_path], check=True)
        logger.info("SonarQube scan completed successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"SonarQube scan failed: {e}")

def run_type_checks(module_path):
    result = subprocess.run(['mypy', module_path], capture_output=True, text=True)
    logger.info(f"Type check report for {module_path}: {result.stdout}")
