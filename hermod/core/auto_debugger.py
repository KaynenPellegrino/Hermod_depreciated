import os
import re
from hermod.core.code_generation import generate_code, save_code
from hermod.utils.logger import setup_logger

# Initialize logger
logger = setup_logger()


def analyze_logs_and_suggest_fixes(log_file='logs/hermod.log'):
    """
    Analyzes error logs and suggests fixes for identified issues.
    """
    logger.info(f"Analyzing logs from {log_file}...")

    # Step 1: Read the log file
    if not os.path.exists(log_file):
        logger.error(f"Log file {log_file} does not exist.")
        return None

    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            logs = f.read()
    except Exception as e:
        logger.error(f"Failed to read log file {log_file}: {e}")
        return None

    # Step 2: Extract the last error message (You can customize this for your log format)
    error_pattern = r'ERROR.*'
    errors = re.findall(error_pattern, logs)

    if not errors:
        logger.info("No errors found in the logs.")
        return None

    last_error = errors[-1]  # Focus on the last error
    logger.info(f"Identified error: {last_error}")

    # Step 3: Use AI to generate a suggested fix
    prompt = f"Analyze the following error log and suggest a specific code fix:\n\n{last_error}"
    suggested_fix = generate_code(prompt)

    if suggested_fix:
        logger.info(f"Suggested fix:\n{suggested_fix}")
        return suggested_fix
    else:
        logger.error("Failed to generate a code fix.")
        return None


def apply_suggested_fix(module_path, suggested_fix):
    """
    Applies the suggested fix to the module if approved.
    """
    try:
        # Apply the fix to the module
        with open(module_path, 'a', encoding='utf-8') as f:
            f.write(f"\n# Applied Fix:\n{suggested_fix}")

        logger.info(f"Applied the suggested fix to {module_path}.")
    except Exception as e:
        logger.error(f"Failed to apply fix to {module_path}: {e}")


# Example usage in your main debugging workflow
if __name__ == "__main__":
    suggested_fix = analyze_logs_and_suggest_fixes()
    if suggested_fix:
        apply_suggested_fix('hermod/core/example_module.py', suggested_fix)
