from hermod.core.auto_debugger import analyze_logs_and_suggest_fixes
from hermod.core.self_refactor import refactor_module_with_validation

def self_debugging_loop(module_path):
    """
    Implements the self-debugging loop. Hermod analyzes logs, refactors, and validates changes.
    """
    logger.info(f"Starting self-debugging loop for {module_path}.")

    # Step 1: Analyze logs for errors and suggest fixes
    suggested_fix = analyze_logs_and_suggest_fixes()

    if suggested_fix:
        # Step 2: Apply the suggested fix if available
        apply_suggested_fix(module_path, suggested_fix)

    # Step 3: Refactor the module and validate the changes
    refactor_module_with_validation(module_path, auto_approve=True)

    logger.info("Self-debugging loop completed.")

# Example usage
if __name__ == "__main__":
    self_debugging_loop('hermod/core/example_module.py')
