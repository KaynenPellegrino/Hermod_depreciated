"""
Module: workflow.py

Defines the workflow for end-to-end application development and optimization.
"""

import os
import subprocess
from hermod.core.code_generation import generate_code, save_code
from hermod.core.metrics_collector import collect_performance_metrics, get_module_metrics
from hermod.core.project_generator import generate_project
from hermod.core.test_generator import generate_tests_for_module
from hermod.core.self_refactor import refactor_module
from hermod.core.auto_debugger import analyze_logs_and_suggest_fixes
from hermod.core.deploy import deploy_project
from hermod.core.documentation import generate_documentation
from hermod.core.security import run_security_scan
from hermod.core.test_logger import store_and_update_tests
from hermod.core.version_control import rollback_changes
from hermod.utils.logger import setup_logger
from hermod.core.ai_refactor_system import AIRefactorSystem
from hermod.core.self_refactor import refactor_module_with_optimization
from hermod.core.test_manager import run_tests_and_validate
from hermod.core.performance_monitor import monitor_performance_after_refactor
from hermod.core.approval_manager import assess_risk_and_approve
import cProfile
import subprocess
from hermod.core.self_refactor import refactor_module
from hermod.core.self_tuner import analyze_and_adjust_strategy
from hermod.utils.logger import setup_logger

# Initialize logger
logger = setup_logger()

def log_test_results(module_path, log_file="logs/test_results.log"):
    """
    Logs the results of the generated tests for a module.

    Args:
        module_path (str): The path to the module being tested.
        log_file (str): The path to the log file where test results will be stored.
    """
    logger.info(f"Running tests for {module_path}")
    test_filename = os.path.join('tests', f"test_{os.path.basename(module_path)}")

    if not os.path.exists(test_filename):
        logger.error(f"Test file {test_filename} does not exist.")
        return

    # Run the tests using Python's unittest framework
    result = subprocess.run(['python', '-m', 'unittest', test_filename], capture_output=True, text=True)

    # Log the results
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"Test Results for {module_path}:\n")
        f.write(result.stdout)
        f.write(result.stderr)

    if result.returncode == 0:
        logger.info(f"All tests passed for {module_path}")
    else:
        logger.error(f"Tests failed for {module_path}. Check {log_file} for details.")


def profile_code(module_path):
    profiler = cProfile.Profile()
    profiler.enable()
    exec(open(module_path).read())
    profiler.disable()
    profiler.print_stats()

def develop_application(specification, project_name="GeneratedProject", language="Python", auto_approve=False):
    logger.info("Starting application development workflow for project: %s", project_name)

    # Step 1: Project Generation
    generate_project(specification, project_name=project_name, language=language)

    # Step 2: Initialize AI Refactor System
    ai_advisor = AIRefactorSystem()

    modules_dir = os.path.join(project_name, 'core')
    if os.path.exists(modules_dir):
        for module_file in os.listdir(modules_dir):
            if module_file.endswith('.py') and not module_file.startswith('__'):
                module_path = os.path.join(modules_dir, module_file)

                # Collect initial metrics (e.g., complexity, test results)
                initial_metrics = get_module_metrics(module_path)

                # Get current state from metrics
                state = ai_advisor.get_state(initial_metrics)

                # Choose an action (refactor or do nothing)
                action = ai_advisor.choose_action(state)

                if action == 'refactor':
                    refactor_module_with_optimization(module_path, auto_approve)

                # Run tests and collect new metrics
                test_passed = run_tests_and_validate(module_path)
                performance_improved = monitor_performance_after_refactor(module_path)

                # Calculate reward based on test and performance results
                reward = ai_advisor.give_reward(initial_metrics, action, test_passed, performance_improved)

                # Get new state
                new_metrics = get_module_metrics(module_path)
                next_state = ai_advisor.get_state(new_metrics)

                # Update Q-Table
                ai_advisor.update_q_table(state, action, reward, next_state)

    # Step 3: Profiling the code
    profile_code('hermod/main.py')

    # Step 4: Run Tests
    test_results = run_tests_in_project(project_name)
    if not test_results['passed']:
        rollback_changes(project_name)

    # Step 5: Documentation Generation
    generate_documentation(project_name)

    # Step 6: Security Scanning
    run_security_scan(project_name)

    # Step 7: Deployment
    deploy_project(project_name)

    # Step 8: Self-Optimization and Self-Correction
    self_optimize_and_correct(project_name, language, auto_approve)

    logger.info("Application development workflow completed successfully.")

def refactor_module_in_project(project_name, auto_approve=False):
    """
    Refactors all modules in the project directory.

    Args:
        project_name (str): The name of the project directory.
        auto_approve (bool): Whether the AI should automatically approve optimizations.
    """
    logger.info(f"Refactoring all modules in project: {project_name}")
    modules_dir = os.path.join(project_name)
    if os.path.exists(modules_dir):
        for root, _, files in os.walk(modules_dir):
            for file in files:
                if file.endswith('.py') and not file.startswith('__'):
                    module_path = os.path.join(root, file)
                    refactor_module(module_path, iterations=1, approval_required=not auto_approve)
                    logger.info(f"Refactored: {module_path}")
    else:
        logger.warning(f"Modules directory {modules_dir} does not exist.")


def run_tests_in_project(project_name):
    """
    Runs tests on all test files in the project directory and returns the results.

    Args:
        project_name (str): The name of the project directory.

    Returns:
        dict: A dictionary containing the test results.
    """
    logger.info(f"Running tests for project: {project_name}")
    tests_dir = os.path.join(project_name, 'tests')

    if os.path.exists(tests_dir):
        # Run all tests in the test directory using unittest
        result = subprocess.run(['python', '-m', 'unittest', 'discover', tests_dir], capture_output=True, text=True)

        # Log the test output
        if result.returncode == 0:
            logger.info("All tests passed.")
            return {"passed": True, "output": result.stdout}
        else:
            logger.error("Some tests failed.")
            return {"passed": False, "output": result.stderr}
    else:
        logger.warning(f"Tests directory {tests_dir} does not exist.")
        return {"passed": False, "output": "Tests directory not found."}


def self_optimize_and_correct(project_name, language, auto_approve):
    """
    Optimizes and corrects the generated project code based on logs, performance, and linting.

    Args:
        project_name (str): The name of the project directory.
        language (str): The programming language to use.
        auto_approve (bool): Whether to automatically approve optimizations.
    """
    logger.info("Starting self-optimization and correction for project: %s", project_name)

    # Step 1: Analyze logs and suggest fixes
    logs_path = os.path.join('logs', 'hermod.log')
    analyze_logs_and_suggest_fixes(log_file=logs_path)

    # Step 2: Optimize the code using AI-based refactoring
    modules_dir = os.path.join(project_name, 'core')
    if os.path.exists(modules_dir):
        for module_file in os.listdir(modules_dir):
            if module_file.endswith('.py') and not module_file.startswith('__'):
                module_path = os.path.join(modules_dir, module_file)

                # Refactor the code using AI-based optimization
                logger.info(f"Optimizing and refactoring module: {module_path}")
                refactor_module(module_path)

                # If auto-approve is enabled, commit changes directly
                if auto_approve:
                    logger.info(f"Automatically approved refactoring for: {module_path}")
                else:
                    logger.info(f"Awaiting approval for refactoring: {module_path}")
                    # Here we could implement user feedback to approve or reject changes

    logger.info("Self-optimization and correction completed.")


def iterative_optimization(project_name, iterations=2, auto_approve=False):
    """
    Performs iterative optimization and refactoring based on test results and performance metrics.

    Args:
        project_name (str): The name of the project directory.
        iterations (int): Number of optimization iterations.
        auto_approve (bool): Whether to auto-approve changes.
    """
    logger.info("Starting iterative optimization for project: %s", project_name)

    for i in range(iterations):
        logger.info(f"Optimization iteration {i + 1}")

        # Step 1: Refactor Code
        refactor_module_in_project(project_name, auto_approve)

        # Step 2: Run Tests
        test_results = run_tests_in_project(project_name)
        if not test_results['passed']:
            logger.error("Tests failed after refactoring. Rolling back changes.")
            rollback_changes(project_name)
            break

        # Step 3: Monitor Performance
        performance_metrics = collect_performance_metrics(project_name)
        logger.info(f"Performance metrics: {performance_metrics}")

        # Optionally, use metrics to decide if further optimization is needed

    logger.info("Iterative optimization completed.")


def generate_and_run_tests_for_module(module_path):
    """
    Integrates dynamic test generation, storage, and test logging for a module.

    Args:
        module_path (str): The path to the module to generate tests for.
    """
    # Generate tests for the module
    test_code = generate_tests_for_module(module_path)

    if test_code:
        # Store and update the generated test cases
        store_and_update_tests(test_code, module_path)

        # Log the test results
        log_test_results(module_path)

    else:
        logger.error(f"Test generation failed for {module_path}")


def self_optimize_with_metrics(project_name, module_path, initial_strategy):
    """
    Perform self-optimization with metrics collection and strategy adjustment.

    Args:
        project_name (str): The name of the project.
        module_path (str): The path to the module being optimized.
        initial_strategy (dict): The initial refactoring strategy.
    """
    logger.info(f"Starting self-optimization for {module_path} in {project_name}...")

    # Initial refactoring
    refactor_module(module_path, iterations=initial_strategy['iterations'], approval_required=False)

    # Collect metrics and adjust strategy
    updated_strategy = analyze_and_adjust_strategy(module_path, initial_strategy)

    # Reapply refactoring based on updated strategy
    refactor_module(module_path, iterations=updated_strategy['iterations'], approval_required=False)

    logger.info(f"Self-optimization completed for {module_path}.")
