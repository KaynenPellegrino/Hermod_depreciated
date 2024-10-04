from core.code_generator import GPTClient
from core.learning_model import LearningModel
from core.data_synthesizer import DataSynthesizer
from core.data_collector import DataCollector
from core.refactor_module import RefactorModule
from core.error_detector import ErrorDetector
from core.feedback_log import FeedbackLog
from core.version_control import VersionControl
from core.self_modification import SelfModification
from core.self_diagnostics import check_health
from core.security_checker import run_security_checks
from core.git_manager import GitManager
from core.auto_docstring import add_docstrings
from core.documentation_updater import update_readme
from threading import Thread
from core.gui_interface import HermodGUI, run_hermod_actions
from core.test_manager import TestManager
from core.ai_refinement import LocalModel, refine_gpt_prompt


class HermodAI:
    def __init__(self):
        self.gpt_client = GPTClient()
        self.version_control = VersionControl()
        self.feedback_log = FeedbackLog()
        self.code_manager = CodeManager()
        self.error_detector = ErrorDetector()
        self.git_manager = GitManager(repo_path='C:/Users/kayne/PycharmProjects/Hermod')
        self.self_modification = SelfModification(self.gpt_client)
        self.test_manager = TestManager()
        self.learning_model = LearningModel()
        self.data_synthesizer = DataSynthesizer(self.gpt_client)
        self.data_collector = DataCollector()
        self.refactor_module = RefactorModule()

    def update_code(self, project_name, success_rate, data, labels, current_prompt, feedback):
        # Step 1: Generate code
        code = self.gpt_client.generate_code(project_name)

        # Step 2: Save and version the code
        self.version_control.save_project(code, project_name)

        # Step 3: Error detection and performance analysis
        errors = self.error_detector.detect_errors(code)
        runtime_errors = self.error_detector.detect_runtime_errors(f'./projects/{project_name}/main.py')
        performance_issues = self.error_detector.detect_performance_bottlenecks(f'./projects/{project_name}/main.py')
        bad_patterns = self.error_detector.detect_bad_code_patterns(code)

        # Step 4: Log feedback
        self.feedback_log.log_performance(errors + [runtime_errors], performance_issues, bad_patterns, code)

        # Step 5: Train the learning model based on feedback
        learning_model = self.learning_model
        learning_model.log_performance(success_rate)

        # Suggest optimizations
        suggestion = learning_model.suggest_optimization()
        optimization_suggestions = self.feedback_log.analyze_feedback()
        if optimization_suggestions:
            print(f'Optimizing based on feedback: {optimization_suggestions}')
            optimization_prompt = ' '.join(optimization_suggestions)
            code = self.gpt_client.modify_code(code, optimization_prompt)

        # Step 6: Train local models
        local_model = LocalModel()
        local_model.train(data, labels)
        new_prompt = refine_gpt_prompt(current_prompt, feedback)

        # Step 7: Commit and push the changes
        commit_message = f'Updated project {project_name} with new changes.'
        self.git_manager.commit_and_push(commit_message)

        # Step 8: Auto-generate docstrings and update the README
        add_docstrings('C:/Users/kayne/PycharmProjects/Hermod/src')
        update_readme(f'Updated project functionality to include X feature.')

        return code, errors

    def modify_core(self):
        """
        Modify Hermod's core code and commit the changes.
        """
        core_file_path = '/src/ai_core.py'
        self.self_modification.modify_core_code(core_file_path)

        # Run tests to ensure modification is successful
        if self.test_manager.run_tests():
            commit_message = "Modified Hermod's core functionality."
            self.git_manager.commit_and_push(commit_message)
        else:
            print('Core modification tests failed. Rolling back.')
            self.git_manager.pull_changes()

    def rollback_code(self, project_name):
        rollback_code = self.version_control.rollback_project(project_name)
        if rollback_code:
            print('Rolled back to a previous version.')
        else:
            print('No rollback available.')
        return rollback_code

    def run_project_pipeline(self, project_name):
        # This function integrates the update_code, error detection, ML, and other processes
        project_code = self.update_code(project_name, success_rate=0.9, data=[], labels=[], current_prompt='',
                                        feedback=[])

        # Check system health and security
        run_security_checks()
        health_status = check_health()
        print(f"System Health: {health_status}")

        # Data Collection
        self.data_collector.log_data("Code Generation", "Success")

        # Train machine learning model
        accuracy = self.learning_model.train_model(data=[], labels=[])
        print(f"Model trained with accuracy: {accuracy}")

        # Generate synthetic data for the project
        synthetic_code = self.data_synthesizer.generate_synthetic_code(project_name)
        print(f"Synthesized code: {synthetic_code}")

        return project_code


def main():
    project_name = 'AdvancedAIProject'
    hermod_ai = HermodAI()
    hermod_ai.run_project_pipeline(project_name)

    gui = HermodGUI(HermodAi)
    Thread(target=run_hermod_actions, args=(gui, hermod_ai)).start()
    gui.run()


if __name__ == '__main__':
    main()
