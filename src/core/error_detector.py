import re
import subprocess
import time
import tracemalloc
import ast
import astor


class ErrorDetector:

    def detect_errors(self, code):
        """
        Detects common structural errors like missing function definitions or entry points.
        """
        errors = []
        if not re.search('def\\s+\\w+\\s*\\(.*\\)\\s*:', code):
            errors.append('No function definitions found.')
        if "if __name__ == '__main__':" not in code:
            errors.append('Missing main entry point.')
        return errors

    def detect_runtime_errors(self, file_path):
        """
        Runs the code and checks for runtime errors.
        """
        try:
            subprocess.run(['python', file_path], check=True,
                capture_output=True, text=True)
            return None
        except subprocess.CalledProcessError as e:
            return f'Runtime error: {e.output}'

    def detect_performance_bottlenecks(self, file_path):
        """
        Tracks memory and execution time to detect performance bottlenecks.
        """
        tracemalloc.start()
        start_time = time.time()
        try:
            subprocess.run(['python', file_path], check=True,
                capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            return f'Runtime error while tracking performance: {e.output}'
        execution_time = time.time() - start_time
        memory_used = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        if execution_time > 2:
            return (
                f'Performance bottleneck detected: Execution time = {execution_time}s'
                )
        if memory_used > 1024 * 1024:
            return (
                f'Memory bottleneck detected: Memory used = {memory_used / (1024 * 1024)} MB'
                )
        return None

    def detect_bad_code_patterns(self, code):
        """
        Detects bad code patterns using static analysis.
        """
        tree = ast.parse(code)
        bad_patterns = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and len(node.body) > 50:
                bad_patterns.append(
                    f'Function {node.name} is too long with {len(node.body)} lines.'
                    )
        return bad_patterns
