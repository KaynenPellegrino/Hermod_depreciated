# code_generation/language_models/syntax_checker.py

import subprocess
import os
import logging
from typing import Dict, Any

from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(
    filename='hermod_syntax_checker.log',
    level=logging.DEBUG,
    format='%(asctime)s:%(levelname)s:%(message)s'
)


class SyntaxCheckerInterface(ABC):
    """
    Interface for Syntax Checkers.
    """

    @abstractmethod
    def check_syntax(self, code: str, language: str) -> Dict[str, Any]:
        """
        Checks the syntax of the provided code.

        :param code: The code snippet to check
        :param language: The programming language of the code
        :return: Dictionary containing the result of the syntax check
        """
        pass


class PythonSyntaxChecker(SyntaxCheckerInterface):
    """
    Syntax checker for Python code.
    """

    def check_syntax(self, code: str, language: str) -> Dict[str, Any]:
        logging.info(f"Checking syntax for Python code.")
        try:
            compile(code, '<string>', 'exec')
            logging.debug("Python syntax check passed.")
            return {"success": True, "error": None}
        except SyntaxError as e:
            logging.error(f"Python syntax error: {e}")
            return {"success": False, "error": str(e)}


class JavaScriptSyntaxChecker(SyntaxCheckerInterface):
    """
    Syntax checker for JavaScript code.
    Requires Node.js and eslint installed.
    """

    def __init__(self, eslint_path: str = 'eslint'):
        """
        Initializes the JavaScriptSyntaxChecker.

        :param eslint_path: Path to the eslint executable
        """
        self.eslint_path = eslint_path
        logging.info("JavaScriptSyntaxChecker initialized.")

    def check_syntax(self, code: str, language: str) -> Dict[str, Any]:
        logging.info(f"Checking syntax for JavaScript code.")
        try:
            # Write code to a temporary file
            temp_file = 'temp_js_code.js'
            with open(temp_file, 'w') as f:
                f.write(code)

            # Run eslint on the temporary file
            result = subprocess.run([self.eslint_path, temp_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True)

            # Remove the temporary file
            os.remove(temp_file)

            if result.returncode == 0:
                logging.debug("JavaScript syntax check passed.")
                return {"success": True, "error": None}
            else:
                logging.error(f"JavaScript syntax errors: {result.stdout}")
                return {"success": False, "error": result.stdout}
        except FileNotFoundError:
            logging.error(f"eslint not found at '{self.eslint_path}'.")
            return {"success": False, "error": "eslint not found."}
        except Exception as e:
            logging.error(f"JavaScript syntax check failed: {e}")
            return {"success": False, "error": str(e)}


class JavaSyntaxChecker(SyntaxCheckerInterface):
    """
    Syntax checker for Java code.
    Requires javac installed.
    """

    def __init__(self, javac_path: str = 'javac'):
        """
        Initializes the JavaSyntaxChecker.

        :param javac_path: Path to the javac executable
        """
        self.javac_path = javac_path
        logging.info("JavaSyntaxChecker initialized.")

    def check_syntax(self, code: str, language: str) -> Dict[str, Any]:
        logging.info(f"Checking syntax for Java code.")
        try:
            # Write code to a temporary file
            temp_file = 'TempJavaCode.java'
            with open(temp_file, 'w') as f:
                f.write(code)

            # Run javac on the temporary file
            result = subprocess.run([self.javac_path, temp_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True)

            # Remove the temporary file and any generated class files
            if os.path.exists(temp_file):
                os.remove(temp_file)
            class_file = temp_file.replace('.java', '.class')
            if os.path.exists(class_file):
                os.remove(class_file)

            if result.returncode == 0:
                logging.debug("Java syntax check passed.")
                return {"success": True, "error": None}
            else:
                logging.error(f"Java syntax errors: {result.stderr}")
                return {"success": False, "error": result.stderr}
        except FileNotFoundError:
            logging.error(f"javac not found at '{self.javac_path}'.")
            return {"success": False, "error": "javac not found."}
        except Exception as e:
            logging.error(f"Java syntax check failed: {e}")
            return {"success": False, "error": str(e)}


# Example usage and test case
if __name__ == "__main__":
    # Initialize syntax checkers
    python_checker = PythonSyntaxChecker()
    js_checker = JavaScriptSyntaxChecker()
    java_checker = JavaSyntaxChecker()

    # Define code snippets
    python_code = """
def hello_world():
    print("Hello, World!")
"""

    python_code_with_error = """
def hello_world()
    print("Hello, World!")
"""

    js_code = """
function helloWorld() {
    console.log("Hello, World!");
}
"""

    js_code_with_error = """
function helloWorld() {
    console.log("Hello, World!")
}
"""

    java_code = """
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}
"""

    java_code_with_error = """
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!")
    }
}
"""

    # Check Python syntax
    print("Checking Python code syntax:")
    result = python_checker.check_syntax(python_code, "Python")
    print(result)
    result = python_checker.check_syntax(python_code_with_error, "Python")
    print(result)

    # Check JavaScript syntax
    print("\nChecking JavaScript code syntax:")
    result = js_checker.check_syntax(js_code, "JavaScript")
    print(result)
    result = js_checker.check_syntax(js_code_with_error, "JavaScript")
    print(result)

    # Check Java syntax
    print("\nChecking Java code syntax:")
    result = java_checker.check_syntax(java_code, "Java")
    print(result)
    result = java_checker.check_syntax(java_code_with_error, "Java")
    print(result)
