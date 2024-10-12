# code_generation/language_models/code_gen_model.py

import logging
from typing import Dict, Any
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(
    filename='hermod_code_gen_model.log',
    level=logging.DEBUG,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

class CodeGenModelInterface(ABC):
    """
    Interface for Code Generation Models.
    """
    @abstractmethod
    def generate_code(self, prompt: str, language: str) -> str:
        """
        Generates code based on the provided prompt and programming language.

        :param prompt: Description of the code to generate
        :param language: Programming language for the generated code
        :return: Generated code as a string
        """
        pass

class OpenAIModel(CodeGenModelInterface):
    """
    Concrete implementation of CodeGenModelInterface using OpenAI's GPT models.
    """
    def __init__(self, api_key: str, model_name: str = "text-davinci-003"):
        """
        Initializes the OpenAIModel with necessary configurations.

        :param api_key: API key for accessing OpenAI services
        :param model_name: Name of the OpenAI model to use
        """
        import openai
        self.api_key = api_key
        self.model_name = model_name
        openai.api_key = self.api_key
        logging.info(f"OpenAIModel initialized with model '{self.model_name}'.")

    def generate_code(self, prompt: str, language: str) -> str:
        """
        Generates code using OpenAI's GPT model.

        :param prompt: Description of the code to generate
        :param language: Programming language for the generated code
        :return: Generated code as a string
        """
        import openai

        logging.info(f"Generating code for language='{language}' with prompt='{prompt}'.")
        try:
            full_prompt = f"Generate a {language} code snippet for the following description:\n\n{prompt}"
            response = openai.chat.completions.create(
                engine=self.model_name,
                prompt=full_prompt,
                max_tokens=500,
                temperature=0.3,
                n=1,
                stop=None
            )
            code = response.choices[0].text.strip()
            logging.debug(f"Generated code: {code}")
            return code
        except Exception as e:
            logging.error(f"Error generating code with OpenAIModel: {e}")
            raise e

class MockCodeGenModel(CodeGenModelInterface):
    """
    Mock implementation of CodeGenModelInterface for testing purposes.
    """
    def generate_code(self, prompt: str, language: str) -> str:
        logging.info(f"Mock generating {language} code for prompt: {prompt}")
        # Return a dummy code snippet based on language
        dummy_code = f"// {language} code generated for: {prompt}\n"
        if language.lower() == "python":
            dummy_code += "def hello_world():\n    print('Hello, World!')\n"
        elif language.lower() == "javascript":
            dummy_code += "function helloWorld() {\n    console.log('Hello, World!');\n}\n"
        elif language.lower() == "java":
            dummy_code += "public class HelloWorld {\n    public static void main(String[] args) {\n        System.out.println(\"Hello, World!\");\n    }\n}\n"
        else:
            dummy_code += "// Code generation for this language is not implemented in the mock.\n"
        logging.debug(f"Mock generated code: {dummy_code}")
        return dummy_code

# Example usage and test case
if __name__ == "__main__":
    # Initialize mock CodeGenModel
    mock_model = MockCodeGenModel()

    # Define prompts
    prompts = [
        {"prompt": "Create a Python function that returns the Fibonacci sequence up to n.", "language": "Python"},
        {"prompt": "Develop a JavaScript function to validate email addresses.", "language": "JavaScript"},
        {"prompt": "Implement a Java class for a simple calculator.", "language": "Java"}
    ]

    # Generate code using MockCodeGenModel
    for item in prompts:
        code = mock_model.generate_code(item["prompt"], item["language"])
        print(f"--- Generated {item['language']} Code ---\n{code}\n")
