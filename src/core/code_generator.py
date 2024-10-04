import openai
import os
import time

from dotenv import load_dotenv

load_dotenv()

class GPTClient:
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if openai.api_key is None:
            raise ValueError("OpenAI API key not found! Ensure that the environment variable is set.")
        print(f"Using OpenAI API Key: {openai.api_key[:5]}...")  # Only print the first 5 characters for security

    def generate_code(self, project_name):
        prompt = (
            f'Generate a Python project for {project_name}. Write simple code to get started.'
        )
        return self._call_gpt(prompt)

    def modify_code(self, current_code, modification_prompt):
        prompt = (
            f'Modify the following code:\n\n{current_code}\n\n{modification_prompt}'
        )
        return self._call_gpt(prompt)

    def _call_gpt(self, prompt):
        attempts = 3
        for attempt in range(attempts):
            try:
                response = openai.chat.completions.create(
                    model='chatgpt-4o-latest',  # Replace with 'gpt-4' if you have access
                    messages=[
                        {"role": "system", "content": "You are a coding ai."},
                        {"role": "user", "content": prompt}
                    ]
                )
                # Access the message content using dot notation
                content = response.choices[0].message["content"].strip()

                # Replace dictionary subscripting with dot notation for usage tokens
                token_dict = {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens,
                }

                print(f"Token Usage: {token_dict}")
                return content

            except openai.OpenAIError as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < attempts - 1:
                    print("Retrying after a brief delay...")
                    time.sleep(2)
                else:
                    raise Exception(f'GPT API failed after {attempts} attempts. Error: {e}')