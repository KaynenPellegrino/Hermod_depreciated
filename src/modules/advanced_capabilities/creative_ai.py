# src/modules/advanced_capabilities/creative_ai.py

import logging
from typing import Dict, Any, List, Optional

from src.utils.logger import get_logger
from src.utils.helpers import format_code_snippet
from src.utils.config_loader import ConfigurationManager
import requests
import json

class CreativeAI:
    """
    CreativeAI provides AI capabilities for creative tasks, such as generating design ideas,
    content creation, or artistic outputs.
    """

    def __init__(self):
        # Initialize the logger
        self.logger = get_logger(__name__)

        # Load configuration settings
        self.config = ConfigurationManager.get_config()

        # Initialize AI models or API clients
        self.ai_api_key = self.config.get('creative_ai.api_key', '')
        self.ai_api_url = self.config.get('creative_ai.api_url', 'https://api.openai.com/v1/engines/davinci-codex/completions')
        self.image_gen_api_url = self.config.get('creative_ai.image_gen_api_url', 'https://api.example.com/v1/generate-image')

        if not self.ai_api_key:
            self.logger.warning("CreativeAI API key is not set. Some functionalities may not work.")

        self.logger.info("CreativeAI initialized successfully.")

    def generate_design_ideas(self, prompt: str, max_ideas: int = 5) -> Dict[str, Any]:
        """
        Generates design ideas based on the provided prompt.

        Args:
            prompt (str): The prompt or theme for which to generate design ideas.
            max_ideas (int, optional): The number of design ideas to generate. Defaults to 5.

        Returns:
            Dict[str, Any]: A dictionary containing the status and list of design ideas.
        """
        self.logger.debug(f"Generating design ideas with prompt: '{prompt}' and max_ideas: {max_ideas}")

        try:
            # Prepare the request payload for the AI model
            payload = {
                "prompt": f"Generate {max_ideas} creative design ideas based on the following theme:\n\n{prompt}\n\nIdeas:",
                "max_tokens": 150,
                "n": 1,
                "stop": None,
                "temperature": 0.7,
            }

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.ai_api_key}"
            }

            response = requests.post(self.ai_api_url, headers=headers, data=json.dumps(payload))

            if response.status_code != 200:
                self.logger.error(f"AI API request failed with status code {response.status_code}: {response.text}")
                return {
                    "status": "error",
                    "message": "Failed to generate design ideas."
                }

            ai_response = response.json()
            ideas_text = ai_response.get('choices', [{}])[0].get('text', '').strip()

            # Split the ideas by newline or numbering
            ideas = [idea.strip() for idea in ideas_text.split('\n') if idea.strip()]
            # If less ideas than max_ideas, adjust
            ideas = ideas[:max_ideas]

            self.logger.info(f"Generated {len(ideas)} design ideas.")

            return {
                "status": "success",
                "design_ideas": ideas
            }

        except Exception as e:
            self.logger.error(f"Error in generate_design_ideas: {e}", exc_info=True)
            return {
                "status": "error",
                "message": "An error occurred while generating design ideas."
            }

    def create_content(self, topic: str, content_type: str = "article", length: str = "medium") -> Dict[str, Any]:
        """
        Creates written content based on the given topic.

        Args:
            topic (str): The topic or title for the content.
            content_type (str, optional): The type of content to create (e.g., 'article', 'blog post', 'marketing copy'). Defaults to "article".
            length (str, optional): The desired length of the content (e.g., 'short', 'medium', 'long'). Defaults to "medium".

        Returns:
            Dict[str, Any]: A dictionary containing the status and the generated content.
        """
        self.logger.debug(f"Creating content with topic: '{topic}', content_type: '{content_type}', length: '{length}'")

        try:
            # Define length parameters based on user input
            length_map = {
                "short": 300,
                "medium": 600,
                "long": 1000
            }
            max_tokens = length_map.get(length.lower(), 600)

            # Prepare the request payload for the AI model
            prompt = f"Write a {length} {content_type} about '{topic}'."
            payload = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "n": 1,
                "stop": None,
                "temperature": 0.7,
            }

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.ai_api_key}"
            }

            response = requests.post(self.ai_api_url, headers=headers, data=json.dumps(payload))

            if response.status_code != 200:
                self.logger.error(f"AI API request failed with status code {response.status_code}: {response.text}")
                return {
                    "status": "error",
                    "message": "Failed to create content."
                }

            ai_response = response.json()
            content = ai_response.get('choices', [{}])[0].get('text', '').strip()

            self.logger.info(f"Generated {content_type} on topic '{topic}'.")

            return {
                "status": "success",
                "content": content
            }

        except Exception as e:
            self.logger.error(f"Error in create_content: {e}", exc_info=True)
            return {
                "status": "error",
                "message": "An error occurred while creating content."
            }

    def generate_artistic_output(self, description: str, style: Optional[str] = None, size: Optional[str] = None) -> Dict[str, Any]:
        """
        Generates artistic outputs like images or illustrations based on the provided description.

        Args:
            description (str): The description of the artistic output to generate.
            style (Optional[str], optional): The artistic style to apply (e.g., 'impressionist', 'modern'). Defaults to None.
            size (Optional[str], optional): The desired size of the image (e.g., '512x512'). Defaults to None.

        Returns:
            Dict[str, Any]: A dictionary containing the status and the URL or binary data of the generated image.
        """
        self.logger.debug(f"Generating artistic output with description: '{description}', style: '{style}', size: '{size}'")

        try:
            # Prepare the request payload for the image generation API
            payload = {
                "description": description,
                "style": style or "realistic",
                "size": size or "512x512"
            }

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.ai_api_key}"
            }

            response = requests.post(self.image_gen_api_url, headers=headers, data=json.dumps(payload))

            if response.status_code != 200:
                self.logger.error(f"Image Generation API request failed with status code {response.status_code}: {response.text}")
                return {
                    "status": "error",
                    "message": "Failed to generate artistic output."
                }

            # Assuming the API returns a JSON with a 'url' key for the generated image
            image_response = response.json()
            image_url = image_response.get('url', '')

            if not image_url:
                self.logger.error("Image Generation API did not return an image URL.")
                return {
                    "status": "error",
                    "message": "Failed to retrieve the generated image."
                }

            self.logger.info(f"Generated artistic output available at: {image_url}")

            return {
                "status": "success",
                "image_url": image_url
            }

        except Exception as e:
            self.logger.error(f"Error in generate_artistic_output: {e}", exc_info=True)
            return {
                "status": "error",
                "message": "An error occurred while generating artistic output."
            }

    def _interact_with_ai_api(self, prompt: str, max_tokens: int = 150) -> Optional[str]:
        """
        Internal method to interact with the AI API for generating text-based content.

        Args:
            prompt (str): The prompt to send to the AI model.
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 150.

        Returns:
            Optional[str]: The generated text if successful, else None.
        """
        try:
            payload = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "n": 1,
                "stop": None,
                "temperature": 0.7,
            }

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.ai_api_key}"
            }

            response = requests.post(self.ai_api_url, headers=headers, data=json.dumps(payload))

            if response.status_code != 200:
                self.logger.error(f"AI API request failed with status code {response.status_code}: {response.text}")
                return None

            ai_response = response.json()
            generated_text = ai_response.get('choices', [{}])[0].get('text', '').strip()

            return generated_text

        except Exception as e:
            self.logger.error(f"Error in _interact_with_ai_api: {e}", exc_info=True)
            return None

    def _interact_with_image_api(self, payload: Dict[str, Any]) -> Optional[str]:
        """
        Internal method to interact with the Image Generation API.

        Args:
            payload (Dict[str, Any]): The payload to send to the image generation API.

        Returns:
            Optional[str]: The URL of the generated image if successful, else None.
        """
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.ai_api_key}"
            }

            response = requests.post(self.image_gen_api_url, headers=headers, data=json.dumps(payload))

            if response.status_code != 200:
                self.logger.error(f"Image Generation API request failed with status code {response.status_code}: {response.text}")
                return None

            image_response = response.json()
            image_url = image_response.get('url', '')

            return image_url

        except Exception as e:
            self.logger.error(f"Error in _interact_with_image_api: {e}", exc_info=True)
            return None


if __name__ == "__main__":
    # Example usage
    creative_ai = CreativeAI()

    # Example 1: Generate Design Ideas
    prompt = "Sustainable packaging for cosmetics"
    design_ideas = creative_ai.generate_design_ideas(prompt, max_ideas=3)
    print("Design Ideas:")
    print(design_ideas)

    # Example 2: Create Content
    topic = "The impact of renewable energy on global economies"
    content = creative_ai.create_content(topic, content_type="article", length="long")
    print("\nGenerated Content:")
    print(content)

    # Example 3: Generate Artistic Output
    description = "A serene landscape with mountains and a river during sunset"
    artistic_output = creative_ai.generate_artistic_output(description, style="impressionist", size="1024x1024")
    print("\nArtistic Output:")
    print(artistic_output)
