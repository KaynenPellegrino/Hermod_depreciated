# src/modules/nlu/language_models/roberta_model.py

import logging
from typing import Dict, Any, List

import torch
from transformers import (
    RobertaTokenizer,
    RobertaModel,
    RobertaForSequenceClassification,
    RobertaForQuestionAnswering,
    pipeline,
)
from transformers.pipelines import Pipeline as TransformersPipeline

from src.utils.logger import get_logger
from src.utils.configuration_manager import ConfigurationManager


class RoBERTAModel:
    """
    RoBERTAModel provides methods to utilize RoBERTa for tasks like embeddings generation,
    classification, and question-answering.
    """

    def __init__(self, project_id: str):
        """
        Initializes the RoBERTAModel with configurations specific to the project.

        Args:
            project_id (str): Unique identifier for the project.
        """
        self.logger = get_logger(__name__)
        self.config_manager = ConfigurationManager()
        self.config = self.config_manager.get_configuration(project_id)
        self.project_id = project_id

        # Device Setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.debug(f"Using device: {self.device}")

        # Initialize Tokenizer
        self.tokenizer_name = self.config.get('roberta_model.tokenizer_name', 'roberta-base')
        self.logger.debug(f"Loading tokenizer: {self.tokenizer_name}")
        try:
            self.tokenizer = RobertaTokenizer.from_pretrained(self.tokenizer_name)
            self.logger.info(f"Tokenizer '{self.tokenizer_name}' loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load tokenizer '{self.tokenizer_name}': {e}", exc_info=True)
            raise

        # Initialize Embeddings Model
        self.embedding_model_name = self.config.get('roberta_model.embedding_model_name', 'roberta-base')
        self.logger.debug(f"Loading embedding model: {self.embedding_model_name}")
        try:
            self.embedding_model = RobertaModel.from_pretrained(self.embedding_model_name)
            self.embedding_model.to(self.device)
            self.embedding_model.eval()
            self.logger.info(f"Embedding model '{self.embedding_model_name}' loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load embedding model '{self.embedding_model_name}': {e}", exc_info=True)
            raise

        # Initialize Classification Model
        self.classification_model_name = self.config.get('roberta_model.classification_model_name', 'roberta-base')
        self.logger.debug(f"Loading classification model from: {self.classification_model_name}")
        try:
            self.classification_model = RobertaForSequenceClassification.from_pretrained(
                self.classification_model_name
            )
            self.classification_model.to(self.device)
            self.classification_model.eval()
            self.classification_pipeline = pipeline(
                "text-classification",
                model=self.classification_model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            self.logger.info(f"Classification model loaded successfully from '{self.classification_model_name}'.")
        except Exception as e:
            self.logger.error(f"Failed to load classification model from '{self.classification_model_name}': {e}", exc_info=True)
            raise

        # Initialize Question-Answering Model
        self.qa_model_name = self.config.get('roberta_model.qa_model_name', 'deepset/roberta-base-squad2')
        self.logger.debug(f"Loading QA model from: {self.qa_model_name}")
        try:
            self.qa_model = RobertaForQuestionAnswering.from_pretrained(self.qa_model_name)
            self.qa_model.to(self.device)
            self.qa_model.eval()
            self.qa_pipeline: TransformersPipeline = pipeline(
                "question-answering",
                model=self.qa_model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            self.logger.info(f"QA model loaded successfully from '{self.qa_model_name}'.")
        except Exception as e:
            self.logger.error(f"Failed to load QA model from '{self.qa_model_name}': {e}", exc_info=True)
            raise

    def generate_embeddings(self, text: str) -> Dict[str, Any]:
        """
        Generates RoBERTa embeddings for the given text.

        Args:
            text (str): Input text.

        Returns:
            Dict[str, Any]: Dictionary containing the embeddings.
        """
        self.logger.debug(f"Generating embeddings for text: {text}")
        try:
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
                # Use the [CLS] token representation (for RoBERTa, it's <s> token)
                cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy().tolist()
            self.logger.debug("Embeddings generated successfully.")
            return {
                "status": "success",
                "embeddings": cls_embedding
            }
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}", exc_info=True)
            return {
                "status": "error",
                "message": "An error occurred while generating embeddings."
            }

    def classify_text(self, text: str, return_all_scores: bool = False) -> Dict[str, Any]:
        """
        Classifies the given text into predefined categories using RoBERTa.

        Args:
            text (str): Input text.
            return_all_scores (bool, optional): Whether to return scores for all labels. Defaults to False.

        Returns:
            Dict[str, Any]: Dictionary containing the classification result.
        """
        self.logger.debug(f"Classifying text: {text}")
        try:
            classification_results = self.classification_pipeline(text, return_all_scores=return_all_scores)
            self.logger.debug(f"Classification results: {classification_results}")
            return {
                "status": "success",
                "classification": classification_results
            }
        except Exception as e:
            self.logger.error(f"Error classifying text: {e}", exc_info=True)
            return {
                "status": "error",
                "message": "An error occurred while classifying text."
            }

    def answer_question(self, question: str, context: str) -> Dict[str, Any]:
        """
        Answers a question based on the provided context using RoBERTa.

        Args:
            question (str): The question to answer.
            context (str): The context in which to find the answer.

        Returns:
            Dict[str, Any]: Dictionary containing the answer and its score.
        """
        self.logger.debug(f"Answering question: '{question}' with context: '{context}'")
        try:
            qa_result = self.qa_pipeline(question=question, context=context)
            self.logger.debug(f"QA result: {qa_result}")
            return {
                "status": "success",
                "answer": qa_result
            }
        except Exception as e:
            self.logger.error(f"Error answering question: {e}", exc_info=True)
            return {
                "status": "error",
                "message": "An error occurred while answering the question."
            }


if __name__ == "__main__":
    # Example usage
    project_id = "proj_12345"
    roberta_model = RoBERTAModel(project_id)

    # Example text for embeddings
    text = "Transformers provide state-of-the-art natural language processing capabilities."
    embeddings = roberta_model.generate_embeddings(text)
    print(f"Embeddings:\n{embeddings}\n")

    # Example text classification
    classification = roberta_model.classify_text(text, return_all_scores=True)
    print(f"Classification:\n{classification}\n")

    # Example question answering
    context = (
        "Transformers are models that use self-attention mechanisms to process input data. "
        "They have achieved state-of-the-art results in various NLP tasks such as translation, "
        "summarization, and question-answering."
    )
    question = "What do transformers use to process input data?"
    qa_result = roberta_model.answer_question(question, context)
    print(f"Question Answering:\n{qa_result}\n")
