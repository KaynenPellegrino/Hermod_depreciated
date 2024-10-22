# src/modules/nlu/training_scripts/train_qa.py

import logging
import os
import pandas as pd
from transformers import RobertaTokenizer, RobertaForQuestionAnswering, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from src.utils.configuration_manager import ConfigurationManager
from src.utils.logger import get_logger
from datasets import Dataset, Features, Sequence, Value


def prepare_dataset(df: pd.DataFrame, tokenizer: RobertaTokenizer, max_length: int = 384):
    """
    Tokenizes the dataset for QA.
    """

    def preprocess(examples):
        return tokenizer(examples['question'], examples['context'], truncation="only_second", max_length=max_length,
                         stride=128, return_overflowing_tokens=True, return_offsets_mapping=True, padding="max_length")

    return Dataset.from_pandas(df).map(preprocess, batched=True, remove_columns=df.columns.tolist())


def main():
    project_id = "proj_12345"  # Replace with your actual project ID
    config_manager = ConfigurationManager()
    config = config_manager.get_configuration(project_id)

    logger = get_logger(__name__)

    # Load training data
    training_data_path = os.path.join('data', 'processed', 'nlu_data', 'qa_dataset.json')  # Assuming JSON format
    if not os.path.exists(training_data_path):
        logger.error(f"QA training data not found at '{training_data_path}'.")
        raise FileNotFoundError(f"QA training data not found at '{training_data_path}'.")

    df = pd.read_json(training_data_path)

    # Validate required columns
    required_columns = {'question', 'context', 'answers'}
    if not required_columns.issubset(df.columns):
        logger.error(f"QA training data must contain columns: {required_columns}")
        raise ValueError(f"QA training data must contain columns: {required_columns}")

    # Split dataset
    train_df, eval_df = train_test_split(df, test_size=0.1, random_state=42)

    # Initialize tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForQuestionAnswering.from_pretrained('roberta-base')

    # Prepare datasets
    train_dataset = prepare_dataset(train_df, tokenizer)
    eval_dataset = prepare_dataset(eval_df, tokenizer)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./data/models/nlu_models/qa_model',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir='./logs/train_qa',
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    results = trainer.evaluate()
    logger.info(f"Evaluation results: {results}")

    # Save the model and tokenizer
    model.save_pretrained('./data/models/nlu_models/qa_model')
    tokenizer.save_pretrained('./data/models/nlu_models/qa_model')


if __name__ == "__main__":
    main()
