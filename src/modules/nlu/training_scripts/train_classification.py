# src/modules/nlu/training_scripts/train_classification.py

import logging
import os
import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from src.utils.configuration_manager import ConfigurationManager
from src.utils.logger import get_logger
from datasets import Dataset


def prepare_dataset(df: pd.DataFrame, tokenizer: RobertaTokenizer, max_length: int = 128):
    """
    Tokenizes the dataset.
    """
    return Dataset.from_pandas(df).map(
        lambda e: tokenizer(e['text'], truncation=True, padding='max_length', max_length=max_length),
        batched=True
    )


def main():
    project_id = "proj_12345"  # Replace with your actual project ID
    config_manager = ConfigurationManager()
    config = config_manager.get_configuration(project_id)

    logger = get_logger(__name__)

    # Load training data
    training_data_path = os.path.join('data', 'processed', 'nlu_data', 'intent_classification.csv')
    if not os.path.exists(training_data_path):
        logger.error(f"Training data not found at '{training_data_path}'.")
        raise FileNotFoundError(f"Training data not found at '{training_data_path}'.")

    df = pd.read_csv(training_data_path)

    # Encode labels
    label_list = sorted(df['intent'].unique())
    label_to_id = {label: idx for idx, label in enumerate(label_list)}
    df['label'] = df['intent'].map(label_to_id)

    # Split dataset
    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)

    # Initialize tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=len(label_list))

    # Prepare datasets
    train_dataset = prepare_dataset(train_df[['text', 'label']], tokenizer)
    eval_dataset = prepare_dataset(eval_df[['text', 'label']], tokenizer)

    train_dataset = train_dataset.rename_column("label", "labels")
    eval_dataset = eval_dataset.rename_column("label", "labels")

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./data/models/nlu_models/classification_model',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir='./logs/train_classification',
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
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
    model.save_pretrained('./data/models/nlu_models/classification_model')
    tokenizer.save_pretrained('./data/models/nlu_models/classification_model')


if __name__ == "__main__":
    main()
