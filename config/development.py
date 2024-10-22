# config/development.py

CONFIG = {
    'proj_12345': {  # Replace with your actual project_id
        # RoBERTa Model Configurations
        'roberta_model.tokenizer_name': 'roberta-base',
        'roberta_model.embedding_model_name': 'roberta-base',
        'roberta_model.classification_model_name': 'roberta-base',
        'roberta_model.qa_model_name': 'deepset/roberta-base-squad2',  # Example QA model fine-tuned on SQuAD2

        # Intent Classifier Configurations
        'intent_classifier.spacy_model': 'en_core_web_sm',
        'intent_classifier.model_path': 'models/intent_classifier_proj_12345.joblib',

        # Entity Recognizer Configurations
        'entity_recognizer.spacy_model': 'en_core_web_sm',
        'entity_recognizer.model_path': 'models/entity_recognizer_proj_12345.joblib',

        # Emotion Recognizer Configurations
        'emotion_recognizer.spacy_model': 'en_core_web_sm',
        'emotion_recognizer.model_path': 'models/emotion_recognizer_proj_12345.joblib',

        # Ethical Decision Maker Configurations
        'ethical_decision_maker.spacy_model': 'en_core_web_sm',
        'ethical_decision_maker.model_path': 'models/ethical_decision_maker_proj_12345.joblib',

        # Explainable AI Configurations
        'explainable_ai.model_path': 'models/intent_classifier_proj_12345.joblib',  # Path to the model needing explanations

        # Add any additional module configurations here
    },
    # Add other projects as needed
}
