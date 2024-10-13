# data_management/data_preprocessor.py

import logging
from typing import Optional, List, Any, Dict
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
    LabelEncoder,
    OrdinalEncoder
)
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Configure logging with RotatingFileHandler to prevent log files from becoming too large
from logging.handlers import RotatingFileHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler = RotatingFileHandler('logs/hermod_data_preprocessor.log', maxBytes=10 ** 6, backupCount=5)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)


class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts text features using TF-IDF vectorization.
    """

    def __init__(self, text_columns: List[str], max_features: int = 1000):
        self.text_columns = text_columns
        self.max_features = max_features
        self.vectorizers = {}

    def fit(self, X, y=None):
        for col in self.text_columns:
            vectorizer = TfidfVectorizer(max_features=self.max_features)
            vectorizer.fit(X[col].astype(str))
            self.vectorizers[col] = vectorizer
        return self

    def transform(self, X):
        text_features = []
        for col in self.text_columns:
            vectorizer = self.vectorizers[col]
            transformed = vectorizer.transform(X[col].astype(str))
            df_vector = pd.DataFrame(
                transformed.toarray(),
                columns=[f"{col}_tfidf_{i}" for i in range(transformed.shape[1])],
                index=X.index
            )
            text_features.append(df_vector)
        if text_features:
            return pd.concat(text_features, axis=1)
        else:
            return pd.DataFrame(index=X.index)


class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts date features such as year, month, day, hour, etc.
    """

    def __init__(self, date_columns: List[str]):
        self.date_columns = date_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        date_features = []
        for col in self.date_columns:
            X[col] = pd.to_datetime(X[col], errors='coerce')
            date_features.append(pd.DataFrame({
                f"{col}_year": X[col].dt.year,
                f"{col}_month": X[col].dt.month,
                f"{col}_day": X[col].dt.day,
                f"{col}_dayofweek": X[col].dt.dayofweek,
                f"{col}_hour": X[col].dt.hour
            }, index=X.index))
        if date_features:
            return pd.concat(date_features, axis=1)
        else:
            return pd.DataFrame(index=X.index)


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Selects a subset of features based on provided column names.
    """

    def __init__(self, features: List[str]):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.features]


class DataPreprocessor:
    """
    Implements data preprocessing steps required before training models, such as feature extraction,
    encoding categorical variables, and scaling. Prepares data to be in the right format and structure for modeling.
    """

    def __init__(self):
        logger.info("DataPreprocessor initialized.")
        self.pipeline = None

    def build_preprocessing_pipeline(
            self,
            numerical_features: List[str],
            categorical_features: List[str],
            text_features: Optional[List[str]] = None,
            date_features: Optional[List[str]] = None,
            pca_components: Optional[int] = None
    ) -> Pipeline:
        """
        Builds a preprocessing pipeline using scikit-learn's Pipeline and ColumnTransformer.

        :param numerical_features: List of numerical column names
        :param categorical_features: List of categorical column names
        :param text_features: List of text column names for feature extraction
        :param date_features: List of date column names for feature extraction
        :param pca_components: Number of PCA components for dimensionality reduction
        :return: A scikit-learn Pipeline object
        """
        logger.info("Building preprocessing pipeline.")

        transformers = []

        # Numerical pipeline
        if numerical_features:
            numerical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])
            transformers.append(('num', numerical_pipeline, numerical_features))
            logger.info(f"Added numerical pipeline for features: {numerical_features}")

        # Categorical pipeline
        if categorical_features:
            categorical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ])
            transformers.append(('cat', categorical_pipeline, categorical_features))
            logger.info(f"Added categorical pipeline for features: {categorical_features}")

        # Text feature extraction
        if text_features:
            text_pipeline = Pipeline(steps=[
                ('tfidf', TextFeatureExtractor(text_columns=text_features, max_features=1000))
            ])
            transformers.append(('text', text_pipeline, text_features))
            logger.info(f"Added text feature extraction for features: {text_features}")

        # Date feature extraction
        if date_features:
            date_pipeline = Pipeline(steps=[
                ('date_feat', DateFeatureExtractor(date_columns=date_features))
            ])
            transformers.append(('date', date_pipeline, date_features))
            logger.info(f"Added date feature extraction for features: {date_features}")

        # Combine all transformers
        preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')

        steps = [
            ('preprocessor', preprocessor)
        ]

        # Dimensionality Reduction
        if pca_components and numerical_features:
            steps.append(('pca', PCA(n_components=pca_components)))
            logger.info(f"Added PCA with {pca_components} components.")

        # Final Pipeline
        self.pipeline = Pipeline(steps=steps)
        logger.info("Preprocessing pipeline built successfully.")
        return self.pipeline

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fits the preprocessing pipeline to the DataFrame and transforms the data.

        :param df: Raw DataFrame to preprocess
        :return: Preprocessed DataFrame ready for modeling
        """
        if self.pipeline is None:
            logger.error("Preprocessing pipeline is not built. Call build_preprocessing_pipeline first.")
            raise AttributeError("Preprocessing pipeline is not built. Call build_preprocessing_pipeline first.")

        logger.info("Fitting and transforming the data.")
        try:
            transformed = self.pipeline.fit_transform(df)
            # Get feature names after preprocessing
            feature_names = []
            for name, transformer, features in self.pipeline.named_steps['preprocessor'].transformers_:
                if name == 'num':
                    feature_names.extend(features)
                elif name == 'cat':
                    ohe = transformer.named_steps['encoder']
                    ohe_features = ohe.get_feature_names_out(features)
                    feature_names.extend(ohe_features)
                elif name == 'text':
                    # Assuming each text column creates 'max_features' tfidf features
                    for text_col in features:
                        feature_names.extend([f"{text_col}_tfidf_{i}" for i in range(
                            self.pipeline.named_steps['preprocessor'].transformers_[0][1].named_steps[
                                'tfidf'].max_features)])
                elif name == 'date':
                    # Assuming DateFeatureExtractor creates 5 new features per date column
                    for date_col in features:
                        feature_names.extend([
                            f"{date_col}_year",
                            f"{date_col}_month",
                            f"{date_col}_day",
                            f"{date_col}_dayofweek",
                            f"{date_col}_hour"
                        ])
                # Add more conditions if more transformers are added
            if self.pipeline.named_steps.get('pca'):
                pca_feature_names = [f'pca_{i}' for i in range(self.pipeline.named_steps['pca'].n_components_)]
                feature_names = pca_feature_names
            transformed_df = pd.DataFrame(transformed, columns=feature_names, index=df.index)
            logger.info("Data transformed successfully.")
            return transformed_df
        except Exception as e:
            logger.error(f"Error during preprocessing: {e}")
            raise e

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the DataFrame using the already fitted preprocessing pipeline.

        :param df: Raw DataFrame to preprocess
        :return: Preprocessed DataFrame ready for modeling
        """
        if self.pipeline is None:
            logger.error("Preprocessing pipeline is not built or fitted.")
            raise AttributeError("Preprocessing pipeline is not built or fitted.")

        logger.info("Transforming the data using the existing pipeline.")
        try:
            transformed = self.pipeline.transform(df)
            # Get feature names after preprocessing
            feature_names = []
            for name, transformer, features in self.pipeline.named_steps['preprocessor'].transformers_:
                if name == 'num':
                    feature_names.extend(features)
                elif name == 'cat':
                    ohe = transformer.named_steps['encoder']
                    ohe_features = ohe.get_feature_names_out(features)
                    feature_names.extend(ohe_features)
                elif name == 'text':
                    # Assuming each text column creates 'max_features' tfidf features
                    for text_col in features:
                        feature_names.extend(
                            [f"{text_col}_tfidf_{i}" for i in range(transformer.named_steps['tfidf'].max_features)])
                elif name == 'date':
                    # Assuming DateFeatureExtractor creates 5 new features per date column
                    for date_col in features:
                        feature_names.extend([
                            f"{date_col}_year",
                            f"{date_col}_month",
                            f"{date_col}_day",
                            f"{date_col}_dayofweek",
                            f"{date_col}_hour"
                        ])
                # Add more conditions if more transformers are added
            if self.pipeline.named_steps.get('pca'):
                pca_feature_names = [f'pca_{i}' for i in range(self.pipeline.named_steps['pca'].n_components_)]
                feature_names = pca_feature_names
            transformed_df = pd.DataFrame(transformed, columns=feature_names, index=df.index)
            logger.info("Data transformed successfully using the existing pipeline.")
            return transformed_df
        except Exception as e:
            logger.error(f"Error during transforming data: {e}")
            raise e
