# src/modules/auto_ml/hyperparameter_tuner.py

import logging
import os
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Categorical, Integer
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

from src.modules.data_management.staging import DataStorage

# Configure logging with RotatingFileHandler to prevent log files from becoming too large
from logging.handlers import RotatingFileHandler

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = RotatingFileHandler(
    os.path.join('logs', 'hyperparameter_tuner.log'),
    maxBytes=5 * 1024 * 1024,  # 5 MB
    backupCount=5
)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)


class HyperparameterTuner:
    """
    Automates the tuning of hyperparameters for machine learning models,
    searching for optimal configurations to improve model performance.
    Supports Grid Search, Random Search, and Bayesian Optimization.
    """

    def __init__(self):
        """
        Initializes the HyperparameterTuner with necessary configurations.
        """
        try:
            self.data_storage = DataStorage()
            logger.info("HyperparameterTuner initialized successfully.")
        except Exception as e:
            logger.exception(f"Failed to initialize HyperparameterTuner: {e}")
            raise e

    def tune_with_grid_search(self, model: BaseEstimator, param_grid: Dict[str, List[Any]],
                              X_train: pd.DataFrame, y_train: pd.Series, scoring: Optional[str] = None,
                              cv: int = 5, n_jobs: int = -1) -> GridSearchCV:
        """
        Performs hyperparameter tuning using Grid Search.

        :param model: The machine learning model to tune.
        :param param_grid: Dictionary with parameters names as keys and lists of parameter settings to try as values.
        :param X_train: Training features.
        :param y_train: Training labels.
        :param scoring: A string or scorer callable object/function with signature scorer(estimator, X, y).
        :param cv: Determines the cross-validation splitting strategy.
        :param n_jobs: Number of jobs to run in parallel.
        :return: Fitted GridSearchCV object.
        """
        logger.info("Starting Grid Search hyperparameter tuning.")
        try:
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                scoring=scoring,
                cv=cv,
                n_jobs=n_jobs,
                verbose=1
            )
            grid_search.fit(X_train, y_train)
            logger.info(f"Grid Search completed. Best parameters: {grid_search.best_params_}")
            return grid_search
        except Exception as e:
            logger.error(f"Grid Search failed: {e}")
            raise e

    def tune_with_random_search(self, model: BaseEstimator, param_distributions: Dict[str, List[Any]],
                                X_train: pd.DataFrame, y_train: pd.Series, scoring: Optional[str] = None,
                                cv: int = 5, n_iter: int = 50, n_jobs: int = -1,
                                random_state: Optional[int] = None) -> RandomizedSearchCV:
        """
        Performs hyperparameter tuning using Random Search.

        :param model: The machine learning model to tune.
        :param param_distributions: Dictionary with parameters names as keys and distributions or lists of parameters to try.
        :param X_train: Training features.
        :param y_train: Training labels.
        :param scoring: A string or scorer callable object/function with signature scorer(estimator, X, y).
        :param cv: Determines the cross-validation splitting strategy.
        :param n_iter: Number of parameter settings that are sampled.
        :param n_jobs: Number of jobs to run in parallel.
        :param random_state: Controls the randomness of the estimator.
        :return: Fitted RandomizedSearchCV object.
        """
        logger.info("Starting Random Search hyperparameter tuning.")
        try:
            random_search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_distributions,
                n_iter=n_iter,
                scoring=scoring,
                cv=cv,
                random_state=random_state,
                n_jobs=n_jobs,
                verbose=1
            )
            random_search.fit(X_train, y_train)
            logger.info(f"Random Search completed. Best parameters: {random_search.best_params_}")
            return random_search
        except Exception as e:
            logger.error(f"Random Search failed: {e}")
            raise e

    def tune_with_bayesian_optimization(self, model: BaseEstimator, search_spaces: Dict[str, Any],
                                        X_train: pd.DataFrame, y_train: pd.Series, scoring: Optional[str] = None,
                                        cv: int = 5, n_iter: int = 50, n_jobs: int = -1,
                                        random_state: Optional[int] = None) -> Optional[BayesSearchCV]:
        """
        Performs hyperparameter tuning using Bayesian Optimization.

        :param model: The machine learning model to tune.
        :param search_spaces: Dictionary with parameter names (str) as keys and skopt.space.Dimension instances as values.
        :param X_train: Training features.
        :param y_train: Training labels.
        :param scoring: A string or scorer callable object/function with signature scorer(estimator, X, y).
        :param cv: Determines the cross-validation splitting strategy.
        :param n_iter: Number of parameter settings that are sampled.
        :param n_jobs: Number of jobs to run in parallel.
        :param random_state: Controls the randomness of the estimator.
        :return: Fitted BayesSearchCV object or None if skopt is not available.
        """
        if not SKOPT_AVAILABLE:
            logger.error("skopt is not installed. Please install scikit-optimize to use Bayesian Optimization.")
            return None

        logger.info("Starting Bayesian Optimization hyperparameter tuning.")
        try:
            bayes_search = BayesSearchCV(
                estimator=model,
                search_spaces=search_spaces,
                n_iter=n_iter,
                scoring=scoring,
                cv=cv,
                n_jobs=n_jobs,
                random_state=random_state,
                verbose=1
            )
            bayes_search.fit(X_train, y_train)
            logger.info(f"Bayesian Optimization completed. Best parameters: {bayes_search.best_params_}")
            return bayes_search
        except Exception as e:
            logger.error(f"Bayesian Optimization failed: {e}")
            raise e

    def save_best_model(self, searcher: Union[GridSearchCV, RandomizedSearchCV, BayesSearchCV], model_name: str,
                        file_path: str) -> bool:
        """
        Saves the best estimator from the hyperparameter search to a file.

        :param searcher: Fitted hyperparameter search object.
        :param model_name: Name identifier for the model.
        :param file_path: Destination file path.
        :return: True if saving is successful, False otherwise.
        """
        logger.info(f"Saving best model '{model_name}' to '{file_path}'.")
        try:
            best_model = searcher.best_estimator_
            self.data_storage.save_model(best_model, file_path=file_path)
            logger.info(f"Best model '{model_name}' saved successfully to '{file_path}'.")
            return True
        except NotFittedError as e:
            logger.error(f"Model is not fitted yet: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to save best model '{model_name}': {e}")
            return False

    def get_best_params(self, searcher: Union[GridSearchCV, RandomizedSearchCV, BayesSearchCV]) -> Dict[str, Any]:
        """
        Retrieves the best hyperparameters found during the search.

        :param searcher: Fitted hyperparameter search object.
        :return: Dictionary of best parameters.
        """
        logger.info("Retrieving best hyperparameters.")
        try:
            best_params = searcher.best_params_
            logger.info(f"Best hyperparameters: {best_params}")
            return best_params
        except Exception as e:
            logger.error(f"Failed to retrieve best hyperparameters: {e}")
            raise e

    def evaluate_best_model(self, searcher: Union[GridSearchCV, RandomizedSearchCV, BayesSearchCV],
                            X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluates the best model from the hyperparameter search on test data.

        :param searcher: Fitted hyperparameter search object.
        :param X_test: Test features.
        :param y_test: Test labels.
        :return: Dictionary containing evaluation metrics.
        """
        logger.info("Evaluating best model on test data.")
        try:
            best_model = searcher.best_estimator_
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')

            metrics = {
                'test_accuracy': accuracy,
                'test_f1_score': f1
            }

            logger.info(f"Best model evaluation metrics: {metrics}")
            return metrics
        except NotFittedError as e:
            logger.error(f"Model is not fitted yet: {e}")
            raise e
        except Exception as e:
            logger.error(f"Failed to evaluate best model: {e}")
            raise e

    # --------------------- Example Usage --------------------- #

def example_usage(self):
    """
    Demonstrates example usage of the HyperparameterTuner class.
    """
    import pandas as pd
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from skopt.space import Categorical, Integer

    try:
        # Initialize HyperparameterTuner
        tuner = HyperparameterTuner()

        # Load dataset
        iris = load_iris()
        X = pd.DataFrame(iris.data, columns=iris.feature_names)
        y = pd.Series(iris.target)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define model
        model = RandomForestClassifier(random_state=42)

        # Define parameter grids/distributions
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }

        param_distributions = {
            'n_estimators': Integer(50, 200),
            'max_depth': Integer(5, 50),
            'min_samples_split': Integer(2, 20),
            'bootstrap': [True, False]
        }

        # Grid Search
        grid_search = tuner.tune_with_grid_search(
            model=model,
            param_grid=param_grid,
            X_train=X_train,
            y_train=y_train,
            scoring='accuracy',
            cv=5,
            n_jobs=-1
        )

        # Random Search
        random_search = tuner.tune_with_random_search(
            model=model,
            param_distributions=param_distributions,
            X_train=X_train,
            y_train=y_train,
            scoring='accuracy',
            cv=5,
            n_iter=20,
            n_jobs=-1,
            random_state=42
        )

        # Bayesian Optimization
        if SKOPT_AVAILABLE:
            search_spaces = {
                'n_estimators': Integer(50, 200),
                'max_depth': Integer(5, 50),
                'min_samples_split': Integer(2, 20),
                'bootstrap': Categorical([True, False])
            }
            bayes_search = tuner.tune_with_bayesian_optimization(
                model=model,
                search_spaces=search_spaces,
                X_train=X_train,
                y_train=y_train,
                scoring='accuracy',
                cv=5,
                n_iter=20,
                n_jobs=-1,
                random_state=42
            )
        else:
            bayes_search = None

        # Save best models
        tuner.save_best_model(grid_search, 'rf_grid_search', 'models/rf_grid_search.joblib')
        tuner.save_best_model(random_search, 'rf_random_search', 'models/rf_random_search.joblib')
        if bayes_search:
            tuner.save_best_model(bayes_search, 'rf_bayes_search', 'models/rf_bayes_search.joblib')

        # Evaluate best models
        grid_metrics = tuner.evaluate_best_model(grid_search, X_test, y_test)
        random_metrics = tuner.evaluate_best_model(random_search, X_test, y_test)
        if bayes_search:
            bayes_metrics = tuner.evaluate_best_model(bayes_search, X_test, y_test)
        else:
            bayes_metrics = None

        # Print evaluation metrics
        print("Grid Search Best Model Metrics:", grid_metrics)
        print("Random Search Best Model Metrics:", random_metrics)
        if bayes_metrics:
            print("Bayesian Optimization Best Model Metrics:", bayes_metrics)

    except Exception as e:
        logger.exception(f"Error in example usage: {e}")

# --------------------- Main Execution --------------------- #

if __name__ == "__main__":
    # Run the hyperparameter tuner example
    example_usage()
