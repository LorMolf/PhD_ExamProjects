"""
train.py

Worker script that:
1. Finds an unclaimed hyperparams_*.json in /app/shared_volume/<model_name>/experiment_<timestamp>/iteration_<i>/.
2. Loads the relevant hyperparams, trains the chosen model (SVM / XGB / LGBM)
   on the Breast Cancer dataset (via sklearn).
3. Outputs results_*.json with "loss" = 1 - accuracy, "model_path", "training_time", and "worker_id".
4. Exits after processing one set.
"""

import os
import time
import json
import logging
import uuid
from pathlib import Path

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

import torch
import numpy as np

import joblib  # For saving SVM models
import re

# Constants and Environment Variables
SHARED_DIR = Path("/app/shared_volume")
MODEL_CHOICE = os.environ.get("MODEL_CHOICE", "svm").lower()
SEED = int(os.environ.get("SEED", "42"))

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_and_claim_hyperparams():
    """
    Atomically find and claim an unclaimed hyperparameters file by renaming it.
    Returns the claimed hyperparams file path if successful, None otherwise.
    """
    logger.debug("Searching for unclaimed hyperparameters files.")

    # Define the model-specific directory
    model_dir = SHARED_DIR / MODEL_CHOICE
    if not model_dir.exists():
        logger.error(f"Model directory does not exist: {model_dir}")
        return None

    # Recursively search for all experiment_* directories within the model directory
    for experiment_dir in sorted(model_dir.glob("experiment_*")):
        # Within each experiment, search for all iteration_* directories
        for iteration_dir in sorted(experiment_dir.glob("iteration_*")):
            # Within each iteration, search for all hyperparams_*.json files
            for hp_file in sorted(iteration_dir.glob("hyperparams_*.json")):
                if "processing" in hp_file.stem:
                    continue  # Skip processing files

                result_file = iteration_dir / hp_file.name.replace("hyperparams", "results")
                processing_file = iteration_dir / f"{hp_file.stem}.processing.json"

                if result_file.exists() or processing_file.exists():
                    continue  # Skip if results already exist or already being processed

                try:
                    # Attempt to atomically rename the hyperparams file to processing_file
                    hp_file.rename(processing_file)
                    logger.debug(f"Claimed hyperparameters file: {processing_file}")
                    return processing_file
                except FileNotFoundError:
                    continue  # File was processed by another worker
                except PermissionError:
                    continue  # File was processed by another worker
                except Exception as e:
                    logger.error(f"Error claiming file {hp_file}: {e}")
                    continue

    logger.debug("No unclaimed hyperparameters files found.")
    return None

def extract_index_from_processing_file(processing_file):
    """
    Extracts the index from a processing hyperparameters file.
    E.g., 'hyperparams_0.processing.json' -> '0'
    """
    match = re.match(r'hyperparams_(\d+)\.processing\.json', processing_file.name)
    if match:
        return match.group(1)
    else:
        return None

def validate_hyperparams(params, model_choice):
    """
    Validates the presence and correctness of required hyperparameters based on the model choice.
    Raises ValueError if validation fails.
    """
    logger.debug("Validating hyperparameters.")

    # Common required hyperparameters
    hyperparams = []
    if model_choice == "svm":
        hyperparams.extend(["C", "gamma"])
    elif model_choice == "xgb":
        hyperparams.extend(["max_depth", "learning_rate", "n_estimators"])
    elif model_choice == "lgbm":
        hyperparams.extend(["num_leaves", "learning_rate", "n_estimators"])
    else:
        raise ValueError(f"Unsupported MODEL_CHOICE for hyperparameter validation: {model_choice}")

    missing_params = [param for param in hyperparams if param not in params]
    if missing_params:
        raise ValueError(f"Missing required hyperparameters for {model_choice}: {missing_params}")

    # Additional validation for LightGBM's 'goss' boosting_type
    if model_choice == "lgbm":
        boosting_type = params.get('boosting_type', 'gbdt')
        if boosting_type not in ['gbdt', 'dart', 'goss']:
            raise ValueError(f"Invalid boosting_type '{boosting_type}' for LightGBM. Choose from 'gbdt', 'dart', 'goss'.")
        if boosting_type == 'goss':
            # 'goss' requires 'top_rate' and 'other_rate'
            if 'top_rate' not in params or 'other_rate' not in params:
                raise ValueError("Missing 'top_rate' or 'other_rate' for 'goss' boosting_type in LightGBM.")
            if not (0 < params['top_rate'] < 1) or not (0 < params['other_rate'] < 1):
                raise ValueError("'top_rate' and 'other_rate' must be between 0 and 1 for 'goss' boosting_type.")

    # Additional validation for XGBoost GPU parameters
    if model_choice == "xgb":
        if 'tree_method' in params:
            if params['tree_method'] not in ['auto', 'exact', 'approx', 'hist', 'gpu_hist']:
                raise ValueError(f"Invalid tree_method '{params['tree_method']}' for XGBoost.")
        # No specific GPU parameters needed beyond tree_method for basic usage

    logger.debug("Hyperparameters validation passed.")

def train_model(params, X_train, y_train, X_test, y_test, model_choice, device):
    """
    Trains the chosen model with the given params. Returns test accuracy and trained model.
    """
    logger.debug(f"Training model: {model_choice} with params: {params} on device: {device}")

    if model_choice == "svm":
        model = SVC(C=params["C"], gamma=params["gamma"], random_state=SEED)
        logger.debug("Fitting SVM model...")
        model.fit(X_train, y_train)
        logger.debug("SVM model fitted. Predicting...")
        accuracy = model.score(X_test, y_test)
    elif model_choice == "xgb":
        # Configure tree_method based on device
        if device == "cuda":
            tree_method = 'hist'
            xgb_params = {
                "learning_rate": params["learning_rate"],
                "max_depth": params["max_depth"],
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "tree_method": tree_method,
                "device" : device,
                "seed": SEED
            }
        else:
            tree_method = 'hist'
            xgb_params = {
                "learning_rate": params["learning_rate"],
                "max_depth": params["max_depth"],
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "tree_method": tree_method,
                "seed": SEED
            }

        logger.debug(f"Training XGBoost with parameters: {xgb_params}")
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        model = xgb.train(xgb_params, dtrain, num_boost_round=params["n_estimators"])
        predictions = model.predict(dtest)
        predictions = np.round(predictions)
        accuracy = accuracy_score(y_test, predictions)
        logger.debug("XGBoost model trained.")
    elif model_choice == "lgbm":
        boosting_type = params.get('boosting_type', 'gbdt')  # Default to 'gbdt' if not specified
        lgbm_params = {
            "learning_rate": params["learning_rate"],
            "n_estimators": params["n_estimators"],
            "num_leaves": params["num_leaves"],
            "random_state": SEED,
            "boosting_type": boosting_type,
            "min_data_in_leaf": params.get("min_data_in_leaf", 20),
            "min_gain_to_split": params.get("min_gain_to_split", 0.01),
            "verbose": -1  # Suppress LightGBM warnings
        }

        # Add GPU support for LightGBM if desired and available
        if 'gpu_device_id' in params and device == "cuda":
            lgbm_params["device"] = "gpu"
            lgbm_params["gpu_device_id"] = params.get("gpu_device_id", 0)

        if boosting_type == 'goss':
            lgbm_params["top_rate"] = params["top_rate"]
            lgbm_params["other_rate"] = params["other_rate"]

        model = LGBMClassifier(**lgbm_params)
        logger.debug(f"Training LightGBM with parameters: {lgbm_params}")
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        logger.debug("LightGBM model trained.")
    else:
        raise ValueError(f"Unsupported MODEL_CHOICE: {model_choice}")

    logger.debug(f"Model training complete. Accuracy: {accuracy:.4f}")
    return accuracy, model

def main():
    # Determine if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.debug(f"CUDA Available: {torch.cuda.is_available()}, Using device: {device}")

    # Try to find and claim one job
    processing_file = find_and_claim_hyperparams()

    if not processing_file:
        logger.debug("No available work. Exiting with code 0.")
        return 0  # Clean exit, no work found

    try:
        # The original hyperparams file is processing_file renamed from hyperparams_*.json
        hp_file = processing_file
        logger.info(f"Processing hyperparameters file: {hp_file}")
        with open(hp_file, "r") as f:
            params = json.load(f)
        logger.debug(f"Loaded hyperparameters: {params}")

        # Validate hyperparameters
        validate_hyperparams(params, MODEL_CHOICE)

        # Extract index to write results_i.json
        index = extract_index_from_processing_file(processing_file)
        if index is None:
            logger.error(f"Could not extract index from file name: {processing_file.name}")
            return 1  # Error exit code

        # Generate a unique worker ID for this run
        worker_id = str(uuid.uuid4())
        logger.debug(f"Assigned Worker ID: {worker_id}")

        # Load and split the dataset
        data = load_breast_cancer()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

        # Start training
        start_time = time.time()
        accuracy, model = train_model(params, X_train, y_train, X_test, y_test, MODEL_CHOICE, device)
        training_time = time.time() - start_time

        loss = 1.0 - accuracy

        # Save the model
        timestamp = int(time.time())
        model_filename = f"model_{timestamp}.model"
        model_path = hp_file.parent / model_filename
        if MODEL_CHOICE == "xgb":
            model.save_model(str(model_path))
        elif MODEL_CHOICE == "svm":
            joblib.dump(model, str(model_path))
        elif MODEL_CHOICE == "lgbm":
            model.booster_.save_model(str(model_path))
        logger.debug(f"Model saved to {model_path}")

        # Write the results
        result_file = hp_file.parent / f"results_{index}.json"
        result_data = {
            "loss": loss,
            "accuracy": accuracy,
            "model_path": str(model_path),
            "training_time": training_time,
            "worker_id": worker_id
        }
        with open(result_file, "w") as f:
            json.dump(result_data, f, indent=4)
        logger.info(f"Results saved to {result_file}: Loss={loss:.4f}, Accuracy={accuracy:.4f}, Worker_ID={worker_id}, Training_Time={training_time:.2f}s")

        # Remove the processing file after processing
        hp_file.unlink()
        logger.debug(f"Removed hyperparameters processing file: {hp_file}")

        return 0  # Success

    except Exception as e:
        logger.error(f"Error processing file: {e}")
        return 1  # Error exit code

    finally:
        # No cleanup needed since we've renamed and processed the file
        pass

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
