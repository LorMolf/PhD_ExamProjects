"""
train.py

Worker script that:
1. Finds an unclaimed hyperparams_*.json in /app/shared_volume.
2. Loads the relevant hyperparams, trains the chosen model (SVM / XGB / LGBM)
   on the Breast Cancer dataset (via sklearn).
3. Outputs results_*.json with "loss" = 1 - accuracy.
4. Exits after one set.

If you scale the worker service, multiple containers can pick up multiple hyperparam sets in parallel.
"""

import os
import time
import json
import logging
import numpy as np
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import torch

# Possible models
from sklearn.svm import SVC
from xgboost import XGBClassifier
import xgboost as xgb
from lightgbm import LGBMClassifier

SHARED_DIR = Path("/app/shared_volume")
MODEL_CHOICE = os.environ.get("MODEL_CHOICE", "svm").lower()
SEED = int(os.environ.get("SEED", "42"))

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_unclaimed_hyperparams():
    """
    Return the first hyperparams_*.json that doesn't have a matching results_*.json,
    or None if none found.
    """
    logger.debug("Searching for unclaimed hyperparams files.")
    files = sorted(SHARED_DIR.glob("hyperparams_*.json"))
    for hp_file in files:
        result_file = SHARED_DIR / hp_file.name.replace("hyperparams", "results")
        if not result_file.exists():
            logger.debug(f"Found unclaimed hyperparams file: {hp_file}")
            return hp_file
    logger.debug("No unclaimed hyperparams files found.")
    return None

def train_model(params, X_train, y_train, X_test, y_test):
    """
    Trains the chosen model with the given params. Returns test accuracy.
    """

    # Set device to "cuda" if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.debug(f"Training model: {MODEL_CHOICE} with params: {params} on device: {device}")

    if MODEL_CHOICE == "svm":
        model = SVC(
            C=params["C"],
            gamma=params["gamma"],
            random_state=SEED
        )
        logger.debug("Fitting model...")
        model.fit(X_train, y_train)
        logger.debug("Model fitted. Predicting...")
        accuracy = model.score(X_test, y_test)
    elif MODEL_CHOICE == "xgb":
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        xgb_params = {
            "learning_rate": params["learning_rate"],
            "n_estimators": params["n_estimators"],
            "max_depth": params["max_depth"],
            "eval_metric": "logloss",
            "tree_method": "hist" if device == "cuda" else "auto",
            #"predictor": "gpu_predictor" if device == "cuda" else "cpu_predictor",
            "seed": SEED
        }
        model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=xgb_params["n_estimators"]
        )
        logger.debug("Model fitted. Predicting...")
        predictions = model.predict(dtest)
        predictions = np.round(predictions)  # Round predictions to binary values (0 or 1)
        accuracy = np.mean(predictions == y_test)
    elif MODEL_CHOICE == "lgbm":
        model = LGBMClassifier(
            learning_rate=params["learning_rate"],
            n_estimators=params["n_estimators"],
            num_leaves=params["num_leaves"],
            random_state=SEED,
            device_type="gpu" if device == "cuda" else "cpu",  # Use GPU if available
            gpu_use_dp=True,  # Use double precision if needed
        )
        logger.debug("Fitting model...")
        model.fit(X_train, y_train)
        logger.debug("Model fitted. Predicting...")
        accuracy = model.score(X_test, y_test)
    else:
        raise ValueError(f"Unsupported MODEL_CHOICE={MODEL_CHOICE}")

    logger.debug(f"Model training complete. Accuracy: {accuracy:.4f}")
    return accuracy


def main():
    while True:
        # Look for a hyperparams file without a corresponding results file
        hp_file = find_unclaimed_hyperparams()
        if not hp_file:
            # No available work => wait and re-check
            logger.debug("No available work. Waiting and re-checking.")
            time.sleep(30)
            continue

        logger.info(f"Found hyperparams => {hp_file.name}")

        # Derive iteration/index from file name: hyperparams_{iter}_{i}.json
        iteration_str, index_str = hp_file.stem.split("_")[1:]
        iteration = int(iteration_str)
        index = int(index_str)
        logger.debug(f"Derived iteration: {iteration}, index: {index} from file name.")

        with open(hp_file, "r") as f:
            params = json.load(f)
        logger.debug(f"Loaded hyperparameters: {params}")

        # Load dataset
        data = load_breast_cancer()
        X, y = data.data, data.target
        logger.debug("Loaded Breast Cancer dataset.")

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=SEED
        )
        logger.debug("Performed train/test split.")

        # Train model
        accuracy = train_model(params, X_train, y_train, X_test, y_test)
        loss = 1.0 - accuracy  # We want to minimize loss

        # Write results
        result_file = SHARED_DIR / f"results_{iteration}_{index}.json"
        out_data = {**params, "loss": loss}
        with open(result_file, "w") as rf:
            json.dump(out_data, rf)
        logger.info(f"Model={MODEL_CHOICE}, Accuracy={accuracy:.4f}, Loss={loss:.4f}")
        logger.info(f"Wrote {result_file.name}\n")

        # By design, one worker handles exactly one hyperparam set, then exits.
        break

if __name__ == "__main__":
    main()
