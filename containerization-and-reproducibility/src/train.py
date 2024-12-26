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
    logger.debug("Searching for unclaimed hyperparameters files.")
    for iteration_dir in sorted(SHARED_DIR.glob("iteration_*")):
        for hp_file in sorted(iteration_dir.glob("hyperparams_*.json")):
            result_file = iteration_dir / hp_file.name.replace("hyperparams", "results")
            if not result_file.exists():
                logger.debug(f"Found unclaimed hyperparameters file: {hp_file}")
                return hp_file
    logger.debug("No unclaimed hyperparameters files found.")
    return None

def train_model(params, X_train, y_train, X_test, y_test):
    """
    Trains the chosen model with the given params. Returns test accuracy.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.debug(f"Training model: {MODEL_CHOICE} with params: {params} on device: {device}")

    if MODEL_CHOICE == "svm":
        model = SVC(C=params["C"], gamma=params["gamma"], random_state=SEED)
        logger.debug("Fitting SVM model...")
        model.fit(X_train, y_train)
        logger.debug("SVM model fitted. Predicting...")
        accuracy = model.score(X_test, y_test)
    elif MODEL_CHOICE == "xgb":
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        xgb_params = {
            "learning_rate": params["learning_rate"],
            "n_estimators": params["n_estimators"],
            "max_depth": params["max_depth"],
            "eval_metric": "logloss",
            "tree_method": "gpu_hist" if device == "cuda" else "auto",
            "seed": SEED
        }
        logger.debug("Training XGBoost...")
        model = xgb.train(xgb_params, dtrain, num_boost_round=xgb_params["n_estimators"])
        predictions = model.predict(dtest)
        predictions = np.round(predictions)
        accuracy = np.mean(predictions == y_test)
        logger.debug("XGBoost model trained.")
    elif MODEL_CHOICE == "lgbm":
        model = LGBMClassifier(
            learning_rate=params["learning_rate"],
            n_estimators=params["n_estimators"],
            num_leaves=params["num_leaves"],
            random_state=SEED,
            device_type="gpu" if device == "cuda" else "cpu"
        )
        logger.debug("Training LightGBM...")
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        logger.debug("LightGBM model trained.")
    else:
        raise ValueError(f"Unsupported MODEL_CHOICE: {MODEL_CHOICE}")

    logger.debug(f"Model training complete. Accuracy: {accuracy:.4f}")
    return accuracy

def main():
    while True:
        hp_file = find_unclaimed_hyperparams()
        if not hp_file:
            logger.debug("No available work. Retrying in 30 seconds...")
            time.sleep(30)
            continue

        logger.info(f"Processing hyperparameters file: {hp_file}")
        with open(hp_file, "r") as f:
            params = json.load(f)
        logger.debug(f"Loaded hyperparameters: {params}")

        data = load_breast_cancer()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

        accuracy = train_model(params, X_train, y_train, X_test, y_test)
        loss = 1.0 - accuracy

        result_file = hp_file.parent / hp_file.name.replace("hyperparams", "results")
        with open(result_file, "w") as f:
            json.dump({"loss": loss, "accuracy": accuracy}, f)
        logger.info(f"Results saved to {result_file}: Loss={loss:.4f}, Accuracy={accuracy:.4f}")
        break

if __name__ == "__main__":
    main()
