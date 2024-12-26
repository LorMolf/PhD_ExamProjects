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


import fcntl
import errno
from contextlib import contextmanager

SHARED_DIR = Path("/app/shared_volume")
MODEL_CHOICE = os.environ.get("MODEL_CHOICE", "svm").lower()
SEED = int(os.environ.get("SEED", "42"))

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@contextmanager
def file_lock(lock_file):
    """Context manager for file locking to ensure atomic operations."""
    lock = open(lock_file, 'wb')
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield True
    except IOError as e:
        if e.errno != errno.EAGAIN:
            raise
        yield False
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()

def find_unclaimed_hyperparams():
    """
    Atomically find and claim an unclaimed hyperparameters file.
    Returns (hp_file, lock_file) tuple if successful, (None, None) otherwise.
    """
    logger.debug("Searching for unclaimed hyperparameters files.")
    
    for iteration_dir in sorted(SHARED_DIR.glob("iteration_*")):
        for hp_file in sorted(iteration_dir.glob("hyperparams_*.json")):
            result_file = iteration_dir / hp_file.name.replace("hyperparams", "results")
            lock_file = iteration_dir / f"{hp_file.stem}.lock"
            
            if result_file.exists():
                continue
                
            try:
                # Create lock file if it doesn't exist
                lock_file.touch(exist_ok=True)
                
                with file_lock(lock_file) as acquired:
                    if acquired and not result_file.exists():
                        return hp_file, lock_file
            except Exception as e:
                logger.error(f"Error checking file {hp_file}: {e}")
                continue
    
    return None, None


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
    # Try to find and claim one job
    hp_file, lock_file = find_unclaimed_hyperparams()
    
    if not hp_file:
        logger.debug("No available work. Exiting with code 0.")
        return 0  # Clean exit, no work found
        
    try:
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
        
        return 0  # Success
        
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        return 1  # Error exit code
        
    finally:
        # Clean up lock file
        try:
            if lock_file and lock_file.exists():
                lock_file.unlink()
        except Exception as e:
            logger.error(f"Error cleaning up lock file: {e}")


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
