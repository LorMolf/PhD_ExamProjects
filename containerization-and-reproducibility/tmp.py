## train.py
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
        hp_file = find_unclaimed_hyperparams()
        if not hp_file:
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


## manager.py
"""
manager.py

Coordinates Bayesian hyperparameter optimization for either SVM, XGB, or LGBM.
Uses skopt to propose hyperparams, writes them to /app/shared_volume, waits for
workers to produce results, then updates the optimizer.
"""

import os
import time
import json
import logging
from pathlib import Path

import numpy as np
from skopt import Optimizer
from skopt.space import Real, Integer

# Environment variables
NUM_ITERATIONS = int(os.environ.get("NUM_ITERATIONS", "3"))
NUM_HYPERPARAM_SETS = int(os.environ.get("NUM_HYPERPARAM_SETS", "2"))
MODEL_CHOICE = os.environ.get("MODEL_CHOICE", "svm").lower()
SEED = int(os.environ.get("SEED", "42"))

SHARED_DIR = Path("/app/shared_volume")

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_search_space(model_name):
    """
    Return the skopt search space for the chosen model.
    - svm: C, gamma
    - xgb: learning_rate, n_estimators, max_depth
    - lgbm: learning_rate, n_estimators, num_leaves
    """
    logger.debug(f"Getting search space for model: {model_name}")
    if model_name == "svm":
        return [
            Real(1e-3, 1.0, name='C', prior='log-uniform'),
            Real(1e-4, 1e-1, name='gamma', prior='log-uniform')
        ]
    elif model_name == "xgb":
        return [
            Real(0.01, 0.3, name='learning_rate', prior='log-uniform'),
            Integer(50, 300, name='n_estimators'),
            Integer(2, 10, name='max_depth')
        ]
    elif model_name == "lgbm":
        return [
            Real(0.01, 0.3, name='learning_rate', prior='log-uniform'),
            Integer(50, 300, name='n_estimators'),
            Integer(10, 100, name='num_leaves')
        ]
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def wait_for_results(iteration, result_files_needed):
    """
    Waits until all required result files for the current iteration are available.
    """
    logger.info("Waiting for workers to finish...")
    while True:
        existing_files = {x.name for x in SHARED_DIR.glob("results_*.json")}
        missing_files = [f for f in result_files_needed if f not in existing_files]
        if not missing_files:
            break
        logger.debug(f"Still waiting for result files: {missing_files}")
        time.sleep(30)  # Check periodically

def convert_to_standard_types(obj):
    """
    Recursively converts all numpy types in a dictionary or list to standard Python types.
    """
    if isinstance(obj, dict):
        return {k: convert_to_standard_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_standard_types(i) for i in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def main():
    logger.info(f"Starting with MODEL_CHOICE={MODEL_CHOICE}, SEED={SEED}")
    dimensions = get_search_space(MODEL_CHOICE)

    # Create the Bayesian optimizer
    optimizer = Optimizer(
        dimensions=dimensions,
        base_estimator="GP",
        acq_func="EI",
        random_state=SEED
    )

    # Clean up leftover JSON files from previous runs
    for file in SHARED_DIR.glob("*_*.json"):
        try:
            file.unlink()
            logger.debug(f"Deleted leftover file: {file}")
        except Exception as e:
            logger.warning(f"Failed to delete {file}: {e}")

    observations_X = []
    observations_y = []

    for iteration in range(NUM_ITERATIONS):
        logger.info(f"=== Iteration {iteration+1}/{NUM_ITERATIONS} ===")

        # Ask the optimizer for NUM_HYPERPARAM_SETS proposals
        proposals = optimizer.ask(n_points=NUM_HYPERPARAM_SETS)
        logger.debug(f"Proposals: {proposals}")

        # Write hyperparameters for workers
        for i, hp_list in enumerate(proposals):
            if MODEL_CHOICE == "svm":
                params_dict = {"C": hp_list[0], "gamma": hp_list[1]}
            elif MODEL_CHOICE == "xgb":
                params_dict = {
                    "learning_rate": hp_list[0],
                    "n_estimators": hp_list[1],
                    "max_depth": hp_list[2]
                }
            elif MODEL_CHOICE == "lgbm":
                params_dict = {
                    "learning_rate": hp_list[0],
                    "n_estimators": hp_list[1],
                    "num_leaves": hp_list[2]
                }
            else:
                raise ValueError("Invalid MODEL_CHOICE")

            # Convert parameters to standard types
            params_dict = convert_to_standard_types(params_dict)

            fname = SHARED_DIR / f"hyperparams_{iteration}_{i}.json"
            try:
                with open(fname, "w") as fp:
                    json.dump(params_dict, fp)
                logger.info(f"Wrote {fname.name} => {params_dict}")
            except Exception as e:
                logger.error(f"Failed to write {fname}: {e}")
                continue

        # Wait for the results from each of these proposals
        result_files_needed = [f"results_{iteration}_{i}.json" for i in range(NUM_HYPERPARAM_SETS)]
        wait_for_results(iteration, result_files_needed)

        # Read and process results
        new_losses = []
        for i in range(NUM_HYPERPARAM_SETS):
            result_file = SHARED_DIR / f"results_{iteration}_{i}.json"
            try:
                with open(result_file, "r") as rf:
                    data = json.load(rf)
                loss_val = data["loss"]
                new_losses.append(loss_val)

                observations_X.append(proposals[i])
                observations_y.append(loss_val)

                logger.info(f"{result_file.name}: loss={loss_val:.4f}")
            except Exception as e:
                logger.error(f"Failed to read {result_file}: {e}")

        # Update optimizer with results
        optimizer.tell(proposals, new_losses)
        logger.info(f"Iteration {iteration}, average loss={np.mean(new_losses):.4f}")

    # Summarize best result
    best_idx = int(np.argmin(observations_y))
    best_params = observations_X[best_idx]
    best_loss = observations_y[best_idx]
    logger.info("\n=== Optimization Complete ===")

    if MODEL_CHOICE == "svm":
        logger.info(f"SVM best => C={best_params[0]:.5f}, gamma={best_params[1]:.5f}, loss={best_loss:.4f}")
    elif MODEL_CHOICE == "xgb":
        logger.info(f"XGB best => learning_rate={best_params[0]:.5f}, n_estimators={best_params[1]}, "
                    f"max_depth={best_params[2]}, loss={best_loss:.4f}")
    elif MODEL_CHOICE == "lgbm":
        logger.info(f"LGBM best => learning_rate={best_params[0]:.5f}, n_estimators={best_params[1]}, "
                    f"num_leaves={best_params[2]}, loss={best_loss:.4f}")

    logger.info(f"(Accuracy ~ {1 - best_loss:.4f})")

if __name__ == "__main__":
    main()
