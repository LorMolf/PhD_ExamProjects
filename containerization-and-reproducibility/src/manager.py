"""
manager.py

Coordinates Bayesian hyperparameter optimization for SVM, XGB, or LGBM.
Uses skopt to propose hyperparameters, waits for workers to process results,
and updates the optimizer. Outputs final results.
"""

import os
import time
import json
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
from skopt import Optimizer
from skopt.space import Real, Integer

# Environment variables
NUM_ITERATIONS = int(os.environ.get("NUM_ITERATIONS", "3"))
NUM_HYPERPARAM_SETS = int(os.environ.get("NUM_HYPERPARAM_SETS", "5"))
MODEL_CHOICE = os.environ.get("MODEL_CHOICE", "svm").lower()
SEED = int(os.environ.get("SEED", "42"))
SHARED_DIR = Path("/app/shared_volume")
EXPERIMENT_DIR = SHARED_DIR / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
FINAL_RESULTS_FILE = EXPERIMENT_DIR / "final_results.json"

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_search_space():
    """
    Returns the hyperparameter search space for the specified model.
    """
    if MODEL_CHOICE == "svm":
        return [Real(1e-3, 1.0, name='C', prior='log-uniform'), Real(1e-4, 1e-1, name='gamma', prior='log-uniform')]
    elif MODEL_CHOICE == "xgb":
        return [
            Real(0.01, 0.3, name='learning_rate', prior='log-uniform'),
            Integer(50, 300, name='n_estimators'),
            Integer(2, 10, name='max_depth')
        ]
    elif MODEL_CHOICE == "lgbm":
        return [
            Real(0.01, 0.3, name='learning_rate', prior='log-uniform'),
            Integer(50, 300, name='n_estimators'),
            Integer(10, 100, name='num_leaves')
        ]
    else:
        raise ValueError(f"Unsupported MODEL_CHOICE: {MODEL_CHOICE}")

def convert_to_standard_types(obj):
    """
    Recursively converts all NumPy types in a dictionary or list to standard Python types.
    """
    if isinstance(obj, dict):
        return {k: convert_to_standard_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_standard_types(i) for i in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()  # Convert to native Python types
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert NumPy arrays to lists
    else:
        return obj

def write_hyperparams_atomic(filepath, data):
    """
    Writes hyperparameters JSON file atomically to prevent read/write conflicts.
    """
    try:
        standard_data = convert_to_standard_types(data)
        temp_file = filepath.with_suffix('.tmp')
        with temp_file.open('w') as f:
            json.dump(standard_data, f, indent=4)
        temp_file.rename(filepath)
        logger.info(f"Atomically wrote hyperparameters file: {filepath}")
    except Exception as e:
        logger.error(f"Error writing hyperparameters file {filepath}: {e}")

def main():
    logger.info("Starting manager.py")
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Experiment directory created: {EXPERIMENT_DIR}")

    optimizer = Optimizer(get_search_space(), random_state=SEED)
    best_metric = float("inf")
    best_model_path = None

    for iteration in range(NUM_ITERATIONS):
        logger.info(f"Starting iteration {iteration + 1}/{NUM_ITERATIONS}")
        iteration_dir = EXPERIMENT_DIR / f"iteration_{iteration}"
        iteration_dir.mkdir(parents=True, exist_ok=True)

        proposals = optimizer.ask(n_points=NUM_HYPERPARAM_SETS)
        logger.debug(f"Hyperparameter proposals for iteration {iteration}: {proposals}")

        for i, params in enumerate(proposals):
            params_file = iteration_dir / f"hyperparams_{i}.json"
            params_dict = {"C": params[0], "gamma": params[1]} if MODEL_CHOICE == "svm" else params
            logger.debug(f"Writing hyperparameters file: {params_file} => {params_dict}")
            write_hyperparams_atomic(params_file, params_dict)

        for i in range(NUM_HYPERPARAM_SETS):
            result_file = iteration_dir / f"results_{i}.json"
            while not result_file.exists():
                logger.debug(f"Waiting for result file: {result_file}")
                time.sleep(10)

        # Process results
        losses = []
        for i in range(NUM_HYPERPARAM_SETS):
            result_file = iteration_dir / f"results_{i}.json"
            try:
                with open(result_file, "r") as f:
                    result = json.load(f)
                losses.append(result["loss"])
                if result["loss"] < best_metric:
                    best_metric = result["loss"]
                    best_model_path = result.get("model_path", None)
                logger.info(f"Processed {result_file}: Loss={result['loss']}")
            except Exception as e:
                logger.error(f"Error processing result file {result_file}: {e}")

        optimizer.tell(proposals, losses)

    # Save final results
    final_results = {"best_metric": best_metric, "best_model_path": best_model_path}
    try:
        with FINAL_RESULTS_FILE.open("w") as f:
            json.dump(final_results, f, indent=4)
        logger.info(f"Final results saved to {FINAL_RESULTS_FILE}")
    except Exception as e:
        logger.error(f"Failed to write final results file: {e}")

    logger.info("Manager completed successfully.")

if __name__ == "__main__":
    main()
