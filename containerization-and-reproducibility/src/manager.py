"""
manager.py

Orchestrates Bayesian hyperparameter optimization for one model type (SVM, XGB, or LGBM),
depending on the MODEL_CHOICE environment variable.
"""

import os
import time
import json
from pathlib import Path

import numpy as np
from skopt import Optimizer
from skopt.space import Real, Integer

# ENV
NUM_ITERATIONS = int(os.environ.get("NUM_ITERATIONS", "3"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "2"))
MODEL_CHOICE = os.environ.get("MODEL_CHOICE", "svm").lower()
SEED = int(os.getenv("SEED", 42))


SHARED_DIR = Path("/app/shared_volume")

def get_search_space(model_name):
    """
    Returns the skopt search space for the chosen model.
      - svm: C, gamma
      - xgb: learning_rate, n_estimators, max_depth
      - lgbm: learning_rate, n_estimators, num_leaves
    """
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
        raise ValueError(f"Unsupported MODEL_CHOICE: {model_name}")

def main():
    print(f"[Manager] MODEL_CHOICE={MODEL_CHOICE}")
    dimensions = get_search_space(MODEL_CHOICE)

    optimizer = Optimizer(dimensions=dimensions, 
                          base_estimator="GP",
                          acq_func="EI", 
                          random_state=SEED)

    # Clean up leftover JSONs
    for file in SHARED_DIR.glob("*_*.json"):
        file.unlink()

    observations_X = []
    observations_y = []

    for iteration in range(NUM_ITERATIONS):
        print(f"=== [Manager] Iteration {iteration + 1}/{NUM_ITERATIONS} ===")

        proposals = optimizer.ask(n_points=BATCH_SIZE)

        # Write hyperparams to disk
        for i, hyperparams in enumerate(proposals):
            if MODEL_CHOICE == "svm":
                param_dict = {"C": hyperparams[0], "gamma": hyperparams[1]}
            elif MODEL_CHOICE == "xgb":
                param_dict = {
                    "learning_rate": hyperparams[0],
                    "n_estimators": hyperparams[1],
                    "max_depth": hyperparams[2]
                }
            elif MODEL_CHOICE == "lgbm":
                param_dict = {
                    "learning_rate": hyperparams[0],
                    "n_estimators": hyperparams[1],
                    "num_leaves": hyperparams[2]
                }
            else:
                raise ValueError("Invalid model choice.")

            fname = SHARED_DIR / f"hyperparams_{iteration}_{i}.json"
            with open(fname, "w") as f:
                json.dump(param_dict, f)
            print(f"[Manager] Wrote {fname} => {param_dict}")

        # Wait for results
        needed_files = [f"results_{iteration}_{i}.json" for i in range(BATCH_SIZE)]
        print("[Manager] Waiting for worker results...")
        while True:
            existing = {x.name for x in SHARED_DIR.glob("results_*.json")}
            if all(nf in existing for nf in needed_files):
                break
            time.sleep(5)
        print("[Manager] All worker results found.")

        # Read & tell optimizer
        new_losses = []
        for i in range(BATCH_SIZE):
            result_file = SHARED_DIR / f"results_{iteration}_{i}.json"
            with open(result_file, "r") as rf:
                data = json.load(rf)
            loss = data["loss"]
            new_losses.append(loss)

            observations_X.append(proposals[i])
            observations_y.append(loss)

            print(f"[Manager] {result_file.name} => loss={loss:.4f}")

        optimizer.tell(proposals, new_losses)
        print(f"[Manager] Iteration {iteration} average loss={np.mean(new_losses):.4f}")

    # Find best
    best_idx = int(np.argmin(observations_y))
    best_params = observations_X[best_idx]
    best_loss = observations_y[best_idx]
    print("=== [Manager] Complete ===")

    if MODEL_CHOICE == "svm":
        print(f"SVM best: C={best_params[0]:.4f}, gamma={best_params[1]:.4f}, loss={best_loss:.4f}")
    elif MODEL_CHOICE == "xgb":
        print(f"XGB best: lr={best_params[0]:.4f}, n_est={best_params[1]}, max_depth={best_params[2]}, loss={best_loss:.4f}")
    elif MODEL_CHOICE == "lgbm":
        print(f"LGBM best: lr={best_params[0]:.4f}, n_est={best_params[1]}, num_leaves={best_params[2]}, loss={best_loss:.4f}")

    print(f"(Corresponding accuracy={1 - best_loss:.4f})")


if __name__ == "__main__":
    main()
