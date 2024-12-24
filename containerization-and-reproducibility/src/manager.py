"""
manager.py

Coordinates Bayesian hyperparameter optimization for either SVM, XGB, or LGBM.
Uses skopt to propose hyperparams, writes them to /app/shared_volume, waits for
workers to produce results, then updates the optimizer.
"""

import os
import time
import json
from pathlib import Path

import numpy as np
from skopt import Optimizer
from skopt.space import Real, Integer

# Environment variables
NUM_ITERATIONS = int(os.environ.get("NUM_ITERATIONS", "3"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "2"))
MODEL_CHOICE = os.environ.get("MODEL_CHOICE", "svm").lower()
SEED = int(os.environ.get("SEED", "42"))

SHARED_DIR = Path("/app/shared_volume")

def get_search_space(model_name):
    """
    Return the skopt search space for the chosen model.
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
        raise ValueError(f"Unsupported model: {model_name}")

def main():
    print(f"[Manager] Starting with MODEL_CHOICE={MODEL_CHOICE}, SEED={SEED}")
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
        file.unlink()

    # We'll keep track of proposals & their losses
    observations_X = []
    observations_y = []

    for iteration in range(NUM_ITERATIONS):
        print(f"=== [Manager] Iteration {iteration+1}/{NUM_ITERATIONS} ===")

        # Ask the optimizer for BATCH_SIZE hyperparams
        proposals = optimizer.ask(n_points=BATCH_SIZE)

        # Write them out so workers can pick them up
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

            fname = SHARED_DIR / f"hyperparams_{iteration}_{i}.json"
            with open(fname, "w") as fp:
                json.dump(params_dict, fp)
            print(f"[Manager] Wrote {fname.name} => {params_dict}")

        # Wait for the results from each of these proposals
        result_files_needed = [f"results_{iteration}_{i}.json" for i in range(BATCH_SIZE)]
        print("[Manager] Waiting for workers to finish...")
        while True:
            existing = {x.name for x in SHARED_DIR.glob("results_*.json")}
            if all(rf in existing for rf in result_files_needed):
                break
            time.sleep(3)
        print("[Manager] All results found for iteration", iteration)

        # Read each result, update the optimizer
        new_losses = []
        for i in range(BATCH_SIZE):
            result_file = SHARED_DIR / f"results_{iteration}_{i}.json"
            with open(result_file, "r") as rf:
                data = json.load(rf)
            loss_val = data["loss"]
            new_losses.append(loss_val)

            observations_X.append(proposals[i])
            observations_y.append(loss_val)

            print(f"[Manager] {result_file.name}: loss={loss_val:.4f}")

        optimizer.tell(proposals, new_losses)
        print(f"[Manager] Iteration {iteration}, average loss={np.mean(new_losses):.4f}")

    # Summarize best result
    best_idx = int(np.argmin(observations_y))
    best_params = observations_X[best_idx]
    best_loss = observations_y[best_idx]
    print("\n=== [Manager] Optimization Complete ===")

    if MODEL_CHOICE == "svm":
        print(f"SVM best => C={best_params[0]:.5f}, gamma={best_params[1]:.5f}, loss={best_loss:.4f}")
    elif MODEL_CHOICE == "xgb":
        print(f"XGB best => learning_rate={best_params[0]:.5f}, n_estimators={best_params[1]}, "
              f"max_depth={best_params[2]}, loss={best_loss:.4f}")
    elif MODEL_CHOICE == "lgbm":
        print(f"LGBM best => learning_rate={best_params[0]:.5f}, n_estimators={best_params[1]}, "
              f"num_leaves={best_params[2]}, loss={best_loss:.4f}")

    print(f"(Accuracy ~ {1 - best_loss:.4f})")

if __name__ == "__main__":
    main()
