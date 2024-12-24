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
import numpy as np
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Possible models
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

SHARED_DIR = Path("/app/shared_volume")
MODEL_CHOICE = os.environ.get("MODEL_CHOICE", "svm").lower()
SEED = int(os.environ.get("SEED", "42"))

def find_unclaimed_hyperparams():
    """
    Return the first hyperparams_*.json that doesn't have a matching results_*.json,
    or None if none found.
    """
    files = sorted(SHARED_DIR.glob("hyperparams_*.json"))
    for hp_file in files:
        result_file = SHARED_DIR / hp_file.name.replace("hyperparams", "results")
        if not result_file.exists():
            return hp_file
    return None

def train_model(params, X_train, y_train, X_test, y_test):
    """
    Trains the chosen model with the given params. Returns test accuracy.
    """
    if MODEL_CHOICE == "svm":
        model = SVC(
            C=params["C"],
            gamma=params["gamma"],
            random_state=SEED
        )
    elif MODEL_CHOICE == "xgb":
        model = XGBClassifier(
            learning_rate=params["learning_rate"],
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=SEED
        )
    elif MODEL_CHOICE == "lgbm":
        model = LGBMClassifier(
            learning_rate=params["learning_rate"],
            n_estimators=params["n_estimators"],
            num_leaves=params["num_leaves"],
            random_state=SEED
        )
    else:
        raise ValueError(f"Unsupported MODEL_CHOICE={MODEL_CHOICE}")

    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return accuracy

def main():
    while True:
        # Look for a hyperparams file without a corresponding results file
        hp_file = find_unclaimed_hyperparams()
        if not hp_file:
            # No available work => wait and re-check
            time.sleep(3)
            continue

        print(f"[Worker] Found hyperparams => {hp_file.name}")

        # Derive iteration/index from file name: hyperparams_{iter}_{i}.json
        iteration_str, index_str = hp_file.stem.split("_")[1:]
        iteration = int(iteration_str)
        index = int(index_str)

        with open(hp_file, "r") as f:
            params = json.load(f)

        # Load dataset
        data = load_breast_cancer()
        X, y = data.data, data.target

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=SEED
        )

        # Train model
        accuracy = train_model(params, X_train, y_train, X_test, y_test)
        loss = 1.0 - accuracy  # We want to minimize loss

        # Write results
        result_file = SHARED_DIR / f"results_{iteration}_{index}.json"
        out_data = {**params, "loss": loss}
        with open(result_file, "w") as rf:
            json.dump(out_data, rf)

        print(f"[Worker] => Model={MODEL_CHOICE}, Accuracy={accuracy:.4f}, Loss={loss:.4f}")
        print(f"[Worker] Wrote {result_file.name}\n")

        # By design, one worker handles exactly one hyperparam set, then exits.
        break

if __name__ == "__main__":
    main()
