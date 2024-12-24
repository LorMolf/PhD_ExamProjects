"""
train.py

Worker script for training one model (SVM, XGBoost, or LightGBM) 
based on MODEL_CHOICE environment variable. 
Finds unclaimed hyperparam JSON => trains => writes results JSON with loss=1-accuracy.
"""

import os
import time
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# For SVM
from sklearn.svm import SVC
# For XGB
from xgboost import XGBClassifier
# For LGBM
from lightgbm import LGBMClassifier

SHARED_DIR = Path("/app/shared_volume")
MODEL_CHOICE = os.environ.get("MODEL_CHOICE", "svm").lower()
SEED = int(os.getenv("SEED", 42))

def find_unclaimed_hyperparams():
    files = sorted(SHARED_DIR.glob("hyperparams_*.json"))
    for hp_file in files:
        # matching results => replace "hyperparams" with "results"
        result_file = SHARED_DIR / hp_file.name.replace("hyperparams", "results")
        if not result_file.exists():
            return hp_file
    return None

def train_model(params, X_train, y_train, X_test, y_test):
    if MODEL_CHOICE == "svm":
        model = SVC(C=params["C"], gamma=params["gamma"])
    elif MODEL_CHOICE == "xgb":
        model = XGBClassifier(
            learning_rate=params["learning_rate"],
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            use_label_encoder=False,
            eval_metric="logloss"
        )
    elif MODEL_CHOICE == "lgbm":
        model = LGBMClassifier(
            learning_rate=params["learning_rate"],
            n_estimators=params["n_estimators"],
            num_leaves=params["num_leaves"]
        )
    else:
        raise ValueError(f"Unsupported MODEL_CHOICE={MODEL_CHOICE}")

    model.fit(X_train, y_train)
    return model.score(X_test, y_test)

def main():
    while True:
        hp_file = find_unclaimed_hyperparams()
        if not hp_file:
            time.sleep(3)
            continue

        print(f"[Worker] Found hyperparams => {hp_file.name}")
        # Extract iteration/index from file name
        iteration_str, index_str = hp_file.stem.split("_")[1:]
        iteration = int(iteration_str)
        index = int(index_str)

        with open(hp_file, "r") as f:
            params = json.load(f)

        # Load dataset
        data = load_breast_cancer()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                            test_size=0.2, 
                                                            random_state=SEED)
        # Train + evaluate
        acc = train_model(params, X_train, y_train, X_test, y_test)
        loss = 1.0 - acc

        # Write results
        result_file = SHARED_DIR / f"results_{iteration}_{index}.json"
        with open(result_file, "w") as rf:
            out_data = {**params, "loss": loss}
            json.dump(out_data, rf)

        print(f"[Worker] => Model={MODEL_CHOICE}, accuracy={acc:.4f}, loss={loss:.4f}")
        print(f"[Worker] Wrote {result_file}\n")

        # Exit after one set (so multiple containers handle multiple sets)
        break

if __name__ == "__main__":
    main()
