# Multi-Model Bayesian Optimization Orchestration with Docker Compose
> **@author** Lorenzo Molfetta

This project demonstrates how to run **Bayesian hyperparameter optimization** for **three different models**—**SVM**, **XGBoost**, or **LightGBM**—using **scikit-optimize** in a **Docker Compose** setup. The pipeline uses a **manager** service to coordinate the optimization process and multiple **worker** services to train models in parallel.

> ‼️ In `build/logs.log` you can find the output of a distributed training run.

## Table of Contents
1. [Directory Layout](#directory-layout)
2. [Usage](#usage)
3. [How It Works](#how-it-works)
4. [Customization](#customization)

---

## Directory Layout

```
multi-model-bayesian-opt/
├─ build/
│   ├─ docker-compose.yml
│   ├─ manager/
│   │   ├─ Dockerfile
│   │   └─ requirements.txt
│   └─ worker/
│       ├─ Dockerfile
│       └─ requirements.txt
├─ src/
│   ├─ manager.py
│   └─train.py
└─ shared_volume/
```

- **build/**: Contains Docker and configuration files
  - **docker-compose.yml**: Defines services, env vars, and volumes
  - **manager/**: Manager service configuration
  - **worker/**: Worker service configuration
- **src/**: Python source code
  - **manager.py**: Orchestrates optimization process
  - **train.py**: Handles model training and evaluation
- **shared_volume/**: Shared storage for inter-container communication

---

## Usage

### 1. Choose a Model

By default, the project uses `MODEL_CHOICE=svm` in `docker-compose.yml`. You can switch to XGBoost or LightGBM by editing the environment variables in **`build/variables.env`**:

```env
SEED=42                       # reproducibility random seed
NUM_ITERATIONS=3              # total Bayesian optimization iterations
NUM_HYPERPARAM_SETS=2         # number of hyperparam sets per iteration
MODEL_CHOICE=svm              # "svm" or "xgb" or "lgbm"
```

### 2. Build and Run

From the project root:

```bash
cd build
docker-compose up --build --scale worker=4
```

The `--scale worker=4` flag launches 4 worker containers for parallel processing. Adjust this number based on your system's capabilities and requirements.

### 3. Check Logs

- The **manager** container logs will show each iteration of Bayesian optimization:
  1. Generating hyperparameter sets (the `ask` step).  
  2. Waiting for results from all workers.  
  3. Reading results, computing average loss, and telling (`.tell()`) the optimizer.  
- The **worker** container logs will show each worker picking up one hyperparameter set, training the model, computing accuracy, and writing results back.

After **NUM_ITERATIONS** are complete, the manager logs will show the **best hyperparameters** found so far (with their final loss and accuracy).

### 4. Stop the Containers

When finished, press **Ctrl+C** in your terminal or run:

```bash
docker-compose down --rmi all --volumes --remove-orphans
```

---

## How It Works

1. **Manager**:
   - Maintains a **`skopt.Optimizer`** for the chosen model’s hyperparameter space.
   - Each iteration:
     - **ask(...)** to propose **NUM_HYPERPARAM_SETS** hyperparam sets.  
     - Saves them to JSON files in the shared volume, e.g. `hyperparams_0_0.json`, `hyperparams_0_1.json`, etc.  
     - Waits until corresponding worker result files (`results_0_0.json`, etc.) appear.  
     - Reads results, **tell(...)** the optimizer to refine future proposals.

2. **Workers**:
   - Each worker looks in **`/app/shared_volume`** (the mounted folder) for an unclaimed `hyperparams_x_y.json`.
   - Trains the selected model (SVM, XGBoost, LightGBM) on a standard scikit-learn dataset (Breast Cancer) using those hyperparameters.
   - Evaluates test-set **accuracy**, transforms it to **loss = 1 - accuracy**, and saves to `results_x_y.json`.
   - Exits after handling one set of hyperparams, so multiple workers can run in parallel.

3. **Shared Volume**:
   - A local folder **`shared_volume/`** is mounted into both containers at the path **`/app/shared_volume`**.
   - The manager writes hyperparam files, workers read them and write results, and the manager reads those results.

---

## Customization

1. **Models**:  
   - Currently supports **SVM** (`sklearn.svm.SVC`), **XGBoost** (`XGBClassifier`), **LightGBM** (`LGBMClassifier`).  
   - Update `manager.py` > `get_search_space()` and `train.py` > `train_model()` for more hyperparameters (e.g., `subsample`, `colsample_bytree` for XGB, or `reg_alpha`, `reg_lambda` for LGBM).

2. **Dataset**:  
   - **Breast Cancer** classification

3. **Parallelism**:  
   - Scale workers with `--scale worker=N`
   - Automatic work distribution

4. **Optimization Strategy**:  
   - Default: Gaussian Process with EI
   - Support for RF and ET base estimators

5. **Monitoring**:
   - Real-time progress tracking
   - Detailed worker logs
   - Result persistence in `shared_volume`
