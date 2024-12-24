# Multi-Model Bayesian Optimization with Docker Compose

This project demonstrates how to run **Bayesian hyperparameter optimization** for **three different models**—**SVM**, **XGBoost**, or **LightGBM**—using **scikit-optimize** in a **Docker Compose** setup. The pipeline uses a **manager** service to coordinate the optimization process and multiple **worker** services to train models in parallel.

## Table of Contents
1. [Directory Layout](#directory-layout)
2. [Requirements](#requirements)
3. [Setup](#setup)
4. [Usage](#usage)
5. [How It Works](#how-it-works)
6. [Customization](#customization)

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
│   └─ train.py
└─ shared_volume/
```

- **build/**:
  - **docker-compose.yml**: Defines the manager and worker services, environment variables, and volumes.
  - **manager/**: Dockerfile + `requirements.txt` for the manager service.
  - **worker/**: Dockerfile + `requirements.txt` for the worker service.
- **src/**:
  - **manager.py**: Orchestrates Bayesian optimization (asks for hyperparameters, collects results, updates the optimization process).
  - **train.py**: Worker script that trains one hyperparameter set and writes results.
- **shared_volume/**: A directory **on the host** (outside the containers) that is mounted as `/app/shared_volume` in both manager and worker containers. We use this folder to exchange `.json` files with hyperparams and results.

---

## Usage

### 1. Choose a Model

By default, the project uses `MODEL_CHOICE=svm` in `docker-compose.yml`. You can switch to XGBoost or LightGBM by editing the environment variables in **`build/docker-compose.yml`**:

```yaml
environment:
  - NUM_ITERATIONS=3     # total Bayesian optimization iterations
  - BATCH_SIZE=2         # number of hyperparam sets per iteration
  - MODEL_CHOICE=svm     # or "xgb" or "lgbm"
```

Make sure the manager’s and worker’s `MODEL_CHOICE` environment variables match.

### 2. Build and Run

In the **project root** (which contains `build/` and `src/`), do:

```bash
cd build
docker-compose up --build --scale worker=4
```
where **`--scale worker=4`** will launch 4 worker containers in parallel.

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
docker-compose down
```

---

## How It Works

1. **Manager**:
   - Maintains a **`skopt.Optimizer`** for the chosen model’s hyperparameter space.
   - Each iteration:
     - **ask(...)** to propose **BATCH_SIZE** hyperparam sets.  
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
   - Currently uses the **Breast Cancer** dataset via `sklearn.datasets.load_breast_cancer()`.  
   - Switch to **Iris**, **California Housing**, or any custom dataset by editing `train.py` accordingly.

3. **Parallelism**:  
   - Increase **`BATCH_SIZE`** or **`--scale worker=<N>`** for more parallel workers.  

4. **Optimization Strategy**:  
   - Currently uses Gaussian Process with Expected Improvement (`base_estimator="GP", acq_func="EI"`).  
   - Try `"RF"` (random forest) or `"ET"` (extra trees) as the `base_estimator` in the `Optimizer` if GP becomes too slow or if you want a different strategy.