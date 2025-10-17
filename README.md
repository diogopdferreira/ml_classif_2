# Training Pipeline Overview

This project provides a simple experiment runner that trains multiple classification models on a
single dataset and selects the model with the highest cross-validated F1 score.

## Data Inputs
The training script expects two files in the project root:

- `Xtrain2.pkl`: Pickled feature matrix. Pandas DataFrames are automatically converted to NumPy
  arrays when loaded.
- `Ytrain2.npy`: NumPy array containing the target labels.

The loader automatically flattens the target array so row- or column-vector encodings both work
without triggering a sample-count mismatch during model evaluation.

## Snippets at a Glance

### 1. Load the data
```python
X, y = load_data()
```
**What it does:** Uses `load_data` to deserialize the feature matrix and targets, handling either
NumPy arrays or pandas DataFrames and ensuring the samples align.

**Why it matters:** This step standardises the raw inputs so that every downstream model receives a
clean numeric matrix with a matching target vector.

### 2. Build and score candidate models
```python
best_name, best_model, best_params, best_score = evaluate_models(X, y)
```
**What it does:** Iterates through logistic regression, SVM, random forest, and gradient boosting
pipelines, evaluating each hyperparameter combination with stratified 5-fold macro F1 cross
validation.

**Why it matters:** The exhaustive search across algorithms and hyperparameters ensures the pipeline
selects the classifier that generalises best on the provided dataset.

### 3. Fit the best estimator and persist it
```python
best_model.set_params(**best_params)
best_model.fit(X, y)
joblib.dump(best_model, "best_model.joblib")
```
**What it does:** Refits the winning model on the full dataset using the optimal parameters and
stores the trained artefact on disk.

**Why it matters:** Persisting the trained classifier allows it to be reused for inference without
re-running the full training pipeline.

### 4. Run the training script
```bash
python train.py
```
**What it does:** Executes the `main` entry point which chains together loading, evaluation, and
model persistence, printing the chosen model, hyperparameters, and cross-validated score.

**Why it matters:** Running this command is all that is needed to reproduce the experiment and obtain
an updated best model file after tweaking the configuration or the data.

## Extending the Pipeline
To adjust the experiment, edit `train.py` to modify candidate models or their hyperparameter sets.
Because the script relies on standard scikit-learn APIs, additional classifiers can be inserted by
adding them to the `candidates` list in `evaluate_models`.
