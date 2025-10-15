# Training Pipeline Overview

This project provides a simple experiment runner that trains multiple classification models on a
single dataset and selects the model with the highest cross-validated F1 score.

## Data Inputs
The training script expects two files in the project root:

- `X_train2.pkl`: Pickled feature matrix. Pandas DataFrames are automatically converted to NumPy
  arrays when loaded.
- `Y_train2.npy`: NumPy array containing the target labels.

## How Training Works
Run the training process with:

```bash
python train.py
```

The script performs the following steps:

1. Loads the feature and label arrays from the files listed above.
2. Builds several candidate classifiers with intentionally wide hyperparameter configurations to
   encourage diverse model behavior.
3. Evaluates every hyperparameter combination using stratified 5-fold cross-validation and the
   macro F1 score.
4. Selects the best-performing model, refits it on the full dataset, and saves it to
   `best_model.joblib`.

## Extending the Pipeline
To adjust the experiment, edit `train.py` to modify candidate models or their hyperparameter sets.
Because the script relies on standard scikit-learn APIs, additional classifiers can be inserted by
adding them to the `candidates` list in `evaluate_models`.

