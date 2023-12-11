# CS8803-MDS-Data-Preprocessing-Transferability


## Model Transfer:

We use Auto-sklearn to test transferring cleaned data from one model to another. Auto-sklearn is an automated machine learning toolkit and a drop-in replacement for a scikit-learn estimator. It frees a machine learning user from algorithm selection and hyperparameter tuning by leveraging recent advantages in Bayesian optimization, meta-learning and ensemble construction.

We created a pluggable interface by updating Auto-Sklearn to either 1) clean and train a dataset for some specified end model, the data cleaning pipeline therein includes missing value imputation, removing low-variance features, and normalization for numerical features, along with category shift, missing value imputation, minority coalescence and one-hot encoding for categorical features, or 2) directly training a dataset for a target model without performing any cleaning.

Example for training a Random Forest Classifier without any pre-processing:

```bat
automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=120,
    per_run_time_limit=30,
    include = {
        'classifier': ["random_forest"],
        'feature_preprocessor': ["no_preprocessing"],
    },
    tmp_folder="tmp/autosklearn_classification_example_tmp4",
)
```

We have included code for our tests in the auto-sklearn directory. It includes Jupyter notebooks and scripts for running various classifiers with or without cleaning enabled. 
