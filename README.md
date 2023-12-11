# CS8803-MDS-Data-Preprocessing-Transferability


## Model Transfer:
---
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

We have included code for our tests in the auto-sklearn/experiments directory. It includes Jupyter notebooks and scripts for running various classifiers with and without data preprocessing enabled. 


## Task Transfer:
---

Installation of Diffprep is the same as the original [source code](https://github.com/chu-data-lab/DiffPrep). In addition to this, for training and analysis, Jupyter Notebook needs to be installed.

All resources for task transfer are under the Diffprep directory. All plots generated are stored in `Diffprep/plots`.

For task transfer, Diffprep code needs to be modified to freeze a pipeline when its transferred from another task. To run in transfer mode, the constructor of FirstTransformer in `Diffprep/pipeline/diffprep_fix_pipeline.py` needs to have the line `self.is_sampled = False` added at the end. To ensure the pipeline isn't trained, the train method of the `DiffPrepSGD` class needs to skip the pipeline update, thus the line `self.update_prep_pipeline line` can be removed.

With these two changes, Diffprep runs in transfer mode, undoing these changes make it run in regular training mode.

The datasets are present in `Diffprep/data` directory, and the notebooks for each dataset are present in their transfer directory. For example, Airbnb notebooks are in `Diffprep/airbnb_transfer`.

For each transfer directory, `training.ipynb` trains a Diffprep pipeline on a source task and saves it. `pipeline_transfer_<dataset>.ipynb` generates tasks similar to the source task and transfers the source-cleaned pipeline to train end models for the target task. By modifying the code to training mode, we use it to also train separate pipelines on these synthetic tasks. All the pipelines are saved in `Diffprep/result/Diffprep_fix/<dataset>`.

`pipeline_synth_analysis_<dataset>.ipynb` loads all the pipelines trained separately for the synthetic tasks, computes their entropies with the source pipeline, and also analyses test accuracies with and without transfer.

For optimal-transport dataset distance, we had difficulty setting it up locally due to dependency conflicts and OS support issues, so we run it on Google Colab. The file is under `Diffprep/otdd/otdd_final.ipynb`. Once the OTDD code has been cloned on Colab, one small change in the source code needs to be done to make it work with arbitrary datasets (their example used Torch loader datasets). This change is documented in the notebook. To compute the dataset distances for House Prices for example, copy all the CVS files (cleaned datasets) at all similarity levels from the `Diffprep/house_prices_transfer` directory over to the `sample_data` directory in Colab, and run all cells to produce correlations and plots. 