import pandas as pd
from sklearn.model_selection import train_test_split
from autosklearn import classification
from pprint import pprint
from sklearn import metrics
clean_data = '/home/preethi/projects/CS8803-MDS-Data-Preprocessing-Transferability/autosklearn-pipeline/tr_file_labels.csv'
file_path_airbnb = '/home/preethi/projects/CS8803-MDS-Data-Preprocessing-Transferability/data/airbnb.csv'
df = pd.read_csv(file_path_airbnb)
df = df.sample(n = 1000, random_state=16)
# print(df.info)
# df_y = pd.read_csv(file_path_airbnb)
# df = df.dropna(subset=['Rating'])
y = df['Rating']

X = df
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

############################################################################
# Build and fit a classifier
# ==========================

# from autosklearn.experimental.askl2 import AutoSklearn2Classifier

automl = classification.AutoSklearnClassifier(
    time_left_for_this_task=120,
    per_run_time_limit=30,
    include = {
        'classifier': ["random_forest"],
        'feature_preprocessor': ["no_preprocessing"],
        # 'data_preprocessor': ["NoPreprocessing"]
    },
    tmp_folder="tmp/autosklearn_classification_example_tmp8",
)

automl.fit(X_train, y_train, dataset_name="airbnb")
## get configuration for a model/run
run_key = list(automl.automl_.runhistory_.data.keys())[0]
run_value = automl.automl_.runhistory_.data[run_key]
config=automl.automl_.runhistory_.ids_config[run_key.config_id]
print(config)



############################################################################
# View the models found by auto-sklearn
# =====================================

print(automl.leaderboard())

############################################################################
# Print the final ensemble constructed by auto-sklearn
# ====================================================

# pprint(automl.show_models(), indent=4)

###########################################################################
# Get the Score of the final ensemble
# ===================================

# predictions = automl.predict(X_test)
# print("Accuracy score:", metrics.accuracy_score(y_test, predictions))
