#pip install pandas numpy scikit-learn flaml[automl] matplotlib joblib pyarrow fastparquet
import argparse
import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score, recall_score,
    precision_score, f1_score, matthews_corrcoef
)
from flaml import AutoML
import matplotlib.pyplot as plt
import joblib

log_dir_path = "/n/groups/patel/adithya/Log_Dir_Maximal_Test/"
LOG_FILE_PATH = os.path.expanduser(f'{log_dir_path}experiment_log.txt')

model_path = "/n/groups/patel/adithya/Log_Dir_Maximal/lasso_model_all_features.pkl"
maximal_classifier = joblib.load(model_path)
print("Model loaded successfully from", model_path)

# Load and preprocess gene expression data
train_matrix = pd.read_parquet('/home/adm808/NormalizedCellMatrixSyn18485175.parquet').T
test_matrix = pd.read_parquet('/home/adm808/NormalizedCellMatrixSyn16780177.parquet').T
train_matrix_filtered = train_matrix.drop(select_missing_genes(train_matrix), axis=1)
test_matrix_filtered = test_matrix.drop(select_missing_genes(test_matrix), axis=1)

print("All preprocessing steps done!")

print("Begining to merge metadata!")
train_data = train_matrix_filtered.merge(
    train_metadata[['TAG', 'alzheimers_or_control']],
    left_index=True, right_on='TAG', how='inner'
).set_index('TAG')

test_data = test_matrix_filtered.merge(
    test_metadata[['TAG', 'alzheimers_or_control']],
    left_index=True, right_on='TAG', how='inner'
).set_index('TAG')

# Common genes and column cleaning
common_genes = train_data.columns.intersection(test_data.columns)
X_train = train_data[common_genes].drop(columns=['alzheimers_or_control'])
X_test = test_data[common_genes].drop(columns=['alzheimers_or_control'])

original_columns = common_genes
cleaned_columns = original_columns.str.replace(r'[^A-Za-z0-9_]+', '', regex=True)
column_mapping = dict(zip(cleaned_columns, original_columns))


# Extract and map top features
top_features_cleaned = get_top_features(maximal_classifier, n_top=100)
top_features_original = [column_mapping.get(feature, feature) for feature in top_features_cleaned]

# Incremental evaluation
incremental_results = []
for i in range(1, 101):
    current_features = top_features_cleaned[:i]
    X_train_top_i = X_train[current_features]
    X_test_top_i = X_test[current_features]

    # Retrain using logs
    incremental_classifier = AutoML()
    incremental_classifier.retrain_from_log(
        X_train_top_i, y_train,
        log_file_name=f"{log_dir_path}/experiment_log.txt"
    )

    # Collect metrics and save results
    y_prob_train = incremental_classifier.predict_proba(X_train_top_i)[:, 1]
    y_prob_test = incremental_classifier.predict_proba(X_test_top_i)[:, 1]
    result = {
        'num_features': i,
        'train_roc_auc': roc_auc_score(y_train, y_prob_train),
        'test_roc_auc': roc_auc_score(y_test, y_prob_test),
    }
    incremental_results.append(result)

pd.DataFrame(incremental_results).to_csv(f'{log_dir_path}incremental_top_features_metrics.csv', index=False)
