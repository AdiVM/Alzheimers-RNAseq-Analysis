import argparse
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_predict
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score, recall_score,
    precision_score, f1_score, matthews_corrcoef
)
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt
from flaml import AutoML
import warnings
import logging
from datetime import datetime
from sklearn.linear_model import LogisticRegression

log_dir_path = "/n/groups/patel/adithya/Log_Dir_withDemographics/"
# Path for logging
LOG_FILE_PATH = os.path.expanduser(f'{log_dir_path}experiment_log.txt')

def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(
        description='Run AutoML on chosen feature sets')

    # Add arguments
    parser.add_argument('--cell_type', type=str,
                        help='options: In, Mic, Oli, Opc, Per, Ast, End, Ex')
     # Parse the arguments
    args = parser.parse_args()
    cell_type = args.cell_type
    if cell_type not in ['In', 'Mic', 'Oli', 'Opc', 'Per', 'Ast', 'End', 'Ex']:
        print("Invalid input")
        exit()
    #End if statement.

    log_message = f"Processing cell type: {cell_type}"
    
    # Print the message to the console
    print(log_message)
    
    # Append the message to the log file
    with open(LOG_FILE_PATH, 'a') as log_file:
        log_file.write(log_message + '\n')

    log_message = "Initializing AutoML"
    print(log_message)
    
    # Initialize AutoML
    my1classifier = AutoML() 
    log_message = "AutoML initialization passed"
    print(log_message)
    
    # Set up the CSV file and the writer
    output_csv = f'{log_dir_path}{cell_type}_V1_output_log.csv'

    #Ensure the file exists and create it if it doesn't
    if not os.path.exists(output_csv):
        with open(output_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Message", "Additional Info"])  # Write header

    def log_to_csv(message, info=""):
        """Logs a message and additional info to the CSV file."""
        with open(output_csv, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([message, info])

    # Load data
    # Bigger dataset is our training set
    train_metadata = pd.read_parquet('/home/adm808/CellMetadataSyn18485175.parquet')
    train_matrix = pd.read_parquet('/home/adm808/NormalizedCellMatrixSyn18485175.parquet')
    test_metadata = pd.read_csv('/home/adm808/UpdatedCellMetadataSyn16780177.csv', low_memory=False)
    test_matrix = pd.read_parquet('/home/adm808/NormalizedCellMatrixSyn16780177.parquet')

    log_message = "Data loaded successfully"
    print(log_message)
    
    # Determine Alzheimer's or control status
    train_metadata['alzheimers_or_control'] = train_metadata['age_first_ad_dx'].notnull().astype(int)
    test_metadata['alzheimers_or_control'] = test_metadata['age_first_ad_dx'].notnull().astype(int)

    def select_missing_genes(filtered_matrix):
        mean_threshold = 1
        missingness_threshold = 95
        mean_gene_expression = np.mean(filtered_matrix, axis=1)
        missingness = (filtered_matrix == 0).sum(axis=1) / filtered_matrix.shape[1] * 100
        null_expression = (missingness > missingness_threshold) & (mean_gene_expression < mean_threshold)
        genes_to_drop = filtered_matrix.index[null_expression].tolist()
        return genes_to_drop

    def drop_missing_genes(matrix, cell_metadata, cell_type):
        cell_type_specific_metadata = cell_metadata[cell_metadata['broad.cell.type'] == cell_type]
        cell_names = cell_type_specific_metadata['TAG']
        matrix_filtered = matrix[cell_names]
        log_to_csv(f'Number of genes and cells that are: {cell_type}{matrix_filtered.shape}')
        genes_to_drop = select_missing_genes(matrix_filtered)
        df_filtered = matrix_filtered.drop(genes_to_drop, axis=0)
        return df_filtered

    def filter_metadata(cell_metadata, cell_type):
        cell_specific_metadata = cell_metadata[cell_metadata['broad.cell.type'] == cell_type]
        return cell_specific_metadata

    train_matrix_filtered = drop_missing_genes(train_matrix, train_metadata, cell_type)
    test_matrix_filtered = drop_missing_genes(test_matrix, test_metadata, cell_type)

    train_metadata_filtered = filter_metadata(train_metadata, cell_type)
    test_metadata_filtered = filter_metadata(test_metadata, cell_type)

    log_to_csv(f"Training data shape: {train_matrix_filtered.shape}")
    log_to_csv(f"Testing data shape: {test_matrix_filtered.shape}")

    train_unique_samples = train_metadata_filtered['sample'].unique()
    test_unique_samples = test_metadata_filtered['sample'].unique()
    
    log_to_csv(f'Samples in training set: {train_unique_samples}')
    log_to_csv(f'Samples in test set: {test_unique_samples}')
    log_to_csv(f'Number of unique samples in training set: {len(train_unique_samples)}')
    log_to_csv(f'Number of unique samples in test set: {len(test_unique_samples)}')

    train_cell_names = train_metadata_filtered['TAG']
    test_cell_names = test_metadata_filtered['TAG']

    X_train = train_matrix_filtered[train_cell_names]
    X_test = test_matrix_filtered[test_cell_names]

    X_train = X_train.T
    X_test = X_test.T

    y_train = train_metadata_filtered.set_index('TAG').loc[train_cell_names, 'alzheimers_or_control']
    y_test = test_metadata_filtered.set_index('TAG').loc[test_cell_names, 'alzheimers_or_control']

    y_train_series = pd.Series(y_train)
    y_test_series = pd.Series(y_test)

    train_class_balance = y_train_series.value_counts()
    test_class_balance = y_test_series.value_counts()

    log_message = "Data splitting completed -- training and testing sets created"
    print(log_message)
    
    log_to_csv(f'Training set class balance:\n{train_class_balance}')
    log_to_csv(f'Testing set class balance:\n{test_class_balance}')
    log_to_csv(X_train.shape)

    # Dummy classifier
    dummy_clf = DummyClassifier(strategy='most_frequent')
    dummy_clf.fit(X_train, y_train)
    y_prob_train_dummy = dummy_clf.predict_proba(X_train)[:, 1]
    y_prob_test_dummy = dummy_clf.predict_proba(X_test)[:, 1]
    y_pred_train_dummy = dummy_clf.predict(X_train)
    y_pred_test_dummy = dummy_clf.predict(X_test)

    train_accuracy_dummy = accuracy_score(y_train, y_pred_train_dummy)
    train_roc_auc_dummy = roc_auc_score(y_train, y_prob_train_dummy)
    train_avg_precision_dummy = average_precision_score(y_train, y_prob_train_dummy)
    train_recall_dummy = recall_score(y_train, y_pred_train_dummy, pos_label=1)
    train_precision_dummy = precision_score(y_train, y_pred_train_dummy, pos_label=1)
    train_f1_dummy = f1_score(y_train, y_pred_train_dummy)
    train_mcc_dummy = matthews_corrcoef(y_train, y_pred_train_dummy)

    test_accuracy_dummy = accuracy_score(y_test, y_pred_test_dummy)
    test_roc_auc_dummy = roc_auc_score(y_test, y_prob_test_dummy)
    test_avg_precision_dummy = average_precision_score(y_test, y_prob_test_dummy)
    test_recall_dummy = recall_score(y_test, y_pred_test_dummy, pos_label=1)
    test_precision_dummy = precision_score(y_test, y_pred_test_dummy, pos_label=1)
    test_f1_dummy = f1_score(y_test, y_pred_test_dummy)
    test_mcc_dummy = matthews_corrcoef(y_test, y_pred_test_dummy)

    log_to_csv(f'Dummy Train Accuracy: {train_accuracy_dummy}')
    log_to_csv(f'Dummy Train ROC AUC: {train_roc_auc_dummy}')
    log_to_csv(f'Dummy Train Average Precision: {train_avg_precision_dummy}')
    log_to_csv(f'Dummy Train Recall: {train_recall_dummy}')
    log_to_csv(f'Dummy Train Precision: {train_precision_dummy}')
    log_to_csv(f'Dummy Train F1 Score: {train_f1_dummy}')
    log_to_csv(f'Dummy Train MCC: {train_mcc_dummy}')

    log_to_csv(f'Dummy Test Accuracy: {test_accuracy_dummy}')
    log_to_csv(f'Dummy Test ROC AUC: {test_roc_auc_dummy}')
    log_to_csv(f'Dummy Test Average Precision: {test_avg_precision_dummy}')
    log_to_csv(f'Dummy Test Recall: {test_recall_dummy}')
    log_to_csv(f'Dummy Test Precision: {test_precision_dummy}')
    log_to_csv(f'Dummy Test F1 Score: {test_f1_dummy}')
    log_to_csv(f'Dummy Test MCC: {test_mcc_dummy}')

    log_message = "Dummy classifier created"
    print(log_message)
    
    warnings.filterwarnings('ignore')
    logging.getLogger().setLevel(logging.WARNING)

    log_file_all_features = f'{cell_type}allfeatureslog.json'
    groups = train_metadata_filtered['sample']

    log_message = "
