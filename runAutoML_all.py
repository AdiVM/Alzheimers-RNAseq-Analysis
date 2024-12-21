import argparse
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedShuffleSplit
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

#tnow = datetime.now()
#timestamp = tnow.strftime("%Y%m%d_%H%M%S")
log_dir_path="/n/groups/patel/adithya/Log_Dir_AutoML/"
# Path for logging
LOG_FILE_PATH = os.path.expanduser(f'{log_dir_path}automl_log.txt')

def main(cell_type):
    # Code to load data, run AutoML, and save results
    import argparse
    import os
    import pandas as pd
    import numpy as np
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedShuffleSplit
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
    log_message = f"Processing cell type: {cell_type}"
    
    # Print the message to the console
    print(log_message)

    # Append the message to the log file
    with open(LOG_FILE_PATH, 'a') as log_file:
        log_file.write(log_message + '\n')

    import csv
    import os
    #from datetime import datetime
    #tnow = datetime.now()
    #timestamp = tnow.strftime("%Y%m%d_%H%M%S")
    
    # Set up the CSV file and the writer
    output_csv = f'{log_dir_path}{cell_type}_output_log.csv'

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
    CellMetadataSyn18485175 = pd.read_parquet('/home/adm808/CellMetadataSyn18485175.parquet')
    CellMatrixSyn18485175 = pd.read_parquet('/home/adm808/NormalizedCellMatrixSyn18485175.parquet')

    # Determine Alzheimer's or control status
    CellMetadataSyn18485175['alzheimers_or_control'] = CellMetadataSyn18485175['age_first_ad_dx'].notnull().astype(int)

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

    cellmatrixsyn18485175_filtered = drop_missing_genes(CellMatrixSyn18485175, CellMetadataSyn18485175, cell_type)
    log_to_csv(cellmatrixsyn18485175_filtered.shape)

    # Set cell type of interest
    cell_type_metadata = CellMetadataSyn18485175[CellMetadataSyn18485175['broad.cell.type'] == cell_type]
    unique_samples = cell_type_metadata[['sample', 'alzheimers_or_control']].drop_duplicates()

    # Use StratifiedShuffleSplit to ensure balanced class distribution
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in sss.split(unique_samples['sample'], unique_samples['alzheimers_or_control']):
        sample_train, sample_test = unique_samples['sample'].values[train_index], unique_samples['sample'].values[test_index]

    log_to_csv(f'Samples in training set: {sample_train}')
    log_to_csv(f'Samples in test set: {sample_test}')
    log_to_csv(f'Number of unique samples in training set: {len(sample_train)}')
    log_to_csv(f'Number of unique samples in test set: {len(sample_test)}')

    train_metadata = cell_type_metadata[cell_type_metadata['sample'].isin(sample_train)]
    test_metadata = cell_type_metadata[cell_type_metadata['sample'].isin(sample_test)]

    log_to_csv(train_metadata.shape)
    log_to_csv(test_metadata.shape)

    train_cell_names = train_metadata['TAG']
    test_cell_names = test_metadata['TAG']

    X_train = cellmatrixsyn18485175_filtered[train_cell_names]
    X_test = cellmatrixsyn18485175_filtered[test_cell_names]

    X_train = X_train.T
    X_test = X_test.T


    y_train = train_metadata.set_index('TAG').loc[train_cell_names, 'alzheimers_or_control']
    y_test = test_metadata.set_index('TAG').loc[test_cell_names, 'alzheimers_or_control']


    y_train_series = pd.Series(y_train)
    y_test_series = pd.Series(y_test)

    train_class_balance = y_train_series.value_counts()
    test_class_balance = y_test_series.value_counts()

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

    # Initialize AutoML
    automl = AutoML()
    warnings.filterwarnings('ignore')
    logging.getLogger().setLevel(logging.WARNING)

    #Naming the log file for the nitial autoML process run on all genes for the cell
    log_file_all_features = f'{cell_type}allfeatureslog.json'

    automl.fit(
        X_train,
        y_train,
        task="classification",
        time_budget=1800,
        metric='log_loss',
        n_jobs=-1,
        eval_method='cv',
        n_splits=100,
        split_type='stratified',
        log_training_metric=True,
        early_stop=True,
        seed=239875,
        model_history=True,
        estimator_list=['lgbm'],
        log_file_name=log_file_all_features
    )
    #clf = automl
    y_prob = automl.predict_proba(X_train)[:, 1]
    youden_stat_initial = []
    thresholds = np.arange(0.0, 1.0, 0.01)
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        tpr = recall_score(y_train, y_pred, pos_label=1)
        tnr = recall_score(y_train, y_pred, pos_label=0)
        youden_stat_initial.append(tpr + tnr - 1)

    optimal_threshold = thresholds[np.argmax(youden_stat_initial)]
    y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
    train_accuracy = accuracy_score(y_train, y_pred_optimal)
    train_roc_auc = roc_auc_score(y_train, y_prob)
    train_avg_precision = average_precision_score(y_train, y_prob)
    train_recall = recall_score(y_train, y_pred_optimal)
    train_precision = precision_score(y_train, y_pred_optimal)
    train_f1 = f1_score(y_train, y_pred_optimal)
    train_mcc = matthews_corrcoef(y_train, y_pred_optimal)


    log_to_csv(f'All Features Train Accuracy: {train_accuracy}')
    log_to_csv(f'All Features Train ROC AUC: {train_roc_auc}')
    log_to_csv(f'All Features Train Average Precision: {train_avg_precision}')
    log_to_csv(f'All Features Train Recall: {train_recall}')
    log_to_csv(f'All Features Train Precision: {train_precision}')
    log_to_csv(f'All Features Train F1 Score: {train_f1}')
    log_to_csv(f'All Features Train MCC: {train_mcc}')

    y_prob_test = automl.predict_proba(X_test)[:, 1]
    y_pred_test_optimal = (y_prob_test >= optimal_threshold).astype(int)
    test_accuracy = accuracy_score(y_test, y_pred_test_optimal)
    test_roc_auc = roc_auc_score(y_test, y_prob_test)
    test_avg_precision = average_precision_score(y_test, y_prob_test)
    test_recall = recall_score(y_test, y_pred_test_optimal)
    test_precision = precision_score(y_test, y_pred_test_optimal)
    test_f1 = f1_score(y_test, y_pred_test_optimal)
    test_mcc = matthews_corrcoef(y_test, y_pred_test_optimal)

    log_to_csv(f'All Features Test Accuracy: {test_accuracy}')
    log_to_csv(f'All Features Test ROC AUC: {test_roc_auc}')
    log_to_csv(f'All Features Test Average Precision: {test_avg_precision}')
    log_to_csv(f'All Features Test Recall: {test_recall}')
    log_to_csv(f'All Features Test Precision: {test_precision}')
    log_to_csv(f'All Features Test F1 Score: {test_f1}')
    log_to_csv(f'All Features Test MCC: {test_mcc}')

    from datetime import datetime
    if automl.feature_importances_ is not None:
    # Get the top 100 features based on importance
        if len(automl.feature_importances_) == 1:
            top_feature_names = np.array(automl.feature_names_in_)[np.argsort(abs(automl.feature_importances_[0]))[::-1]][:100]
        else:
            top_feature_names = np.array(automl.feature_names_in_)[np.argsort(abs(automl.feature_importances_))[::-1]][:100]
            
    else:
        log_to_csv("Feature importances are not available. Make sure the model is properly trained.")

    #next step
    if len(automl.feature_importances_) == 1:
        imp_top_feature_names = np.array(automl.feature_importances_[0])[np.argsort(abs(automl.feature_importances_[0]))[::-1]][:100]
    else:
        imp_top_feature_names = np.array(automl.feature_importances_)[np.argsort(abs(automl.feature_importances_))[::-1]][:100]

    log_to_csv(top_feature_names)
    log_to_csv(imp_top_feature_names)
    
    # Experiment with the top 10 features
    train_accuracies, train_roc_aucs, train_avg_precisions, train_recalls, train_precisions, train_f1s, train_mccs = [], [], [], [], [], [], []
    test_accuracies, test_roc_aucs, test_avg_precisions, test_recalls, test_precisions, test_f1s, test_mccs = [], [], [], [], [], [], []
    num_features, tflist = [], []

    for j, tf in enumerate(top_feature_names):
        tflist.append(tf)
        current_time = datetime.now().time()
        log_to_csv(f'Running top {j+1} features: {tflist}, {current_time}')
    
    # Subset your train and test datasets to the selected features
        X_train_sub = X_train.loc[:, tflist]
        X_test_sub = X_test.loc[:, tflist]

    # Retrain using AutoML with the subset of features
        automl = AutoML()
        automl.retrain_from_log(
            log_file_name=log_file_all_features,
            X_train=X_train_sub,
            y_train=y_train,
            task='classification',
            train_full=True,
            n_jobs=-1,
            train_best=True
        )

    # Get the predicted probabilities for the training set
        y_prob_train = automl.predict_proba(X_train_sub)[:, 1]
    
    # Training set metrics
        youden_stat_iterative = []
        thresholds = np.arange(0.0, 1.0, 0.01)
    
    # Calculate Youden's statistic for the training set
        for threshold in thresholds:
            y_pred_train = (y_prob_train >= threshold).astype(int)
            tpr = recall_score(y_train, y_pred_train, pos_label=1)
            tnr = recall_score(y_train, y_pred_train, pos_label=0)
            youden_stat_iterative.append(tpr + tnr - 1)

        optimal_threshold = thresholds[np.argmax(youden_stat_iterative)]
        y_pred_train_optimal = (y_prob_train >= optimal_threshold).astype(int)
    
        train_accuracies.append(accuracy_score(y_train, y_pred_train_optimal))
        train_roc_aucs.append(roc_auc_score(y_train, y_prob_train))
        train_avg_precisions.append(average_precision_score(y_train, y_prob_train))
        train_recalls.append(recall_score(y_train, y_pred_train_optimal))
        train_precisions.append(precision_score(y_train, y_pred_train_optimal))
        train_f1s.append(f1_score(y_train, y_pred_train_optimal))
        train_mccs.append(matthews_corrcoef(y_train, y_pred_train_optimal))
    
    # Predicted probabilities on testing test
        y_prob_test = automl.predict_proba(X_test_sub)[:, 1]
    
    # Testing set metrics
    
        y_pred_test_optimal = (y_prob_test >= optimal_threshold).astype(int)
        test_accuracies.append(accuracy_score(y_test, y_pred_test_optimal))
        test_roc_aucs.append(roc_auc_score(y_test, y_prob_test))
        test_avg_precisions.append(average_precision_score(y_test, y_prob_test))
        test_recalls.append(recall_score(y_test, y_pred_test_optimal))
        test_precisions.append(precision_score(y_test, y_pred_test_optimal))
        test_f1s.append(f1_score(y_test, y_pred_test_optimal))
        test_mccs.append(matthews_corrcoef(y_test, y_pred_test_optimal))
    
        num_features.append(j + 1)
    #the for loop closes at this point. the for loop indent is here, everything else should indent start here.
    #Next step is to print out the metrics

    train_metrics_df = pd.DataFrame({
        'num_features': num_features,
        'accuracy': train_accuracies,
        'roc_auc': train_roc_aucs,
        'avg_precision': train_avg_precisions,
        'recall': train_recalls,
        'precision': train_precisions,
        'f1': train_f1s,
        'mcc': train_mccs
    })

    test_metrics_df = pd.DataFrame({
        'num_features': num_features,
        'accuracy': test_accuracies,
        'roc_auc': test_roc_aucs,
        'avg_precision': test_avg_precisions,
        'recall': test_recalls,
        'precision': test_precisions,
        'f1': test_f1s,
        'mcc': test_mccs
    })

    # Display the DataFrames
    #log_to_csv("Training Metrics DataFrame:")
    #log_to_csv(train_metrics_df.head())

    #log_to_csv("Testing Metrics DataFrame:")
    #log_to_csv(test_metrics_df.head())
    # Export the combined DataFrame to a CSV file

    combined_metrics_df = pd.DataFrame({
        'num_features': num_features,
        'train_accuracy': train_accuracies,
        'train_roc_auc': train_roc_aucs,
        'train_avg_precision': train_avg_precisions,
        'train_recall': train_recalls,
        'train_precision': train_precisions,
        'train_f1': train_f1s,
        'train_mcc': train_mccs,
        'test_accuracy': test_accuracies,
        'test_roc_auc': test_roc_aucs,
        'test_avg_precision': test_avg_precisions,
        'test_recall': test_recalls,
        'test_precision': test_precisions,
        'test_f1': test_f1s,
        'test_mcc': test_mccs
    })

    output_filename = f'{log_dir_path}/Output_files/{cell_type}_combined_metrics.csv'

    combined_metrics_df.to_csv(output_filename, index=False)

    combined_metrics_df['roc_difference'] = abs(combined_metrics_df['train_roc_auc'] - combined_metrics_df['test_roc_auc'])
    min_roc_difference = combined_metrics_df['roc_difference'].idxmin()
    best_iteration = combined_metrics_df.iloc[min_roc_difference]
    log_to_csv(best_iteration)

    #Next step is to print a graph polotting traing and testing roc
    import matplotlib.pyplot as plt

    #plt.plot(combined_metrics_df['train_roc_auc'], combined_metrics_df['test_roc_auc'], color='blue')
    #plt.plot((0,1),(0,1))
    #plt.grid(True)
    #plt.xlabel('train_roc_auc')
    #plt.ylabel('test_roc_auc')
    #plt.legend()
    #plt.title('trainvstest')
    #plt.savefig()

    from flaml import AutoML
    from sklearn.model_selection import train_test_split

    # Assume X and y are your full datasets
    # Subset to the top 10 features
    top_selected_features = top_feature_names[:10]
    X_top = X_train[top_selected_features]
    log_to_csv("Feature selection done")

    # Initialize AutoML
    automl = AutoML()

    # Run AutoML with 10-fold cross-validation and a 1200-second time budget
    automl.fit(
        X_top,
        y_train,
        task="classification",
        time_budget=1000,
        metrics='log_loss',
        n_jobs=-1,
        eval_method='cv',
        n_splits=10,
        split_type='stratified',
        log_training_metric=True, 
        early_stop=True, 
        seed=239875, 
        model_history=True, 
        estimator_list=['lgbm'],
        log_file_name=f'{cell_type}_automl_top_features.log'
    )

    log_to_csv(top_selected_features)

    youden_stat_top_ten_features = []
    thresholds = np.arange(0.0, 1.0, 0.01)


    y_prob_train = clf.predict_proba(X_train)[:, 1]

    for threshold in thresholds:
        y_pred_train = (y_prob_train >= threshold).astype(int)
        tpr = recall_score(y_train, y_pred_train, pos_label=1)
        tnr = recall_score(y_train, y_pred_train, pos_label=0)
        youden_stat_top_ten_features.append(tpr + tnr - 1)


    optimal_threshold = thresholds[np.argmax(youden_stat_top_ten_features)]


    y_prob_test = clf.predict_proba(X_test)[:, 1]
    y_pred_test_optimal = (y_prob_test >= optimal_threshold).astype(int)


    test_accuracy = accuracy_score(y_test, y_pred_test_optimal)
    test_roc_auc = roc_auc_score(y_test, y_prob_test)
    test_avg_precision = average_precision_score(y_test, y_prob_test)
    test_recall = recall_score(y_test, y_pred_test_optimal)
    test_precision = precision_score(y_test, y_pred_test_optimal)
    test_f1 = f1_score(y_test, y_pred_test_optimal)
    test_mcc = matthews_corrcoef(y_test, y_pred_test_optimal)


    log_to_csv("Best model:", automl.best_estimator)
    log_to_csv("Best hyperparameters:", automl.best_config)
    log_to_csv("Best CV score:", automl.best_loss)

    log_to_csv(f'Test Set Accuracy: {test_accuracy}')
    log_to_csv(f'Test Set ROC AUC: {test_roc_auc}')
    log_to_csv(f'Test Set Average Precision: {test_avg_precision}')
    log_to_csv(f'Test Set Recall: {test_recall}')
    log_to_csv(f'Test Set Precision: {test_precision}')
    log_to_csv(f'Test Set F1 Score: {test_f1}')
    log_to_csv(f'Test Set MCC: {test_mcc}')
if __name__ == "__main__":
    main()