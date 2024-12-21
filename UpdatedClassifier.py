import argparse
import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score, recall_score,
    precision_score, f1_score, matthews_corrcoef
)
from flaml import AutoML
import csv
import matplotlib.pyplot as plt

log_dir_path = "/n/groups/patel/adithya/Log_Dir_withDemographics/"
LOG_FILE_PATH = os.path.expanduser(f'{log_dir_path}experiment_log.txt')

def main():
    import argparse
    import os
    import pandas as pd
    import numpy as np
    from sklearn.metrics import (
        accuracy_score, roc_auc_score, average_precision_score, recall_score,
        precision_score, f1_score, matthews_corrcoef)
    from flaml import AutoML
    import csv
    import matplotlib.pyplot as plt
    
    log_dir_path = "/n/groups/patel/adithya/Log_Dir_withDemographics/"


    # Create the argument parser
    parser = argparse.ArgumentParser(description='Run AutoML on either gene expression or demographic data')
    parser.add_argument('--exp_type', type=str, choices=['genes', 'demographics'], required=True, help='Specify experiment type')
    parser.add_argument('--cell_type', type=str, choices=['In', 'Mic', 'Oli', 'Opc', 'Per', 'Ast', 'End', 'Ex'], required=True, help='Specify the cell type')
    args = parser.parse_args()
    
    exp_type = args.exp_type
    cell_type = args.cell_type
    
    log_message = f"Processing {exp_type} data for cell type: {cell_type}"
    
    #log log message
    with open(LOG_FILE_PATH, 'a') as log_file:
        log_file.write(log_message + '\n')

    # Load the data
    train_metadata = pd.read_parquet('/home/adm808/CellMetadataSyn18485175.parquet')
    #1.1 MB
    print("train metadata is loaded")
    test_metadata = pd.read_csv('/home/adm808/UpdatedCellMetadataSyn16780177.csv', low_memory=False)
    #test_metadata = pd.read_csv('/home/adm808/UpdatedCellMetadataSyn16780177.csv')

    
    #overwriting the train_metadata and test_metadata to make the apoe_genotype a categorical variable
    # train_metadata = pd.get_dummies(train_metadata, columns=["apoe_genotype"])
    # test_metadata = pd.get_dummies(test_metadata, columns=["apoe_genotype"])
    combined_metadata = pd.concat([train_metadata, test_metadata], keys=['train', 'test'])
    combined_metadata = pd.get_dummies(combined_metadata, columns=["apoe_genotype"])

    # Split back into train and test
    train_metadata = combined_metadata.xs('train')
    test_metadata = combined_metadata.xs('test')


    # train_metadata = pd.get_dummies(train_metadata, columns=["apoe_genotype"])
    # test_metadata = pd.get_dummies(test_metadata, columns=["apoe_genotype"])

    # Determine Alzheimer's or control status
    train_metadata['alzheimers_or_control'] = train_metadata['age_first_ad_dx'].notnull().astype(int)
    test_metadata['alzheimers_or_control'] = test_metadata['age_first_ad_dx'].notnull().astype(int)
    print(f"Number of cases in training: {sum(train_metadata['alzheimers_or_control'])}")
    print(f"Number of cases in test: {sum(test_metadata['alzheimers_or_control'])}")
    


    



    # Function to select and drop missing genes
    def select_missing_genes(filtered_matrix):
        mean_threshold = 1
        missingness_threshold = 95
        mean_gene_expression = np.mean(filtered_matrix, axis=1)
        missingness = (filtered_matrix == 0).sum(axis=1) / filtered_matrix.shape[1] * 100
        null_expression = (missingness > missingness_threshold) & (mean_gene_expression < mean_threshold)
        genes_to_drop = filtered_matrix.index[null_expression].tolist()
        return genes_to_drop

    # Function to filter gene matrix for a specific cell type
    def drop_missing_genes(matrix, cell_metadata, cell_type):
        cell_type_specific_metadata = cell_metadata[cell_metadata['broad.cell.type'] == cell_type]
        cell_names = cell_type_specific_metadata['TAG']
        matrix_filtered = matrix[cell_names]
        genes_to_drop = select_missing_genes(matrix_filtered)
        return matrix_filtered.drop(genes_to_drop, axis=0)

    # Function to filter metadata for a specific cell type
    def filter_metadata(cell_metadata, cell_type):
        return cell_metadata[cell_metadata['broad.cell.type'] == cell_type]

    # We pe[erform our data preprocessing based on exp_type
    if exp_type == 'genes':
        train_matrix = pd.read_parquet('/home/adm808/NormalizedCellMatrixSyn18485175.parquet')
        #286.1mb
        print("train matrix is loaded")

            #16.8mb
        print("test metadata is loaded")
        test_matrix = pd.read_parquet('/home/adm808/NormalizedCellMatrixSyn16780177.parquet')
        #714.6mb
        print("test matrix is loaded")

        
        # Filter and prepare gene expression data
        train_matrix_filtered = drop_missing_genes(train_matrix, train_metadata, cell_type)
        test_matrix_filtered = drop_missing_genes(test_matrix, test_metadata, cell_type)

        train_metadata_filtered = filter_metadata(train_metadata, cell_type)
        test_metadata_filtered = filter_metadata(test_metadata, cell_type)

        # Identify common genes between training and testing sets
        common_genes = train_matrix_filtered.index.intersection(test_matrix_filtered.index)
        X_train = train_matrix_filtered.loc[common_genes].T
        X_test = test_matrix_filtered.loc[common_genes].T

        y_train = train_metadata_filtered.set_index('TAG').loc[X_train.index, 'alzheimers_or_control']
        y_test = test_metadata_filtered.set_index('TAG').loc[X_test.index, 'alzheimers_or_control']

        # Run initial classifier
        output_csv = f'{log_dir_path}{cell_type}_genes_output_log.csv'
        classifier = AutoML()
        classifier.fit(
            X_train, y_train,
            task="classification", time_budget=1000, metric='log_loss',
            n_jobs=-1, eval_method='cv', n_splits=10, split_type='stratified',
            log_training_metric=True, early_stop=True, seed=239875, estimator_list=['lgbm']
        )

        # Make predictions
        y_prob_train = classifier.predict_proba(X_train)[:, 1]
        y_prob_test = classifier.predict_proba(X_test)[:, 1]

        # Optimal threshold using Youden's J statistic
        thresholds = np.arange(0.0, 1.0, 0.01)
        youden_stats = [(recall_score(y_train, (y_prob_train >= t).astype(int)) +
                         recall_score(y_train, (y_prob_train >= t).astype(int), pos_label=0) - 1)
                        for t in thresholds]
        optimal_threshold = thresholds[np.argmax(youden_stats)]

        y_pred_train_optimal = (y_prob_train >= optimal_threshold).astype(int)
        y_pred_test_optimal = (y_prob_test >= optimal_threshold).astype(int)

        # Calculate performance metrics for the genes model
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_pred_train_optimal),
            'train_roc_auc': roc_auc_score(y_train, y_prob_train),
            'train_avg_precision': average_precision_score(y_train, y_prob_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test_optimal),
            'test_roc_auc': roc_auc_score(y_test, y_prob_test),
            'test_avg_precision': average_precision_score(y_test, y_prob_test)
        }

        pd.DataFrame([metrics]).to_csv(output_csv, index=False)

        # Top 10 genes classifier
        if classifier.feature_importances_ is not None:
            top_genes = np.array(classifier.feature_names_in_)[np.argsort(classifier.feature_importances_)[::-1][:10]]
            roc_data = []

            for i in range(1, 11):
                top_i_genes = top_genes[:i]
                X_train_top = X_train[top_i_genes]
                X_test_top = X_test[top_i_genes]

                # Retrain classifier using top i genes
                my2classifier.fit(
                    X_train_top, y_train,
                    task="classification", time_budget=500, metric='log_loss',
                    n_jobs=-1, eval_method='cv', n_splits=10, split_type='stratified',
                    log_training_metric=True, early_stop=True, seed=239875, estimator_list=['lgbm']
                )

            y_prob_train = classifier.predict_proba(X_train)[:, 1]
            y_prob_test = classifier.predict_proba(X_test)[:, 1]

            # Optimal threshold using Youden's J statistic
            thresholds = np.arange(0.0, 1.0, 0.01)
            youden_stats = [(recall_score(y_train, (y_prob_train >= t).astype(int)) + 
                             recall_score(y_train, (y_prob_train >= t).astype(int), pos_label=0) - 1)
                            for t in thresholds]
            optimal_threshold = thresholds[np.argmax(youden_stats)]

            y_pred_train_optimal = (y_prob_train >= optimal_threshold).astype(int)
            y_pred_test_optimal = (y_prob_test >= optimal_threshold).astype(int)

        log_to_csv(top_selected_features)
    
        youden_stat_top_ten_features = []
        thresholds = np.arange(0.0, 1.0, 0.01)
    
        y_prob_train = my2classifier.predict_proba(X_top_train)[:, 1]
    
        for threshold in thresholds:
            y_pred_train = (y_prob_train >= threshold).astype(int)
            tpr = recall_score(y_train, y_pred_train, pos_label=1)
            tnr = recall_score(y_train, y_pred_train, pos_label=0)
            youden_stat_top_ten_features.append(tpr + tnr - 1)
    
        optimal_threshold = thresholds[np.argmax(youden_stat_top_ten_features)]
    
        y_prob_test = my2classifier.predict_proba(X_top_test)[:, 1]
        y_pred_test_optimal = (y_prob_test >= optimal_threshold).astype(int)
    
        test_accuracy = accuracy_score(y_test, y_pred_test_optimal)
        test_roc_auc = roc_auc_score(y_test, y_prob_test)
        test_avg_precision = average_precision_score(y_test, y_prob_test)
        test_recall = recall_score(y_test, y_pred_test_optimal)
        test_precision = precision_score(y_test, y_pred_test_optimal)
        test_f1 = f1_score(y_test, y_pred_test_optimal)
        test_mcc = matthews_corrcoef(y_test, y_pred_test_optimal)
    
        log_to_csv(f'Top Features Test Set Accuracy: {test_accuracy}')
        log_to_csv(f'Top Features Test Set ROC AUC: {test_roc_auc}')
        log_to_csv(f'Top Features Test Set Average Precision: {test_avg_precision}')
        log_to_csv(f'Top Features Test Set Recall: {test_recall}')
        log_to_csv(f'Top Features Test Set Precision: {test_precision}')
        log_to_csv(f'Top Features Test Set F1 Score: {test_f1}')
        log_to_csv(f'Top Features Test Set MCC: {test_mcc}')
        
    #DEMOGRAPHICS classifier
    elif exp_type == 'demographics':
        print("we are using demographics")

        #i intialize the lists to hold the train_accuracies, train_roc_aucs, train_avg_precisions and 
        
        train_accuracies = []
        train_roc_aucs = []
        train_avg_precisions = []
        train_recalls = []
        train_precisions = []
        train_f1s = []
        train_mccs = []

        test_accuracies = []
        test_roc_aucs = []
        test_avg_precisions = []
        test_recalls = []
        test_precisions = []
        test_f1s = []
        test_mccs = []

        
        #prepare demographic data
        
        train_metadata_filtered = train_metadata.drop_duplicates(subset = ["sample"])
        print(f"Train metadata filtered shape {train_metadata_filtered.shape}")
        test_metadata_filtered = test_metadata.drop_duplicates(subset = ["sample"])
        print(f"Test metadata filtered shape {test_metadata_filtered.shape}")

        # X_train = train_metadata_filtered[["msex", "apoe_genotype"]]
        # X_test = test_metadata_filtered[["msex", "apoe_genotype"]]
        #this line needs to be updated after we changed the apoe_genotype variable

        apoe_columns = [col for col in train_metadata.columns if col.startswith("apoe_genotype_")]
        X_train = train_metadata_filtered[["msex"] + apoe_columns]
        X_test = test_metadata_filtered[["msex"] + apoe_columns]

        

        # #same for this column need to chagne the name
        # apoe_columns_test = [col for col in test_metadata_filtered.columns if col.startswith("apoe_genotype_")]
        # X_test = test_metadata_filtered[["msex"] + apoe_columns_test]


        y_train = train_metadata_filtered['alzheimers_or_control']
        y_test = test_metadata_filtered['alzheimers_or_control']

        #run classifier on demogrpahic data
        output_csv = f'{log_dir_path}demographics_output_log.csv'
        classifier = AutoML()
        classifier.fit(
            X_train, y_train,
            task="classification", time_budget=60, metric='log_loss',
            n_jobs=-1, eval_method='cv', n_splits=10, split_type='stratified',
            log_training_metric=True, early_stop=True, seed=239875, estimator_list=['lgbm']
        )

        y_prob_train = classifier.predict_proba(X_train)[:, 1]
        y_prob_test = classifier.predict_proba(X_test)[:, 1]

        

        #optimal threshold using Youden's J statistic
        print("calculated youden's statistic")

        youden_stat_demographics = []
        thresholds = np.arange(0.0, 1.0, 0.01)
        
        for threshold in thresholds:
            y_pred_train = (y_prob_train >= threshold).astype(int)
            tpr = recall_score(y_train, y_pred_train, pos_label=1)
            tnr = recall_score(y_train, y_pred_train, pos_label=0)
            youden_stat_demographics.append(tpr + tnr - 1)
    
        optimal_threshold = thresholds[np.argmax(youden_stat_demographics)]
        y_pred_train_optimal = (y_prob_train >= optimal_threshold).astype(int)


        train_accuracies.append(accuracy_score(y_train, y_pred_train_optimal))
        train_roc_aucs.append(roc_auc_score(y_train, y_prob_train))
        train_avg_precisions.append(average_precision_score(y_train, y_prob_train))
        train_recalls.append(recall_score(y_train, y_pred_train_optimal))
        train_precisions.append(precision_score(y_train, y_pred_train_optimal))
        train_f1s.append(f1_score(y_train, y_pred_train_optimal))
        train_mccs.append(matthews_corrcoef(y_train, y_pred_train_optimal))

        
        
        y_pred_test_optimal = (y_prob_test >= optimal_threshold).astype(int)
        test_accuracies.append(accuracy_score(y_test, y_pred_test_optimal))
        test_roc_aucs.append(roc_auc_score(y_test, y_prob_test))
        test_avg_precisions.append(average_precision_score(y_test, y_prob_test))
        test_recalls.append(recall_score(y_test, y_pred_test_optimal))
        test_precisions.append(precision_score(y_test, y_pred_test_optimal))
        test_f1s.append(f1_score(y_test, y_pred_test_optimal))
        test_mccs.append(matthews_corrcoef(y_test, y_pred_test_optimal))
    
        demographics_combined_metrics_df = pd.DataFrame({
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
    
        demographic_output_filename = f'{log_dir_path}Output_files/demographicsonly.csv'
    
        demographics_combined_metrics_df.to_csv(demographic_output_filename, index=False)
    
        print("DEMOGRAPHICS ONLY MODEL COMPLETE")

        print("Finished")

if __name__ == "__main__":
    main()
