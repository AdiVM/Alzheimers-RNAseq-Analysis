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

log_dir_path = "/n/groups/patel/adithya/Log_Dir_Maximal/"
LOG_FILE_PATH = os.path.expanduser(f'{log_dir_path}experiment_log.txt')


def main():
    parser = argparse.ArgumentParser(description='Run AutoML on combined gene expression and metadata data')
    parser.add_argument('--exp_type', type=str, choices=['maximal'], required=True, help='Specify experiment type')
    args = parser.parse_args()
    
    exp_type = args.exp_type

    log_message = f"Processing {exp_type} data with full integration of gene and metadata features"
    with open(LOG_FILE_PATH, 'a') as log_file:
        log_file.write(log_message + '\n')

    # Load the data
    train_metadata = pd.read_parquet('/home/adm808/CellMetadataSyn18485175.parquet')
    print("Train metadata is loaded")
    test_metadata = pd.read_csv('/home/adm808/UpdatedCellMetadataSyn16780177.csv', low_memory=False)
    print("Test metadata is loaded")

    # Process APOE genotype as categorical -- Hot encoding of apoe_genotype
    combined_metadata = pd.concat([train_metadata, test_metadata], keys=['train', 'test'])
    combined_metadata = pd.get_dummies(combined_metadata, columns=["apoe_genotype"])
    apoe_genotype_columns = [col for col in combined_metadata.columns if col.startswith("apoe_genotype_")]

    # Split back into train and test metadata
    train_metadata = combined_metadata.xs('train')
    test_metadata = combined_metadata.xs('test')

    # Define Alzheimer's or control status
    train_metadata = train_metadata.copy()
    test_metadata = test_metadata.copy()
    train_metadata['alzheimers_or_control'] = train_metadata['age_first_ad_dx'].notnull().astype(int)
    test_metadata['alzheimers_or_control'] = test_metadata['age_first_ad_dx'].notnull().astype(int)

    print(f"Number of cases in training: {sum(train_metadata['alzheimers_or_control'])}")
    print(f"Number of cases in test: {sum(test_metadata['alzheimers_or_control'])}")

    # Function to select and drop missing genes
    def select_missing_genes(filtered_matrix):
        mean_threshold = 1
        missingness_threshold = 95
    
        mean_gene_expression = filtered_matrix.mean(axis=0)
        missingness = (filtered_matrix == 0).sum(axis=0) / filtered_matrix.shape[0] * 100
        null_expression = (missingness > missingness_threshold) & (mean_gene_expression < mean_threshold)
        genes_to_drop = filtered_matrix.columns[null_expression].tolist()
    
        return genes_to_drop

    # Transpose and load gene expression matrices
    # Load and transpose gene expression matrices
    train_matrix = pd.read_parquet('/home/adm808/NormalizedCellMatrixSyn18485175.parquet').T
    test_matrix = pd.read_parquet('/home/adm808/NormalizedCellMatrixSyn16780177.parquet').T
    print("Train and test matrices are loaded")
    print(train_matrix.iloc[:, :5].head())
    print(test_matrix.iloc[:, :5].head())

    print("Printing dimensionality of X_train and X_test initallly")
    print(train_matrix.shape)
    print(test_matrix.shape)
    
    # Filter missing genes
    train_matrix_filtered = train_matrix.drop(select_missing_genes(train_matrix), axis=1)
    test_matrix_filtered = test_matrix.drop(select_missing_genes(test_matrix), axis=1)
    
    # Merge the train and test matrices with their respective metadata files

    train_data = train_matrix_filtered.merge(
        train_metadata[['TAG', 'msex', 'broad.cell.type', 'alzheimers_or_control'] + apoe_genotype_columns],
        left_index=True,
        right_on='TAG',
        how='inner'
    ).set_index('TAG')
    
    test_data = test_matrix_filtered.merge(
        test_metadata[['TAG', 'msex', 'broad.cell.type', 'alzheimers_or_control'] + apoe_genotype_columns],
        left_index=True,
        right_on='TAG',
        how='inner'
    ).set_index('TAG')


    
    
    # Clean column names for model compatibility
    train_data.columns = train_data.columns.str.replace(r'[^A-Za-z0-9_]+', '', regex=True)
    test_data.columns = test_data.columns.str.replace(r'[^A-Za-z0-9_]+', '', regex=True)
    
    # Ensure common genes are used between training and testing sets
    common_genes = train_data.columns.intersection(test_data.columns)
    X_train = train_data[common_genes]
    X_test = test_data[common_genes]

    # Drop the alzheimers or control column from the dataset
    X_train = X_train.drop(columns=['alzheimers_or_control'])
    X_test = X_test.drop(columns=['alzheimers_or_control'])
    
    # Map original column names to cleaned names for later interpretability
    original_columns = common_genes  # Use common genes after filtering
    cleaned_columns = original_columns.str.replace(r'[^A-Za-z0-9_]+', '', regex=True)
    column_mapping = dict(zip(cleaned_columns, original_columns))
    
    # Define the target variable
    y_train = train_data['alzheimers_or_control']
    y_test = test_data['alzheimers_or_control']

    print("Printing dimensionality of X_train and X_test post filtering and merging")

    print(X_train.shape)
    print(X_test.shape)


    # Run maximal classification experiment on all data
    output_csv = f'{log_dir_path}maximal_output_log.csv'
    print("Starting all features classification")
    maximal_classifier = AutoML()
    maximal_classifier.fit(
        X_train, y_train,
        task="classification", time_budget=12000, metric='log_loss',
        n_jobs=-1, eval_method='cv', n_splits=10, split_type='stratified',
        log_training_metric=True, early_stop=True, seed=239875, estimator_list=['lgbm'],
        log_file_name=f"{log_dir_path}/all_features_log.txt"
    )

    lgbm_model_path = f"{log_dir_path}lgbm_model_all_features.pkl"
    joblib.dump(maximal_classifier.model, lgbm_model_path)
    print(f"LGBM model saved to {lgbm_model_path}")

    # Predictions and optimal threshold using Youden's J statistic
    y_prob_train = maximal_classifier.predict_proba(X_train)[:, 1]
    y_prob_test = maximal_classifier.predict_proba(X_test)[:, 1]

    thresholds = np.arange(0.0, 1.0, 0.01)
    youden_stats = [(recall_score(y_train, (y_prob_train >= t).astype(int)) +
                     recall_score(y_train, (y_prob_train >= t).astype(int), pos_label=0) - 1)
                    for t in thresholds]
    optimal_threshold = thresholds[np.argmax(youden_stats)]
    
    y_pred_train_optimal = (y_prob_train >= optimal_threshold).astype(int)
    y_pred_test_optimal = (y_prob_test >= optimal_threshold).astype(int)

    # Calculate metrics
    metrics = {
        'train_accuracy': accuracy_score(y_train, y_pred_train_optimal),
        'train_roc_auc': roc_auc_score(y_train, y_prob_train),
        'train_avg_precision': average_precision_score(y_train, y_prob_train),
        'train_recall': recall_score(y_train, y_pred_train_optimal),
        'train_precision': precision_score(y_train, y_pred_train_optimal),
        'train_f1': f1_score(y_train, y_pred_train_optimal),
        'train_mcc': matthews_corrcoef(y_train, y_pred_train_optimal),
        'test_accuracy': accuracy_score(y_test, y_pred_test_optimal),
        'test_roc_auc': roc_auc_score(y_test, y_prob_test),
        'test_avg_precision': average_precision_score(y_test, y_prob_test),
        'test_recall': recall_score(y_test, y_pred_test_optimal),
        'test_precision': precision_score(y_test, y_pred_test_optimal),
        'test_f1': f1_score(y_test, y_pred_test_optimal),
        'test_mcc': matthews_corrcoef(y_test, y_pred_test_optimal)
    }

    pd.DataFrame([metrics]).to_csv(output_csv, index=False)

    # Feature importance for top 100 features and I try to avoid the mismatch error that was occuring
    # if maximal_classifier.feature_importances_ is not None:
    #     # Retrieve the features actually used by the model
    #     used_features = maximal_classifier.feature_names_in_

    #     # Create the feature importance Series with used features only
    #     feature_importance = pd.Series(maximal_classifier.feature_importances_, index=used_features)

    #     # Get the top 100 features and map them back to original names for interpretability
    #     top_features_cleaned = feature_importance.nlargest(100).index
    #     top_features_original = [column_mapping.get(feature, feature) for feature in top_features_cleaned]
        
    #     # Refit model on top features
    #     X_train_top = X_train[top_features_cleaned]
    #     X_test_top = X_test[top_features_cleaned]

    #     top_maximal_classifier = AutoML()
    #     top_maximal_classifier.fit(
    #         X_train_top, y_train,
    #         task="classification", time_budget=800, metric='log_loss',
    #         n_jobs=-1, eval_method='cv', n_splits=10, split_type='stratified',
    #         log_training_metric=True, early_stop=True, seed=239875, estimator_list=['lgbm']
    #     )

    #     y_prob_train_top = top_maximal_classifier.predict_proba(X_train_top)[:, 1]
    #     y_prob_test_top = top_maximal_classifier.predict_proba(X_test_top)[:, 1]


    #     top_metrics = {
    #         'train_accuracy': accuracy_score(y_train, (y_prob_train_top >= optimal_threshold).astype(int)),
    #         'train_roc_auc': roc_auc_score(y_train, y_prob_train_top),
    #         'train_avg_precision': average_precision_score(y_train, y_prob_train_top),
    #         'train_recall': recall_score(y_train, (y_prob_train_top >= optimal_threshold).astype(int)),
    #         'train_precision': precision_score(y_train, (y_prob_train_top >= optimal_threshold).astype(int)),
    #         'train_f1': f1_score(y_train, (y_prob_train_top >= optimal_threshold).astype(int)),
    #         'train_mcc': matthews_corrcoef(y_train, (y_prob_train_top >= optimal_threshold).astype(int)),
    #         'test_accuracy': accuracy_score(y_test, (y_prob_test_top >= optimal_threshold).astype(int)),
    #         'test_roc_auc': roc_auc_score(y_test, y_prob_test_top),
    #         'test_avg_precision': average_precision_score(y_test, y_prob_test_top),
    #         'test_recall': recall_score(y_test, (y_prob_test_top >= optimal_threshold).astype(int)),
    #         'test_precision': precision_score(y_test, (y_prob_test_top >= optimal_threshold).astype(int)),
    #         'test_f1': f1_score(y_test, (y_prob_test_top >= optimal_threshold).astype(int)),
    #         'test_mcc': matthews_corrcoef(y_test, (y_prob_test_top >= optimal_threshold).astype(int))
    #     }

    #     pd.DataFrame([top_metrics]).to_csv(f'{log_dir_path}top_100_features_metrics.csv', index=False)


        
    #     print("Top 100 features model training complete")

    # Feature importance for top 100 features and avoid mismatch error
    print("Starting iterative feature importances")
    if maximal_classifier.feature_importances_ is not None:
        # Retrieve the features actually used by the model
        used_features = maximal_classifier.feature_names_in_
    
        # Create the feature importance Series with used features only
        feature_importance = pd.Series(maximal_classifier.feature_importances_, index=used_features)
    
        # Get the top 100 features and map them back to original names for interpretability
        top_features_cleaned = feature_importance.nlargest(100).index
        top_features_original = [column_mapping.get(feature, feature) for feature in top_features_cleaned]
        
        # --- Start Incremental Evaluation of Top Features ---
    
        # Prepare to store incremental results for top features
        incremental_results = []
    
        # Loop through 1 to 100 features, adding one feature at a time
        for i in range(1, 101):
            # Select the top `i` features
            current_features = top_features_cleaned[:i]
            X_train_top_i = X_train[current_features]
            X_test_top_i = X_test[current_features]
    
            # Train the model on the current subset of top features
            incremental_classifier = AutoML()
            incremental_classifier.fit(
                X_train_top_i, y_train,
                task="classification", time_budget=150, metric='log_loss',
                n_jobs=-1, eval_method='cv', n_splits=10, split_type='stratified',
                log_training_metric=True, early_stop=True, seed=239875, estimator_list=['lgbm']
            )
    
            # Predict probabilities and apply optimal threshold
            y_prob_train_i = incremental_classifier.predict_proba(X_train_top_i)[:, 1]
            y_prob_test_i = incremental_classifier.predict_proba(X_test_top_i)[:, 1]
            
            y_pred_train_i = (y_prob_train_i >= optimal_threshold).astype(int)
            y_pred_test_i = (y_prob_test_i >= optimal_threshold).astype(int)
    
            # Record metrics for this iteration
            result = {
                'num_features': i,
                'names_of_features': current_features,
                'train_accuracy': accuracy_score(y_train, y_pred_train_i),
                'train_roc_auc': roc_auc_score(y_train, y_prob_train_i),
                'train_avg_precision': average_precision_score(y_train, y_prob_train_i),
                'train_recall': recall_score(y_train, y_pred_train_i),
                'train_precision': precision_score(y_train, y_pred_train_i),
                'train_f1': f1_score(y_train, y_pred_train_i),
                'train_mcc': matthews_corrcoef(y_train, y_pred_train_i),
                'test_accuracy': accuracy_score(y_test, y_pred_test_i),
                'test_roc_auc': roc_auc_score(y_test, y_prob_test_i),
                'test_avg_precision': average_precision_score(y_test, y_prob_test_i),
                'test_recall': recall_score(y_test, y_pred_test_i),
                'test_precision': precision_score(y_test, y_pred_test_i),
                'test_f1': f1_score(y_test, y_pred_test_i),
                'test_mcc': matthews_corrcoef(y_test, y_pred_test_i)
            }
    
            incremental_results.append(result)
    
        # Convert results to DataFrame and save to CSV
        incremental_results_df = pd.DataFrame(incremental_results)
        incremental_results_df.to_csv(f'{log_dir_path}incremental_top_features_metrics.csv', index=False)
    
        print("Incremental top features model training completed")

    # --- End Incremental Evaluation of Top Features ---


    print("Maximal experiment completed")


if __name__ == "__main__":
    main()
