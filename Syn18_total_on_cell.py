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
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedGroupKFold

log_dir_path = "/n/groups/patel/adithya/Syn18_Log_Dir_Total_on_cell/"
LOG_FILE_PATH = os.path.expanduser(f'{log_dir_path}experiment_log.txt')


def main():
    parser = argparse.ArgumentParser(description='Run AutoML on combined gene expression and metadata data')
    parser.add_argument('--exp_type', type=str, choices=['maximal'], required=True, help='Specify experiment type')
    parser.add_argument('--cell_type', type=str, required=True, help='Specify the cell type to train on')
    args = parser.parse_args()
    
    exp_type = args.exp_type
    cell_type = args.cell_type


    log_message = f"Processing {exp_type} data with {cell_type} cells using full integration of gene and metadata features"
    with open(LOG_FILE_PATH, 'a') as log_file:
        log_file.write(log_message + '\n')

    # Load the data
    metadata = pd.read_parquet('/home/adm808/CellMetadataSyn18485175.parquet')
    print("Metadata is loaded")

    # Process APOE genotype as categorical -- Hot encoding of apoe_genotype
    metadata = pd.get_dummies(metadata, columns=["apoe_genotype"])
    apoe_genotype_columns = [col for col in metadata.columns if col.startswith("apoe_genotype_")]


    # Stratified Shuffle Split based on `sample_id`to split metadata
    # Define Alzheimer's or control status directly based on `age_first_ad_dx`
    metadata = metadata.copy()
    metadata['alzheimers_or_control'] = metadata['age_first_ad_dx'].notnull().astype(int)

    # Extract unique sample IDs and their associated Alzheimer's/control status -- drop duplicates
    sample_summary = metadata[['sample', 'alzheimers_or_control', 'msex']].drop_duplicates()

    # I need to create a combined stratification variable
    sample_summary['stratify_group'] = sample_summary['alzheimers_or_control'].astype(str) + "_" + sample_summary['msex'].astype(str)

    # Perform stratified train-test split on `sample_id`, stratified by `alzheimers_or_control`
    train_samples, test_samples = train_test_split(
        sample_summary['sample'],
        test_size=0.2,
        random_state=42,
        stratify=sample_summary['stratify_group']
    )

    # Filter metadata by train and test `sample_id`
    train_metadata = metadata[metadata['sample'].isin(train_samples)]
    test_metadata = metadata[metadata['sample'].isin(test_samples)]


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

    # Load and transpose gene expression matrices
    gene_matrix = pd.read_parquet('/home/adm808/NormalizedCellMatrixSyn18485175.parquet').T
    print("Gene matrix is loaded")
    print(gene_matrix.iloc[:, :5].head())

    # Defining training and testing matrices
    train_matrix = gene_matrix.loc[train_metadata['TAG']]
    test_matrix = gene_matrix.loc[test_metadata['TAG']]

    print("Printing dimensionality of X_train and X_test initallly")
    print(train_matrix.shape)
    print(test_matrix.shape)
    
    # Filter missing genes
    train_matrix_filtered = train_matrix.drop(select_missing_genes(train_matrix), axis=1)
    test_matrix_filtered = test_matrix.drop(select_missing_genes(test_matrix), axis=1)
    
    # Merge the train and test matrices with their respective metadata files

    train_data = train_matrix_filtered.merge(
        train_metadata[['TAG', 'msex', 'sample', 'broad.cell.type', 'alzheimers_or_control'] + apoe_genotype_columns],
        left_index=True,
        right_on='TAG',
        how='inner'
    ).set_index('TAG')
    
    test_data = test_matrix_filtered.merge(
        test_metadata[['TAG', 'msex', 'sample', 'broad.cell.type', 'alzheimers_or_control'] + apoe_genotype_columns],
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


    #########################################################################
    # Trying a new method of creating cross validation folds:
    from sklearn.model_selection import StratifiedGroupKFold
    import numpy as np

    def generate_valid_folds(X, y, groups, n_splits=10, max_retries=100):
        """
        Generate valid folds for StratifiedGroupKFold to ensure no fold has only one class.
        Retries until valid folds are created.
        """
        retries = 0
        while retries < max_retries:
            retries += 1
            valid_folds = True
            stratified_group_kfold = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=np.random.randint(1000))
            
            # Generate folds
            folds = list(stratified_group_kfold.split(X, y, groups))

            # Check for class balance in validation folds
            for fold, (train_idx, val_idx) in enumerate(folds):
                train_y, val_y = y.iloc[train_idx], y.iloc[val_idx]
                if len(val_y.unique()) < 2:  # Check if validation set has both classes
                    print(f"Retry {retries}: Fold {fold + 1} has only one class. Retrying...")
                    valid_folds = False
                    break

            if valid_folds:
                print(f"Valid folds generated after {retries} retries.")
                return folds  # Return valid folds

        raise ValueError("Unable to generate valid folds after maximum retries.")
    
    X_test = X_test[test_metadata['broad.cell.type'] == cell_type]
    y_test = y_test[test_metadata['broad.cell.type'] == cell_type]

    # Generate valid folds
    valid_folds = generate_valid_folds(
        X_train,  # Feature matrix
        y_train,  # Target variable
        groups=train_metadata['sample'],  # Group variable
        n_splits=10,
        max_retries=100
    )

    cell_log_dir = os.path.join(log_dir_path, cell_type)

    # Create the directory if it doesn’t exist
    os.makedirs(cell_log_dir, exist_ok=True)


    # For task one I am training on all cell types, but testing only on one specific cell type. Therefore, I will subset just the testing sets for cell type:
    


    # Use valid folds in AutoML
    maximal_classifier = AutoML()
    maximal_classifier.fit(
        X_train, y_train,
        task="classification",
        time_budget=12000,
        metric='log_loss',
        n_jobs=-1,
        eval_method='cv',
        split_type='custom',  # Use pre-split folds
        split=valid_folds,    # Provide the valid folds
        log_training_metric=True,
        early_stop=True,
        seed=239875,
        estimator_list=['lgbm'],
        model_history=True,
        log_file_name=f"{cell_log_dir}/all_features_log.txt"
    )




    # Save the full model using joblib

    joblib.dump(maximal_classifier, f'{cell_log_dir}/maximal_classifier.joblib')


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

    pd.DataFrame([metrics]).to_csv(f'{cell_log_dir}/output_csv.csv', index=False)
    


    # Feature importance for top 100 features and avoid mismatch error
    print("Starting iterative feature importances")
    
    # Extract top features using the function from Randy's code
    def get_top_features(automl, n_top=100):
        """
        Extract top features reliably from an AutoML model.
        Parameters:
        automl (object): The AutoML model object.
        n_top (int): The number of top features to extract.
        Returns:
        list: A list of the top feature names.
        """
        # Handle 1D or multi-dimensional feature_importances_
        if len(automl.feature_importances_) == 1:
            # Sort features by absolute importance
            feature_names = np.array(automl.feature_names_in_)[
                np.argsort(abs(automl.feature_importances_[0]))[::-1]
            ]
            fi = automl.feature_importances_[0][
                np.argsort(abs(automl.feature_importances_[0]))[::-1]
            ]
        else:
            feature_names = np.array(automl.feature_names_in_)[
                np.argsort(abs(automl.feature_importances_))[::-1]
            ]
            fi = automl.feature_importances_[
                np.argsort(abs(automl.feature_importances_))[::-1]
            ]
        
        # Extract the top n features
        feature_names_top = feature_names[:n_top]
        return feature_names_top

    # Start top feature extraction
    try:
        top_features_cleaned = get_top_features(maximal_classifier, n_top=100)
        print(f"Top 100 features extracted:\n{top_features_cleaned}")
    except ValueError as e:
        print(f"Error extracting features: {e}")
        return  # Exit if feature importances are unavailable

    # Map features back to original names for interpretability
    top_features_original = [column_mapping.get(feature, feature) for feature in top_features_cleaned]

    # --- Start Incremental Evaluation ---
    incremental_results = []

    for i, feature_subset in enumerate(top_features_cleaned[:100], start=1):
        print(f"Retraining model with top {i} features")
        current_features = top_features_cleaned[:i]
        
        # Subset data
        X_train_top_i = X_train[current_features]
        X_test_top_i = X_test[current_features]
        
        # Retrain using the existing log
        incremental_classifier = AutoML()
        incremental_classifier.retrain_from_log(log_file_name=f'{cell_log_dir}/all_features_log.txt', 
        X_train=X_train_top_i, 
        y_train=y_train
        )

        # Predict probabilities
        y_prob_train_i = incremental_classifier.predict_proba(X_train_top_i)[:, 1]
        y_prob_test_i = incremental_classifier.predict_proba(X_test_top_i)[:, 1]
        
        y_pred_train_i = (y_prob_train_i >= optimal_threshold).astype(int)
        y_pred_test_i = (y_prob_test_i >= optimal_threshold).astype(int)

        # Collect metrics
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

    # Save results
    incremental_results_df = pd.DataFrame(incremental_results)
    incremental_results_df.to_csv(f'{cell_log_dir}incremental_top_features_metrics.csv', index=False)
    print("Incremental evaluation completed successfully")

    


    print("Maximal experiment completed")


if __name__ == "__main__":
    main()
