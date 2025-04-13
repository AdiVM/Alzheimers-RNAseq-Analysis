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

# Change this path to use for genes only model but kept the actual file path the same

log_dir_path = "/n/groups/patel/adithya/Alz_Outputs/Final_Outputs/Cell_on_cell_both_modules/"
LOG_FILE_PATH = os.path.expanduser(f'{log_dir_path}experiment_log.txt')


def main():
    parser = argparse.ArgumentParser(description='Run AutoML on combined gene expression and metadata data')
    parser.add_argument('--exp_type', type=str, choices=['maximal'], required=True, help='Specify experiment type')
    parser.add_argument('--cell_type', type=str, required=True, help='Specify the cell type to train on')
    args = parser.parse_args()
    
    exp_type = args.exp_type
    cell_type = args.cell_type


    log_message = f"Processing {exp_type} data with {cell_type} cells using full integration of gene and metadata features"
    #log_message = f"Processing {exp_type} data with all cell types using full integration of gene and metadata features"
    with open(LOG_FILE_PATH, 'a') as log_file:
        log_file.write(log_message + '\n')

    cell_log_dir = os.path.join(log_dir_path, cell_type)
        # Create the directory if it doesn’t exist
    os.makedirs(cell_log_dir, exist_ok=True)

    # Load the data
    metadata = pd.read_parquet('/home/adm808/New_CellMetadataSyn1848517.parquet')
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
        random_state=4,
        stratify=sample_summary['stratify_group']
    )

    # Filter metadata by train and test `sample_id`
    train_metadata = metadata[metadata['sample'].isin(train_samples)]
    test_metadata = metadata[metadata['sample'].isin(test_samples)]

    # Filter both the training and testing for cell type -- This is cell on cell prediction
    train_metadata = train_metadata[train_metadata['broad.cell.type'] == cell_type]
    test_metadata = test_metadata[test_metadata['broad.cell.type'] == cell_type]


    print(f"Number of cases in training: {sum(train_metadata['alzheimers_or_control'])}")
    print(f"Number of cases in test: {sum(test_metadata['alzheimers_or_control'])}")

    # Function to select and drop missing genes
    def select_missing_genes(filtered_matrix):
        mean_threshold = 2
        missingness_threshold = 90
    
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
    genes_to_drop = select_missing_genes(train_matrix)
    train_matrix_filtered = train_matrix.drop(columns=genes_to_drop)
    test_matrix_filtered = test_matrix.drop(columns=[g for g in genes_to_drop if g in test_matrix.columns])

    # RFE STARTS HERE
    # from sklearn.feature_selection import RFE
    # from sklearn.feature_selection import RFECV
    # from sklearn.linear_model import LogisticRegression
    # from sklearn.model_selection import StratifiedKFold

    # # Define base estimator
    # rfe_estimator = LogisticRegression(max_iter=1000, solver='saga')

    # # Subset and match target labels
    # X_rfe = train_matrix_filtered.copy()
    # y_rfe = train_metadata.set_index('TAG').loc[X_rfe.index]['alzheimers_or_control']

    # # Run RFECV to find optimal number of features
    # rfecv = RFECV(estimator=rfe_estimator, step=0.1, cv=StratifiedKFold(5), scoring='roc_auc', verbose=1, n_jobs=-1)
    # rfecv.fit(X_rfe, y_rfe)

    # optimal_feature_count = rfecv.n_features_
    # print("Optimal number of features selected by RFECV:", optimal_feature_count)

    # # Use RFE with optimal number of features
    # selector = RFE(estimator=rfe_estimator, n_features_to_select=optimal_feature_count, step=0.1)
    # selector.fit(X_rfe, y_rfe)

    # selected_genes = X_rfe.columns[selector.support_]

    # train_matrix_filtered = train_matrix_filtered[selected_genes]
    # test_matrix_filtered = test_matrix_filtered[selected_genes]

    ################################################################
    # Adding a section to use Scanpy-based Module clustering inorder to perform initial unsupervvised clustering on the genes
    import scanpy as sc

    def cluster_genes_with_scanpy(train_matrix_filtered, test_matrix_filtered, log_dir):
        # Convert training matrix (cells x genes) into AnnData object with genes as variables
        adata = sc.AnnData(train_matrix_filtered)
        adata.var_names = train_matrix_filtered.columns
        adata.obs_names = train_matrix_filtered.index

        # Scale gene expression (gene-wise)
        sc.pp.scale(adata)
        
        # Transpose data to treat genes as observations and cells as variables
        adata_genes = sc.AnnData(adata.X.T)
        adata_genes.var_names = adata.obs_names  # cell names become vars
        adata_genes.obs_names = adata.var_names  # genes become obs

        # Normalize & PCA for clustering
        sc.pp.scale(adata_genes)
        sc.tl.pca(adata_genes, n_comps=10, svd_solver='arpack')
        sc.pp.neighbors(adata_genes, use_rep='X_pca')
        sc.tl.leiden(adata_genes, resolution=1.0)

        # Extract cluster labels for each gene
        gene_to_module = adata_genes.obs['leiden'].to_dict()

        # Save gene-to-module mapping
        gene_module_df = pd.DataFrame({
            'gene': list(gene_to_module.keys()),
            'module': list(gene_to_module.values())
        })
        gene_module_df.to_csv(os.path.join(log_dir, 'gene_module_membership.csv'), index=False)

        def compute_module_matrix(matrix, mapping):
            module_df = pd.DataFrame(index=matrix.index)
            for module in sorted(set(mapping.values())):
                genes = [gene for gene, m in mapping.items() if m == module and gene in matrix.columns]
                if genes:
                    module_df[f'Module_{module}'] = matrix[genes].mean(axis=1)
            return module_df

        # Compute module expression matrix
        train_module_matrix = compute_module_matrix(train_matrix_filtered, gene_to_module)
        test_module_matrix = compute_module_matrix(test_matrix_filtered, gene_to_module)

        return train_module_matrix, test_module_matrix, gene_to_module

    train_module_matrix, test_module_matrix, gene_to_module = cluster_genes_with_scanpy(
    train_matrix_filtered, test_matrix_filtered, cell_log_dir
)


    #################################################################

    
    # Merge the train and test matrices with their respective metadata files

    train_data = train_module_matrix.merge(
        train_metadata[['TAG', 'msex', 'sample', 'broad.cell.type', 'alzheimers_or_control', 'age_death', 'educ', 'cts_mmse30_lv', 'pmi'] + apoe_genotype_columns],
        left_index=True,
        right_on='TAG',
        how='inner'
    ).set_index('TAG')
    
    test_data = test_module_matrix.merge(
        test_metadata[['TAG', 'msex', 'sample', 'broad.cell.type', 'alzheimers_or_control', 'age_death', 'educ', 'cts_mmse30_lv', 'pmi'] + apoe_genotype_columns],
        left_index=True,
        right_on='TAG',
        how='inner'
    ).set_index('TAG')
    
        # Clean column names for model compatibility
    train_data.columns = train_data.columns.str.replace(r'[^A-Za-z0-9_]+', '', regex=True)
    test_data.columns = test_data.columns.str.replace(r'[^A-Za-z0-9_]+', '', regex=True)
    
    # # Ensure common modules are used between training and testing sets
    common_modules = train_data.columns.intersection(test_data.columns)

    # Drop the alzheimers or control column from the dataset
    X_train = train_data[common_modules].drop(columns=['alzheimers_or_control'])
    X_test = test_data[common_modules].drop(columns=['alzheimers_or_control'])

    
    # We no longer need to map back to gene names — module names are already clean
    
    # Define the target variable
    y_train = train_data['alzheimers_or_control']
    y_test = test_data['alzheimers_or_control']

    print("Printing dimensionality of X_train and X_test post filtering and merging")

    print(X_train.shape)
    print(X_test.shape)


    #########################################################################



    # Convert age of death variable to float
    X_train.loc[X_train.age_death == '90+', 'age_death'] = 90
    X_test.loc[X_test.age_death == '90+', 'age_death'] = 90
    X_train.age_death = X_train.age_death.astype(float)
    X_test.age_death = X_test.age_death.astype(float)


    
    



    # Dropping columns from the dataset
    cols_to_drop = ['sample', 'cts_mmse30_lv', 'pmi']
                    
                    #, 'msex', 'broad.cell.type', 'alzheimers_or_control', 'age_death', 'educ'] + apoe_genotype_columns
    X_train = X_train.drop(columns=cols_to_drop, errors='ignore')
    X_test = X_test.drop(columns=cols_to_drop, errors='ignore')

    class_weight_ratio = (len(y_train) / (2 * np.bincount(y_train)))  # inverse frequency
    sample_weight = np.array([class_weight_ratio[label] for label in y_train])


    with open(LOG_FILE_PATH, 'a') as log_file:
        log_file.write(f"Reached module_gene classifier for: {cell_type}\n")



    maximal_classifier = AutoML()

    automl_settings = {
        "X_train": X_train,
        "y_train": y_train,
        "sample_weight": sample_weight,
        "task": "classification",
        "time_budget": 5400,
        "metric": 'log_loss',
        "n_jobs": -1,
        "eval_method": 'cv',
        "split_type": 'group',
        "groups": train_metadata['sample'],
        "log_training_metric": True,
        "early_stop": True,
        "seed": 234567,
        "estimator_list": ['lgbm'],
        "model_history": True,
        "log_file_name": f"{cell_log_dir}/all_features_log.txt"
    }


    # Fit
    maximal_classifier.fit(**automl_settings)




    # Save the full model using joblib

    joblib.dump(maximal_classifier, f'{cell_log_dir}/maximal_classifier.joblib')


    # Predictions and optimal threshold using F1 Precision-Recall Tradeoff Statistic
    y_prob_train = maximal_classifier.predict_proba(X_train)[:, 1]
    y_prob_test = maximal_classifier.predict_proba(X_test)[:, 1]

    from sklearn.metrics import precision_recall_curve

    # Get precision-recall curve and thresholds
    precision, recall, thresholds = precision_recall_curve(y_train, y_prob_train)

    # Avoid divide-by-zero
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

    # Best threshold is the one with max F1
    optimal_index = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_index]

    print(f"Optimal threshold from Precision-Recall curve: {optimal_threshold}")
    
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
        'test_mcc': matthews_corrcoef(y_test, y_pred_test_optimal),
        'optimal_threshold': optimal_threshold
    }

    pd.DataFrame([metrics]).to_csv(f'{cell_log_dir}/output_csv.csv', index=False)

        # Create a DataFrame to store probabilities and classifications
    train_predictions_df = pd.DataFrame({
        'TAG': X_train.index,
        'true_label': y_train.values,
        'predicted_label': y_pred_train_optimal,
        'predicted_proba': y_prob_train
    })

    test_predictions_df = pd.DataFrame({
        'TAG': X_test.index,
        'true_label': y_test.values,
        'predicted_label': y_pred_test_optimal,
        'predicted_proba': y_prob_test
    })

    # Define classification categories
    train_predictions_df['classification_category'] = np.select(
        [
            (train_predictions_df['true_label'] == 1) & (train_predictions_df['predicted_label'] == 1),  # True Positive
            (train_predictions_df['true_label'] == 1) & (train_predictions_df['predicted_label'] == 0),  # False Negative
            (train_predictions_df['true_label'] == 0) & (train_predictions_df['predicted_label'] == 0),  # True Negative
            (train_predictions_df['true_label'] == 0) & (train_predictions_df['predicted_label'] == 1)   # False Positive
        ],
        ['TP', 'FN', 'TN', 'FP'],
        default='Unknown'
    )

    test_predictions_df['classification_category'] = np.select(
        [
            (test_predictions_df['true_label'] == 1) & (test_predictions_df['predicted_label'] == 1),
            (test_predictions_df['true_label'] == 1) & (test_predictions_df['predicted_label'] == 0),
            (test_predictions_df['true_label'] == 0) & (test_predictions_df['predicted_label'] == 0),
            (test_predictions_df['true_label'] == 0) & (test_predictions_df['predicted_label'] == 1)
        ],
        ['TP', 'FN', 'TN', 'FP'],
        default='Unknown'
    )

    # Save predictions to CSV files
    train_predictions_df.to_csv(f'{cell_log_dir}/train_predictions.csv', index=False)
    test_predictions_df.to_csv(f'{cell_log_dir}/test_predictions.csv', index=False)

    print("Prediction probabilities for full classifier saved successfully.")




    # Feature importance for top 100 features and avoid mismatch error
    print("Starting incremental retraining using top gene modules")
    
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

    # Save top module names
    pd.DataFrame({'top_modules': top_features_cleaned}).to_csv(f"{cell_log_dir}/top_modules.csv", index=False)

    # --- Start Incremental Evaluation ---
    incremental_results = []

    for i, feature_subset in enumerate(top_features_cleaned[:25], start=1):
        print(f"Retraining model from scratch with top {i} features")
        current_features = top_features_cleaned[:i]
        
        # Subset data
        X_train_top_i = X_train[current_features]
        X_test_top_i = X_test[current_features]

        # Define settings dictionary
        automl_settings_2 = {
            "X_train": X_train_top_i,
            "y_train": y_train,
            "sample_weight": sample_weight,
            "task": "classification",
            "time_budget": 600,
            "metric": 'log_loss',
            "n_jobs": -1,
            "eval_method": 'cv',
            "split_type": 'group',
            "groups": train_metadata['sample'],
            "log_training_metric": True,
            "early_stop": True,
            "seed": 234567,
            "estimator_list": ['lgbm'],
            "model_history": True,
            "log_file_name": f"{cell_log_dir}/top_{i}_features_log.txt",
        }

        # Retrain from scratch
        incremental_classifier = AutoML()
        incremental_classifier.fit(**automl_settings_2)

        joblib.dump(incremental_classifier, f"{cell_log_dir}/top_{i}_features_classifier.joblib")

        with open(LOG_FILE_PATH, 'a') as log_file:
            log_file.write(f"Incremental classifier finsihed for: {cell_type}\n")

        # Predict probabilities
        y_prob_train_i = incremental_classifier.predict_proba(X_train_top_i)[:, 1]
        y_prob_test_i = incremental_classifier.predict_proba(X_test_top_i)[:, 1]

        # Dynamically calculate best threshold based on train set
        precision, recall, thresholds = precision_recall_curve(y_train, y_prob_train_i)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_index = np.argmax(f1_scores)
        dynamic_threshold = thresholds[optimal_index]

        y_pred_train_i = (y_prob_train_i >= dynamic_threshold).astype(int)
        y_pred_test_i = (y_prob_test_i >= dynamic_threshold).astype(int)

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
            'test_mcc': matthews_corrcoef(y_test, y_pred_test_i),
            'optimal_threshold': optimal_threshold
        }
        incremental_results.append(result)

        # Save predictions
        train_predictions_i = pd.DataFrame({
            'TAG': X_train_top_i.index,
            'true_label': y_train.values,
            'predicted_label': y_pred_train_i,
            'predicted_proba': y_prob_train_i
        })
        train_predictions_i.to_csv(f"{cell_log_dir}/train_predictions_top_{i}_features.csv", index=False)

        test_predictions_i = pd.DataFrame({
            'TAG': X_test_top_i.index,
            'true_label': y_test.values,
            'predicted_label': y_pred_test_i,
            'predicted_proba': y_prob_test_i
        })
        test_predictions_i.to_csv(f"{cell_log_dir}/test_predictions_top_{i}_features.csv", index=False)

    # Save results
    incremental_results_df = pd.DataFrame(incremental_results)
    incremental_results_df.to_csv(f'{cell_log_dir}/incremental_top_features_metrics.csv', index=False)
    print("Incremental evaluation completed successfully")

    # Plot Test ROC AUC vs Number of Features
    plt.figure(figsize=(8, 6))
    plt.plot(incremental_results_df['num_features'], incremental_results_df['test_roc_auc'], marker='o')
    plt.title('Test ROC AUC vs Number of Top Features')
    plt.xlabel('Number of Top Features')
    plt.ylabel('Test ROC AUC')
    plt.grid(True)
    plt.tight_layout()

    # Saving plot
    plot_path = os.path.join(cell_log_dir, 'test_auc_vs_num_features.png')
    plt.savefig(plot_path)
    plt.close()

    print(f"Saved AUC vs. feature count plot to: {plot_path}")

    with open(LOG_FILE_PATH, 'a') as log_file:
            log_file.write(f"Finished demographics and genes without pmi for cell on cell:{cell_type}\n")

    


    print("Maximal experiment completed")


if __name__ == "__main__":
    main()
