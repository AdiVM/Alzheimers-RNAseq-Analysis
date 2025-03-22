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

log_dir_path = "/n/groups/patel/adithya/Alz_Outputs/F1_Demographics_Only/"
LOG_FILE_PATH = os.path.expanduser(f'{log_dir_path}experiment_log.txt')


def main():
    parser = argparse.ArgumentParser(description='Run AutoML on combined gene expression and metadata data')
    parser.add_argument('--exp_type', type=str, choices=['maximal'], required=True, help='Specify experiment type')
    args = parser.parse_args()
    
    exp_type = args.exp_type


    log_message = f"Processing {exp_type} data using full metadata features"
    with open(LOG_FILE_PATH, 'a') as log_file:
        log_file.write(log_message + '\n')

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
        random_state=42,
        stratify=sample_summary['stratify_group']
    )

    # Filter metadata by train and test `sample_id`
    train_metadata = metadata[metadata['sample'].isin(train_samples)]
    test_metadata = metadata[metadata['sample'].isin(test_samples)]

    # Drop duplicate samples to retain one row per person
    train_metadata = train_metadata.drop_duplicates(subset='sample')
    test_metadata = test_metadata.drop_duplicates(subset='sample')

    print(f"Number of cases in training: {sum(train_metadata['alzheimers_or_control'])}")
    print(f"Number of cases in test: {sum(test_metadata['alzheimers_or_control'])}")
    
    
    # Choose only demographic features
    demographic_features = ['msex', 'sample', 'age_death', 'educ', 'cts_mmse30_lv', 'pmi'] + apoe_genotype_columns

    # Define the target variable
    y_train = train_metadata['alzheimers_or_control']
    y_test = test_metadata['alzheimers_or_control']

    X_train = train_metadata[demographic_features].copy()
    X_test = test_metadata[demographic_features].copy()


    print("Printing dimensionality of X_train and X_test post filtering and merging")

    print(X_train.shape)
    print(X_test.shape)


    #########################################################################
    # Trying a new method of creating cross validation folds:
    from sklearn.model_selection import StratifiedGroupKFold
    import numpy as np

    def generate_valid_folds(X, y, groups, n_splits=5, max_retries=100):
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
    
    # Generate valid folds
    # valid_folds = generate_valid_folds(
    #     X_train,  # Feature matrix
    #     y_train,  # Target variable
    #     groups=train_metadata['sample'],  # Group variable
    #     n_splits=10,
    #     max_retries=100
    # )

    cell_log_dir = os.path.join(log_dir_path, "demographics_model")

    # Create the directory if it doesn’t exist
    os.makedirs(cell_log_dir, exist_ok=True)

    # Dropping samples from the dataset
    X_train = X_train.drop(columns=['sample'])
    X_test = X_test.drop(columns=['sample'])

    class_weight_ratio = (len(y_train) / (2 * np.bincount(y_train.astype(int))))  # inverse frequency
    sample_weight = np.array([class_weight_ratio[label] for label in y_train])


    # Use valid folds in AutoML
    maximal_classifier = AutoML()
    maximal_classifier.fit(
        X_train, y_train,
        sample_weight=sample_weight,
        task="classification",
        time_budget=3600,
        metric='log_loss',
        n_jobs=-1,
        eval_method='cv',
        split_type='group',
        groups=train_metadata['sample'],
        log_training_metric=True,
        early_stop=True,
        seed=234567,
        estimator_list=['lgbm'],
        model_history=True,
        log_file_name=f"{cell_log_dir}/all_features_log.txt"
    )




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
    

    # Simple identity column mapping (no renaming needed for demographics model)
    column_mapping = {col: col for col in X_train.columns}

    # Map features back to original names for interpretability
    top_features_original = [column_mapping.get(feature, feature) for feature in top_features_cleaned]

    # --- Start Incremental Evaluation ---
    incremental_results = []

    for i in range(1, min(8, len(top_features_cleaned)+1)):
        current_features = top_features_cleaned[:i]
        print(f"Retraining model from scratch with top {i} features")
        
        # Subset data
        X_train_top_i = X_train[current_features]
        X_test_top_i = X_test[current_features]
        
        # Retrain from scratch
        incremental_classifier = AutoML()
        incremental_classifier.fit(
            X_train_top_i,
            y_train,
            sample_weight=sample_weight,
            task="classification",
            time_budget=600,  # Shorter budget set for smaller feature set
            metric='log_loss',
            n_jobs=-1,
            eval_method='cv',
            split_type='group',
            groups=train_metadata['sample'],
            log_training_metric=True,
            early_stop=True,
            seed=234567,
            estimator_list=['lgbm'],
            model_history=True,
            log_file_name=f"{cell_log_dir}/top_{i}_features_log.txt"
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

    print("Prediction probabilities saved successfully.")

    


    print("Maximal experiment completed")


if __name__ == "__main__":
    main()