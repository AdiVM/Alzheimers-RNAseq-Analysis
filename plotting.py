import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Using merge to create full prediciton df
def add_subcluster(pred_df, meta_df):
    df = pred_df.merge(meta_df[['TAG', 'Subcluster', 'broad.cell.type']], on='TAG', how='left')

    return df

# Using merge to create misclassified df
def create_misclassified(pred_df, meta_df):
    misclassified_df = pred_df[
        ((pred_df['true_label'] == 1) & (pred_df['predicted_label'] == 0)) |  # FN
        ((pred_df['true_label'] == 0) & (pred_df['predicted_label'] == 1))    # FP
    ]

    misclassified_merged_df = misclassified_df.merge(meta_df[['TAG']], on='TAG', how='left')

    return misclassified_merged_df

# Total cells per cluster
def total_cells_per_cluster(df):
    # Order subclusters by descending total count
    subcluster_order = df['Subcluster'].value_counts().sort_values(ascending=False).index

    plt.figure(figsize=(10, 6))
    df['Subcluster'].value_counts().loc[subcluster_order].plot(kind='bar')
    plt.title("Number of Total Cells per Subcluster")
    plt.xlabel("Subcluster")
    plt.ylabel("Cell Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def misclassification_rate_per_cluster(df):
    # Identify misclassified rows
    df['misclassified'] = df['true_label'] != df['predicted_label']
    
    # Count total and misclassified cells per subcluster
    total_per_cluster = df['Subcluster'].value_counts()
    misclassified_per_cluster = df[df['misclassified']]['Subcluster'].value_counts()
    
    # Compute misclassification percentage per subcluster
    misclassification_rate = (misclassified_per_cluster / total_per_cluster) * 100
    misclassification_rate = misclassification_rate.fillna(0)

    # Order subclusters by descending total count
    subcluster_order = total_per_cluster.sort_values(ascending=False).index
    misclassification_rate = misclassification_rate.reindex(subcluster_order)

    # Plot
    plt.figure(figsize=(10, 6))
    misclassification_rate.plot(kind='bar')
    plt.xlabel('Subcluster')
    plt.ylabel('Percentage of Misclassified Cells')
    cell_type = df['broad.cell.type'].iloc[0]
    plt.title(f'Misclassification Rate per Subcluster ({cell_type})')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Misclassificaiton rate
def misclass_rate(pred_df):
    # Total number of predictions
    total_cells = len(pred_df)
    print(f"Total number of cells: {total_cells}")

    # Number of misclassified predictions
    num_misclassified = (pred_df['true_label'] != pred_df['predicted_label']).sum()

    # Percentage of misclassified cells
    percent_misclassified = (num_misclassified / total_cells) * 100

    print(f"Total misclassified cells: {num_misclassified}")
    print(f"Percentage of misclassified cells: {percent_misclassified:.2f}%")

# The

# Histogram that plots probabilities for each cell

def plot_proba_histo(pred_df):
    # Separate predictions
    ad = pred_df[pred_df['true_label'] == 1]['predicted_proba']
    ctrl = pred_df[pred_df['true_label'] == 0]['predicted_proba']

    # Combine to get fixed bin edges for easy comparison
    all_probs = np.concatenate([ad.values, ctrl.values])
    bins = np.linspace(all_probs.min(), all_probs.max(), 30)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.hist(ctrl, bins=bins, alpha=0.5, label=f'Control (n={len(ctrl)})', color='orange')
    plt.hist(ad, bins=bins, alpha=0.5, label=f'AD Case (n={len(ad)})', color='blue')

    plt.axvline(0.5, color='black', linestyle='--', linewidth=1)
    plt.legend()
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.title('Predicted Probabilities by True Label')
    plt.tight_layout()
    plt.show()



def plot_proba_category_dist(pred_df):
    # Define categories with their conditions
    categories = {
        'Correct Case': (pred_df['true_label'] == 1) & (pred_df['predicted_proba'] >= 0.5),
        'Correct Control': (pred_df['true_label'] == 0) & (pred_df['predicted_proba'] <= 0.40),
        'Incorrect Case': (pred_df['true_label'] == 1) & (pred_df['predicted_proba'] <= 0.2),
        'Incorrect Control': (pred_df['true_label'] == 0) & (pred_df['predicted_proba'] >= 0.55)
    }

    # Store counts per subcluster for each category
    subcluster_counts = {}
    for label, condition in categories.items():
        subset = pred_df[condition]
        subcluster_counts[label] = subset['Subcluster'].value_counts()

    # Get all unique subclusters
    all_clusters = sorted(set().union(*[counts.index for counts in subcluster_counts.values()]))

    # Prepare bar heights for each category
    bar_data = {
        label: [subcluster_counts[label].get(cluster, 0) for cluster in all_clusters]
        for label in categories
    }

    # Overlay: total AD and control cell counts per cluster
    total_counts = pred_df.groupby(['Subcluster', 'true_label']).size().unstack(fill_value=0)
    total_ctrl = [total_counts.loc[cluster, 0] if cluster in total_counts.index else 0 for cluster in all_clusters]
    total_ad = [total_counts.loc[cluster, 1] if cluster in total_counts.index else 0 for cluster in all_clusters]

    # Plotting
    x = range(len(all_clusters))
    bar_width = 0.2
    offsets = [-1.5, -0.5, 0.5, 1.5]

    plt.figure(figsize=(12, 6))

    # Plot the 4 confident prediction categories
    for i, (label, values) in enumerate(bar_data.items()):
        plt.bar([xi + offsets[i] * bar_width for xi in x], values, width=bar_width, label=label)

    # Overlay transparent total AD/control bars in background
    plt.bar(x, total_ctrl, width=0.9, alpha=0.15, color='orange', label='Total Control')
    plt.bar(x, total_ad, width=0.9, alpha=0.15, color='blue', label='Total AD Case')

    plt.xticks(x, all_clusters, rotation=45)
    plt.ylabel("Cell Count")
    plt.title("Subcluster Distribution of Confident Predictions (with Total Overlay)")
    plt.legend()
    plt.tight_layout()
    plt.show()

# TSNE coordinates for 4 categories, and plot them using color code for each category

# Repeat for each of 7 cell types

