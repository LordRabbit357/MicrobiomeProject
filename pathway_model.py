import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from Bio import Phylo
import math
import random

NUM_SAMPLES = 500
RATIO = 0.5

TRAIN_PART = 0.8
TEST_PART = 0.2



def read_data(file_path):
    """Reads CSV data from the given file path."""
    return pd.read_csv(file_path + "\\metadata.csv").set_index("SampleID"), pd.read_csv(file_path + "\\microbiome.csv").set_index("SampleID")

def counts_to_relative_abundance(df):
    """Converts count data to relative abundances."""
    return df.div(df.sum(axis=1), axis=0)

def extract_relevant_metadata(metadata_df):
    """Extracts relevant metadata for disease status."""
    ret_metadata = metadata_df.copy()
    ret_metadata["disease_status"] = np.where(ret_metadata["PATGROUPFINAL_C"] == "8", 0, 1)  # Binary encoding of disease status
    
    return ret_metadata["disease_status"]

def root_to_leaf_length(tree):
    """Calculates the root-to-leaf length for a given tree."""
    length = 0.0
    clade = tree
    while not clade.is_terminal():
        if clade.branch_length:
            length += clade.branch_length
        clade = clade.clades[0]
    if clade.branch_length:
        length += clade.branch_length
    return length

def clades_below_threshold(tree, threshold):
    """
    Return a list of leaf-name groups.
    Each group corresponds to a clade where 
    the distance from the clade root to its leaves is <= threshold.
    """

    if tree.is_terminal():
        return [[tree.name]]
    
    if root_to_leaf_length(tree) <= threshold:
        return [[leaf.name for leaf in tree.get_terminals()]]

    qualifying_groups = []
    child1, child2 = tree.clades
    qualifying_groups.extend(clades_below_threshold(child1, threshold))
    qualifying_groups.extend(clades_below_threshold(child2, threshold))

    return qualifying_groups

def sample_data_balanced(df_x, df_y, xy_ratio, n_samples, random_state=42):
    """Samples the data to balance classes."""

    x_sampled = df_x.sample(n=math.ceil(xy_ratio * n_samples), replace=True, random_state=random_state)
    y_sampled = df_y.sample(n=math.ceil((1 - xy_ratio) * n_samples), replace=True, random_state=random_state)

    return pd.concat([x_sampled, y_sampled], axis=0)

def train_model(train_df, target_column, random_state=42):
    """Trains a Random Forest model."""
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]

    model = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1, max_depth=5)
    model.fit(X_train, y_train)
    
    return model


if __name__ == "__main__":
    metadata_df, microbiome_df = read_data(".\\train")
    distance_matrix = pd.read_csv(".\\species_distance_matrix.csv", index_col=0)
    tree = Phylo.read(".\\species_upgma_tree.nwk", "newick")

    seeds = random.sample(range(10000), 10)

    rel_microbiome_df = counts_to_relative_abundance(microbiome_df[distance_matrix.columns.tolist()])
    rel_microbiome_df["disease_status"] = extract_relevant_metadata(metadata_df)

    naive_aupr_scores = []
    for seed in seeds:
        train, test = train_test_split(rel_microbiome_df, train_size=TRAIN_PART, random_state=seed)

        test = sample_data_balanced(test[test["disease_status"] == 1],
                                test[test["disease_status"] == 0], RATIO, NUM_SAMPLES, random_state=seed)
        
        naive_model = train_model(train, "disease_status", random_state=seed)

        X_test = test.drop(columns=["disease_status"])
        y_test = test["disease_status"]

        test_probs = naive_model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, test_probs)
        aupr = -np.trapezoid(precision, recall)
        naive_aupr_scores.append(aupr)

    print(f"Average of average precision scores without clustering: {np.mean(naive_aupr_scores):.4f}")

    length_thresholds = np.linspace(0, root_to_leaf_length(tree.clade), num=100)
    max_mean_score = 0.0
    best_threshold = 0.0


    engineered_aupr_scores = []


    for threshold in length_thresholds:
        microbiome_model_data = rel_microbiome_df.copy()
        groups = clades_below_threshold(tree.clade, threshold)
        
        for i, group in enumerate(groups):
            cols_to_sum = [col for col in microbiome_model_data.columns if col in group]
            if len(cols_to_sum) > 1:
                microbiome_model_data[f"cluster_{i}"] = microbiome_model_data[cols_to_sum].sum(axis=1)
                microbiome_model_data = microbiome_model_data.drop(columns=cols_to_sum)

        threshold_aupr_scores = []

        for seed in seeds:
            train, test = train_test_split(microbiome_model_data, train_size=TRAIN_PART, random_state=seed)

            test = sample_data_balanced(test[test["disease_status"] == 1],
                                    test[test["disease_status"] == 0], RATIO, NUM_SAMPLES, random_state=seed)
            
            engineered_model = train_model(train, "disease_status", random_state=seed)

            X_test = test.drop(columns=["disease_status"])
            y_test = test["disease_status"]

            test_probs = engineered_model.predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, test_probs)
            aupr = -np.trapezoid(precision, recall)
            threshold_aupr_scores.append(aupr)

        if np.mean(threshold_aupr_scores) > max_mean_score:
            max_mean_score = np.mean(threshold_aupr_scores)
            best_threshold = threshold

        engineered_aupr_scores.append(np.mean(threshold_aupr_scores))
    plt.figure(figsize=(10, 6))
    plt.plot(length_thresholds, engineered_aupr_scores, marker='o', linestyle='-')
    plt.title(f'Mean Average Precision vs. Clustering Threshold\nBest Threshold: {best_threshold:.4f}, Mean AP: {max_mean_score:.4f}\nAverage of average precision scores without clustering: {np.mean(naive_aupr_scores):.4f}')
    plt.xlabel("Thresholds")
    plt.ylabel('Mean Average Precision Score')
    plt.grid(True)
    plt.show()