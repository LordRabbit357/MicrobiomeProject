import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from Bio import Phylo
import math



def read_data(file_path):
    """Reads CSV data from the given file path."""
    return pd.read_csv(file_path + "\\metadata.csv").set_index("SampleID"), pd.read_csv(file_path + "\\microbiome.csv").set_index("SampleID")

def counts_to_relative_abundance(df):
    """Converts count data to relative abundances."""
    return df.div(df.sum(axis=1), axis=0)

def extract_relevant_metadata(metadata_df):
    """Extracts relevant metadata for disease status."""
    ret_metadata = metadata_df.fillna(-1)  # Fill missing values with -1
    ret_metadata["disease_status"] = np.where(ret_metadata["PATGROUPFINAL_C"] == "8", 0, 1)  # Binary encoding of disease status

    nhot_center = ret_metadata["CENTER_C"].str.get_dummies()
    ret_metadata = pd.concat([ret_metadata, nhot_center], axis=1)
    
    return ret_metadata.drop(columns=["CENTER_C", "PATGROUPFINAL_C", "SMOKE", "pa_work_2cl", "DDS"])

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


if __name__ == "__main__":
    metadata_df, microbiome_df = read_data(".\\train")
    distance_matrix = pd.read_csv(".\\species_distance_matrix.csv", index_col=0)
    tree = Phylo.read(".\\species_upgma_tree.nwk", "newick")

    rel_microbiome_df = counts_to_relative_abundance(microbiome_df)
    relevant_metadata_df = extract_relevant_metadata(metadata_df)

    length_thresholds = np.linspace(0, root_to_leaf_length(tree.clade), num=100)
    max_mean_score = 0.0
    best_threshold = 0.0

    microbiome_model_data = rel_microbiome_df.join(relevant_metadata_df, how="inner")

    microbiome_model_data = sample_data_balanced(microbiome_model_data[microbiome_model_data["disease_status"] == 0],
                                                    microbiome_model_data[microbiome_model_data["disease_status"] == 1],
                                                    xy_ratio=0.5, n_samples=500, random_state=42)
    
    microbiome_naive_data = microbiome_model_data.copy()

    X = microbiome_naive_data.drop(columns=["disease_status"])
    y = microbiome_naive_data["disease_status"]

    rf_naive = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, max_depth=5)

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    scores_model = cross_val_score(
        rf_naive, X, y, cv=kf, scoring="average_precision", n_jobs=-1
    )
    print(f"Average of average precision scores without clustering: {scores_model.mean():.4f}")

    for threshold in length_thresholds:
        groups = clades_below_threshold(tree.clade, threshold)
        
        for i, group in enumerate(groups):
            cols_to_sum = [col for col in microbiome_model_data.columns if col in group]
            if len(cols_to_sum) > 1:
                microbiome_model_data[f"cluster_{i}"] = microbiome_model_data[cols_to_sum].sum(axis=1)
                microbiome_model_data = microbiome_model_data.drop(columns=cols_to_sum)
        

        X = microbiome_model_data.drop(columns=["disease_status"])
        y = microbiome_model_data["disease_status"]

        rf_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, max_depth=5)

        kf = KFold(n_splits=10, shuffle=True, random_state=42)

        scores_model = cross_val_score(
            rf_model, X, y, cv=kf, scoring="average_precision", n_jobs=-1
        )

        if scores_model.mean() > max_mean_score:
            max_mean_score = scores_model.mean()
            best_threshold = threshold

        print(f"Average of average precision scores for threshold {threshold:.4f}: {scores_model.mean():.4f}")

    print(f"Best threshold: {best_threshold:.4f} with mean average precision: {max_mean_score:.4f}")