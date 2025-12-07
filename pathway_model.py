from pathlib import Path
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
from ete3 import NCBITaxa
from utils import build_species_to_ncbi_map, load_gtdb_metadata

NUM_SAMPLES = 1000
RATIO = 0.1

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

def get_level_taxid(strain_taxid, level):
    """
    Given an NCBI TaxID,
    return the corresponding level TaxID.
    """
    ncbi = NCBITaxa()
    
    # Get lineage of the taxon
    try:
        lineage = ncbi.get_lineage(strain_taxid)
    except:
        raise ValueError(f"Invalid TaxID: {strain_taxid}")
    
    # Get ranks of lineage nodes
    ranks = ncbi.get_rank(lineage)  # {taxid: rank}
    
    # Find the species-level taxid
    for taxid in lineage:
        if ranks.get(taxid) == level:
            return taxid
    
    # No species found in lineage
    return None


if __name__ == "__main__":
    metadata_df, microbiome_df = read_data(".\\train")
    distance_matrix = pd.read_csv(".\\jaccard_distance_matrix.csv", index_col=0)

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
        aupr = -np.trapz(precision, recall)
        naive_aupr_scores.append(aupr)

    print(f"Average of average precision scores without clustering: {np.mean(naive_aupr_scores):.4f}")

    phylo_aupr_best_score = 0
    phylo_aupr_best_lvl = 0
    phylo_aupr_scores = []
    levels = ["phylum", "class", "order", "family", "genus", "species"]
    metadata_path = "bac120_metadata_r207.tsv"
    metadata_df = load_gtdb_metadata(metadata_path)
    sp_to_taxid = build_species_to_ncbi_map(metadata_df)
    for lvl in range(len(levels)):
        level = levels[lvl]
        level_microbiome_df = rel_microbiome_df.copy()
        groups = {}
        for col in level_microbiome_df.columns:
            if col == "disease_status":
                continue
            taxid = sp_to_taxid.get("s__" + col)
            lowest_level = lvl
            while get_level_taxid(taxid, level) is None and lowest_level > 0:
                lowest_level -= 1
            level_taxid = get_level_taxid(taxid, levels[lowest_level])
            if level_taxid not in groups:
                groups[level_taxid] = [col]
            else:
                groups[level_taxid].append(col)
        
        for taxaid, cols in groups.items():
            if len(cols) > 1:
                level_microbiome_df[f"{level}_{taxaid}"] = level_microbiome_df[cols].sum(axis=1)
                level_microbiome_df = level_microbiome_df.drop(columns=cols)
        
        level_aupr_scores = []
        for seed in seeds:
            train, test = train_test_split(level_microbiome_df, train_size=TRAIN_PART, random_state=seed)

            test = sample_data_balanced(test[test["disease_status"] == 1],
                                    test[test["disease_status"] == 0], RATIO, NUM_SAMPLES, random_state=seed)
            
            level_model = train_model(train, "disease_status", random_state=seed)

            X_test = test.drop(columns=["disease_status"])
            y_test = test["disease_status"]

            test_probs = level_model.predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, test_probs)
            aupr = -np.trapz(precision, recall)
            level_aupr_scores.append(aupr)
        
        if np.mean(level_aupr_scores) > phylo_aupr_best_score:
            phylo_aupr_best_score = np.mean(level_aupr_scores)
            phylo_aupr_best_lvl = lvl
        phylo_aupr_scores.append(np.mean(level_aupr_scores))
        
    plt.figure(figsize=(20, 12))
    plt.plot(levels, phylo_aupr_scores, marker='o', linestyle='-')
    plt.title(f'Mean Average Precision vs. Clustering Threshold using phylogenetic tree\nBest level: {levels[phylo_aupr_best_lvl]}, Mean AP: {phylo_aupr_best_score:.4f}\nAverage precision scores without clustering: {np.mean(naive_aupr_scores):.4f}')
    plt.xlabel("Levels")
    plt.ylabel('Mean Average Precision Score')
    plt.grid(True)
    plt.savefig(f'misc/phylo_levels_graph.png')
    plt.close()


    tree_names = [n for n in Path(".").glob("*upgma_tree.nwk")]
    trees = [Phylo.read(str(tree_name), "newick") for tree_name in tree_names]

    best_overall_tree = -1
    best_overall_score = -1
    best_overall_threshold = -1
    best_overall_seed = -1

    for i in range(len(trees)):
        tree = trees[i]
        length_thresholds = np.linspace(0, root_to_leaf_length(tree.clade), num=100)
        engineered_aupr_scores = []
        max_mean_score = 0.0
        best_threshold = 0.0
        for threshold in length_thresholds:
            microbiome_model_data = rel_microbiome_df.copy()
            groups = clades_below_threshold(tree.clade, threshold)
            
            for j, group in enumerate(groups):
                cols_to_sum = [col for col in microbiome_model_data.columns if col in group]
                if len(cols_to_sum) > 1:
                    microbiome_model_data[f"cluster_{j}"] = microbiome_model_data[cols_to_sum].sum(axis=1)
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
                aupr = -np.trapz(precision, recall)
                threshold_aupr_scores.append(aupr)
                if aupr > best_overall_score:
                    best_overall_score = aupr
                    best_overall_tree = i
                    best_overall_threshold = threshold
                    best_overall_seed = seed

            if np.mean(threshold_aupr_scores) > max_mean_score:
                max_mean_score = np.mean(threshold_aupr_scores)
                best_threshold = threshold

            engineered_aupr_scores.append(np.mean(threshold_aupr_scores))
        print(f"finished tree {i+1} ({tree_names[i].stem[:-11]}) out of {len(trees)}")
        plt.figure(figsize=(20, 12))
        plt.plot(length_thresholds, engineered_aupr_scores, marker='o', linestyle='-')
        plt.title(f'Mean Average Precision vs. Clustering Threshold using {tree_names[i].stem[:-11]}\nBest Threshold: {best_threshold:.4f}, Mean AP: {max_mean_score:.4f}\nAverage precision scores without clustering: {np.mean(naive_aupr_scores):.4f}\nBest average precision scores with phylo clustering: {phylo_aupr_best_score:.4f}')
        plt.xlabel("Thresholds")
        plt.ylabel('Mean Average Precision Score')
        plt.grid(True)
        plt.savefig(f'misc/{tree_names[i].stem[:-11]}_graph.png')
        plt.close()


    print(f"Best tree was {tree_names[best_overall_tree]} with threshold {best_overall_threshold} and seed {best_overall_seed}. Score: {best_overall_score}")
    microbiome_model_data = rel_microbiome_df.copy()
    groups = clades_below_threshold(trees[best_overall_tree].clade, best_overall_threshold)
        
    for j, group in enumerate(groups):
        cols_to_sum = [col for col in microbiome_model_data.columns if col in group]
        if len(cols_to_sum) > 1:
            microbiome_model_data[f"cluster_{j}"] = microbiome_model_data[cols_to_sum].sum(axis=1)
            microbiome_model_data = microbiome_model_data.drop(columns=cols_to_sum)

    best_model_tm = train_model(microbiome_model_data, "disease_status", random_state=best_overall_seed)

    test_df = pd.read_csv(".\\test\\microbiome.csv").set_index("SampleID")
    test_df = counts_to_relative_abundance(test_df[distance_matrix.columns.tolist()])

    for j, group in enumerate(groups):
        cols_to_sum = [col for col in test_df.columns if col in group]
        if len(cols_to_sum) > 1:
            test_df[f"cluster_{j}"] = test_df[cols_to_sum].sum(axis=1)
            test_df = test_df.drop(columns=cols_to_sum)

    test_df["predictions"] = best_model_tm.predict_proba(test_df)[:, 1]

    test_df.to_csv("test_predictions.csv")


        

