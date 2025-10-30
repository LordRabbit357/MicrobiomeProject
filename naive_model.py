import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt


def read_data(file_path):
    """Reads CSV data from the given file path."""
    return pd.read_csv(file_path + "\\metadata.csv").set_index("SampleID"), pd.read_csv(file_path + "\\microbiome.csv").set_index("SampleID"), pd.read_csv(file_path + "\\serum_lipo.csv").set_index("SampleID")

def preprocess_data(metadata_df, microbiome_df):
    """Preprocesses the data by handling missing values and normalizing."""
    # Example preprocessing steps
    ret_metadata = metadata_df.fillna(-1)  # Fill missing values with -1
    ret_metadata["disease_status"] = np.where(ret_metadata["PATGROUPFINAL_C"] == "8", 0, 1)  # Binary encoding of disease status
    ret_microbiome_df = microbiome_df.div(microbiome_df.sum(axis=1), axis=0)  # Convert to relative abundances
    return ret_metadata, ret_microbiome_df

def train_model(train_df, target_column):
    """Trains a Random Forest model."""
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]

    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, max_depth=5)
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, test_df, target_column):
    """Evaluates the model and plots the Precision-Recall curve."""
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    test_probs = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, test_probs)
    ap = average_precision_score(y_test, test_probs)
    aupr = -np.trapezoid(precision, recall)

    plt.figure()
    plt.plot(recall, precision, label=f'AUPR={aupr:.4f}\nAP={ap:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    metadata_df, microbiome_df, serum_lipo_df = read_data(".\\train")
    filled_metadata_df, normalized_microbiome_df = preprocess_data(metadata_df, microbiome_df)

    target_column = "disease_status"
    microbiome_model_data = microbiome_df.join(filled_metadata_df[[target_column]], how="inner")
    normalized_microbiome_model_data = normalized_microbiome_df.join(filled_metadata_df[[target_column]], how="inner")
    metabolome_added_model_data = normalized_microbiome_model_data.join(serum_lipo_df, how="inner")

    microbiome_train, microbiome_test = train_test_split(microbiome_model_data, test_size=0.2, random_state=42, stratify=microbiome_model_data[target_column])
    normalized_microbiome_train, normalized_microbiome_test = train_test_split(normalized_microbiome_model_data, test_size=0.2, random_state=42, stratify=normalized_microbiome_model_data[target_column])
    metabolome_added_train, metabolome_added_test = train_test_split(metabolome_added_model_data, test_size=0.2, random_state=42, stratify=metabolome_added_model_data[target_column])

    microbiome_model = train_model(microbiome_train, target_column)
    normalized_microbiome_model = train_model(normalized_microbiome_train, target_column)
    metabolome_added_model = train_model(metabolome_added_train, target_column)

    evaluate_model(microbiome_model, microbiome_test, target_column)
    evaluate_model(normalized_microbiome_model, normalized_microbiome_test, target_column)
    evaluate_model(metabolome_added_model, metabolome_added_test, target_column)
