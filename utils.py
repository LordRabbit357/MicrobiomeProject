import pandas as pd
import numpy as np
from ete3 import NCBITaxa
import cobra
import warnings
from pathlib import Path
import requests
from biotite.sequence.phylo import upgma

def load_gtdb_metadata(metadata_path):
    """
    Load GTDB metadata file which contains columns:
    - 'gtdb_taxonomy' : full GTDB taxonomy string
    - 'ncbi_taxid'    : corresponding NCBI taxonomic ID (can be missing)
    """
    df = pd.read_csv(metadata_path, sep="\t", low_memory=False)
    return df


def extract_gtdb_species(taxonomy_string):
    """
    Extract the s__ field (GTDB species name) from a taxonomy string.
    Example:
      'd__Bacteria;p__Firmicutes;...;s__Escherichia_coli'
    → 's__Escherichia_coli'
    """
    if not isinstance(taxonomy_string, str):
        return None

    for part in taxonomy_string.split(";"):
        if part.startswith("s__"):
            return part
    return None


def build_species_to_ncbi_map(metadata_df):
    """
    Build a dictionary: { 's__Genus_species' → ncbi_taxid }
    """
    metadata_df["gtdb_species"] = metadata_df["gtdb_taxonomy"].apply(extract_gtdb_species)
    mapping = (
        metadata_df.dropna(subset=["gtdb_species", "ncbi_taxid"])
        .set_index("gtdb_species")["ncbi_taxid"]
        .to_dict()
    )
    return mapping


def convert_species_list(species_list, species_to_ncbi):
    """
    Convert a list of GTDB species names to NCBI TaxIDs.
    Returns a dict: {species_name: ncbi_taxid or None}
    """
    result = {}
    for sp in species_list:
        key = sp if sp.startswith("s__") else f"s__{sp}"
        result[sp] = get_species_taxid(species_to_ncbi.get(key, None)) if key in species_to_ncbi else None
    return result


def get_species_taxid(strain_taxid):
    """
    Given an NCBI TaxID for a strain/subspecies,
    return the corresponding species-level TaxID.
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
        if ranks.get(taxid) == "species":
            return taxid
    
    # No species found in lineage
    return None

def download_model(filename, path_to_models):
    """
    Download an SBML model from the Agora repository.
    """
    Path(path_to_models).mkdir(parents=True, exist_ok=True)
    file_url = f"https://www.vmh.life/files/reconstructions/AGORA/1.03/reconstructions/sbml/{filename}.xml"
    try:
        # Send a GET request to the URL
        response = requests.get(file_url, stream=True)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)

        # Open the local file in binary write mode
        with open(f"{path_to_models}{filename}.xml", "wb") as f:
            # Iterate over the response content in chunks to handle large files efficiently
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"File '{filename}.xml' downloaded successfully.")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        return False
    return True

def get_strain_reactinos(strain_name, path_to_models="models/"):
    """
    Load an SBML model for a given strain name and return its reactions.
    Assumes models are stored in SBML format in the specified directory.
    """
    strain_filename = strain_name.strip().replace(' ', '_').replace('.', '').replace('-', '_').replace(',', '_').replace('/', '_').replace('(', '').replace(')', '').replace(":", "_").replace("__", "_")
    if "=" in strain_filename:
        strain_filename = strain_filename.split('=')[0]
        strain_filename = strain_filename.rstrip('_')
    model_path = f"{path_to_models}{strain_filename}.xml"
    path_object = Path(model_path)

    if not path_object.is_file():
        if not download_model(strain_filename, path_to_models):
            print(f"Failed to download model for strain: {strain_name}")
            return None

    try:
        model = cobra.io.read_sbml_model(model_path)
        return model.reactions
    except FileNotFoundError:
        print(f"Model file not found for strain: {strain_name}")
        return None
    
def jaccard_index(set1, set2):
    """
    Compute the Jaccard index between two sets.
    J(A, B) = |A ∩ B| / |A ∪ B|
    """
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    if union == 0:
        return 0.0
    return intersection / union


if __name__ == "__main__":
    # Ignore all UserWarnings
    warnings.filterwarnings('ignore', category=UserWarning)
    
    metadata_path = "bac120_metadata_r207.tsv"

    gtdb_species_list = pd.read_csv("train/microbiome.csv").set_index("SampleID").columns.tolist()

    # Load metadata and build mapping
    metadata_df = load_gtdb_metadata(metadata_path)
    species_to_ncbi = build_species_to_ncbi_map(metadata_df)

    # Convert GTDB → NCBI
    converted = convert_species_list(gtdb_species_list, species_to_ncbi)

    print(f"Number of GTDB species converted: {len(converted)}")

    agora_organisms = pd.read_csv("vmh_species.tsv", sep="\t")

    species_to_strains = {}


    agora_strains = agora_organisms["ncbiid"].dropna().astype(int).tolist()
    print(f"Number of Agora strains: {len(agora_strains)}")
    for strain in agora_strains:
        species_taxid = get_species_taxid(strain)
        if species_taxid is not None:
            species_to_strains.setdefault(species_taxid, []).append(strain)

    

    species_to_strains = {sp: agora_organisms[agora_organisms["ncbiid"].isin(species_to_strains[taxid])]["organism"].astype(str).tolist()
                           for sp, taxid in converted.items() if taxid in species_to_strains}

    print(f"Number of GTDB species with strains in Agora: {len(species_to_strains)}")

    species_to_reactions = {}
    for sp, strains in species_to_strains.items():
        reactions = set()
        for strain in strains:
            strain_reactions = get_strain_reactinos(strain)
            if strain_reactions is not None:
                reactions.update([rxn.id for rxn in strain_reactions])
        species_to_reactions[sp] = reactions

    distance_matrix = pd.DataFrame(index=species_to_reactions.keys(), columns=species_to_reactions.keys(), dtype=float)
    for sp1, reactions1 in species_to_reactions.items():
        for sp2, reactions2 in species_to_reactions.items():
            distance_matrix.at[sp1, sp2] = 1-jaccard_index(reactions1, reactions2)

    distance_matrix.to_csv("species_distance_matrix.csv")
    distance_matrix = pd.read_csv("species_distance_matrix.csv", index_col=0)

    print("Distance matrix computed, Building UPGMA tree...")
    tree = upgma(distance_matrix.to_numpy())
    newick_string = tree.to_newick(include_distance=True)
    for sp in distance_matrix.index:
        newick_string = newick_string.replace(f"({distance_matrix.index.get_loc(sp)}:", f"(\'{sp}\':").replace(f",{distance_matrix.index.get_loc(sp)}:", f",\'{sp}\':")

    with open("species_upgma_tree.nwk", "w") as f:
        f.write(newick_string)