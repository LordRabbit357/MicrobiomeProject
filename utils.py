import pandas as pd
from ete3 import NCBITaxa

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
        result[sp] = species_to_ncbi.get(key, None)
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


if __name__ == "__main__":
    metadata_path = "bac120_metadata_r207.tsv"

    gtdb_species_list = pd.read_csv("train/microbiome.csv").set_index("SampleID").columns.tolist()

    # Load metadata and build mapping
    metadata_df = load_gtdb_metadata(metadata_path)
    species_to_ncbi = build_species_to_ncbi_map(metadata_df)

    # Convert GTDB → NCBI
    converted = convert_species_list(gtdb_species_list, species_to_ncbi)

    agora_organisms = pd.read_csv("vmh_species.tsv", sep="\t")

    species_to_strains = {}


    agora_strains = agora_organisms["ncbiid"].dropna().astype(int).tolist()
    for strain in agora_strains:
        species_taxid = get_species_taxid(strain)
        if species_taxid is not None:
            species_to_strains.setdefault(species_taxid, []).append(strain)

    # Print results
    for sp, taxid in converted.items():
        if taxid is not None and taxid in species_to_strains:
            print(f"{sp} → {taxid} -> {species_to_strains[taxid]}")