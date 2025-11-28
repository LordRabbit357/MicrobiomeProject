import pandas as pd
import numpy as np
from ete3 import NCBITaxa, Tree, TreeStyle, NodeStyle, TextFace
import cobra
import warnings
from pathlib import Path
import requests
from biotite.sequence.phylo import upgma
from grakel import Graph
from grakel.kernels import WeisfeilerLehman, VertexHistogram
import networkx as nx
import pickle
import mantel


PHYLUM_COLORS = {976: "blue", 1224: "orange", 1239: "green", 74201: "red", 200940: "purple", 201174: "brown"}

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

def get_strain_reactions(strain_name, path_to_models="models/"):
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
    
def get_strain_reactions_network(strain_name, path_to_models="models/", currency_metabolites=None):
    """
    Load an SBML model for a given strain name and return its metabolic network as a networkx Graph.
    Nodes represent metabolites, edges represent reactions.
    """
        
    if currency_metabolites is None:
        currency_metabolites = set()

    producers = {}   # metabolite -> set(reactions)
    consumers = {}   # metabolite -> set(reactions)

    G = nx.DiGraph()

    # --- index reactions ---
    for rxn in get_strain_reactions(strain_name, path_to_models):
        rid = rxn.id
        G.add_node(rid, label=rxn.name or rid)

        for met in rxn.products:
            if met.id in currency_metabolites:
                continue
            producers.setdefault(met.id, set()).add(rid)

        for met in rxn.reactants:
            if met.id in currency_metabolites:
                continue
            consumers.setdefault(met.id, set()).add(rid)

    # --- connect reactions ---
    for met_id in producers.keys() & consumers.keys():
        for r_prod in producers[met_id]:
            for r_cons in consumers[met_id]:
                if r_prod != r_cons:
                    G.add_edge(r_prod, r_cons)

    return G

def get_strain_metabolites_network(strain_name, path_to_models="models/", currency_metabolites=None):
    """
    Load an SBML model for a given strain name and return its metabolic network as a networkx Graph.
    Nodes represent reactions, edges represent metabolites.
    """
        
    if currency_metabolites is None:
        currency_metabolites = set()

    G = nx.DiGraph()

    # --- index reactions ---
    for rxn in get_strain_reactions(strain_name, path_to_models):
        rid = rxn.id
        G.add_node(rid, label=rxn.name or rid)

        for product in rxn.products:
            if product.id in currency_metabolites:
                continue
            G.add_node(product.id, label=product.name or product.id)
            for reactant in rxn.reactants:
                if reactant.id in currency_metabolites:
                    continue
                G.add_node(reactant.id, label=reactant.name or reactant.id)
                G.add_edge(reactant.id, product.id)

    return G


def grakel_to_nx(Gk, directed=False):
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    # edges
    for u, v in Gk.get_edges():
        G.add_edge(u, v)

    # node labels
    if Gk.node_labels is not None:
        for n, label in Gk.node_labels.items():
            G.nodes[n]["label"] = label

    return G

def nx_to_grakel(G):
    node_map = {n: i for i, n in enumerate(G.nodes())}

    edges = [
        (node_map[u], node_map[v])
        for u, v in G.edges()
    ]

    labels = {
        node_map[n]: d["label"]
        for n, d in G.nodes(data=True)
    }

    return Graph(
        edges,
        node_labels=labels
    )
    
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

def graph_intersection_index(G1, G2):
    """
    Compute the graph intersection index between two networkx Graphs.
    GI(G1, G2) = |E(G1) ∩ E(G2)| / |E(G1) ∪ E(G2)|
    """
    edges1 = set(G1.edges())
    edges2 = set(G2.edges())
    intersection = len(edges1.intersection(edges2))
    union = len(edges1.union(edges2))
    if union == 0:
        return 0.0
    return intersection / union

def source_sink_index(G1, G2):
    """
    Compute the source-sink index between two networkx Graphs.
    SSI(G1, G2) = |S(G1) ∩ S(G2)| + |T(G1) ∩ T(G2)| / |S(G1) ∪ S(G2)| + |T(G1) ∪ T(G2)|
    where S(G) is the set of source nodes and T(G) is the set of sink nodes in G.
    """
    def get_sources_sinks(G):
        sources = {n for n in G.nodes() if G.in_degree(n) == 0 and G.out_degree(n) > 0}
        sinks = {n for n in G.nodes() if G.out_degree(n) == 0 and G.in_degree(n) > 0}
        return sources, sinks

    sg1, st1 = get_sources_sinks(G1)
    sg2, st2 = get_sources_sinks(G2)

    intersection = len(sg1.intersection(sg2)) + len(st1.intersection(st2))
    union = len(sg1.union(sg2)) + len(st1.union(st2))
    if union == 0:
        return 0.0
    return intersection / union

def wl_graph_kernel_similarity(graphs):
    """
    Compute the Weisfeiler-Lehman graph kernel between two graphs.
    Each graph is represented as a Grakel Graph object.
    Returns the similarity score.
    """
    print("Computing WL graph kernel similarity...")
    wl_kernel = WeisfeilerLehman(n_iter=5, base_graph_kernel=VertexHistogram)
    K = wl_kernel.fit_transform(graphs)
    return K

def graph_edit_distance_labeled(
    G1,
    G2,
    node_label_attr="label",
    node_subst_cost=1.0,
    node_del_cost=5.0,
    node_ins_cost=5.0,
    edge_del_cost=1.0,
    edge_ins_cost=1.0,
    timeout=None
):
    """
    Compute graph edit distance between two labeled NetworkX graphs.

    Parameters
    ----------
    G1, G2 : nx.Graph or nx.DiGraph
        Input graphs.
    node_label_attr : str
        Node attribute name containing labels.
    node_subst_cost : float
        Cost for substituting nodes with different labels.
    node_del_cost, node_ins_cost : float
        Node deletion / insertion cost.
    edge_del_cost, edge_ins_cost : float
        Edge deletion / insertion cost.
    timeout : float or None
        Maximum time (seconds) to spend computing GED.

    Returns
    -------
    float
        Graph edit distance.
    """

    def node_subst(u, v):
        l1 = u["label"]
        l2 = v["label"]
        return 0.0 if l1 == l2 else node_subst_cost

    def node_del(u):
        return node_del_cost

    def node_ins(v):
        return node_ins_cost

    def edge_del(e):
        return edge_del_cost

    def edge_ins(e):
        return edge_ins_cost

    ged = nx.graph_edit_distance(
        G1,
        G2,
        node_subst_cost=node_subst,
        node_del_cost=node_del,
        node_ins_cost=node_ins,
        edge_del_cost=edge_del,
        edge_ins_cost=edge_ins,
        timeout=timeout
    )

    return ged

def get_phylum_taxid(strain_taxid):
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
        if ranks.get(taxid) == "phylum":
            return taxid
    
    # No species found in lineage
    return None

def get_leaf_annotations(node, annotations):
    """Return set of annotations for all descendant leaves."""
    ann = set()
    for leaf in node:
        if leaf.is_leaf():
            ann.add(annotations.get(leaf.name, None))
    return ann

def map_species_to_graph(species_to_strains, path_to_models="models/", currency_metabolites=None, nodes_as_reactions=True, convert_to_grakel=False):
    """Map each species to its combined pathway network graph."""
    species_to_graph = {}
    for sp, strains in species_to_strains.items():
        print(f"Processing species: {sp} with {len(strains)} strains")
        combined_graph = None
        for strain in strains:
            if nodes_as_reactions:
                strain_graph = get_strain_reactions_network(strain, path_to_models=path_to_models, currency_metabolites=currency_metabolites)
            else:
                strain_graph = get_strain_metabolites_network(strain, path_to_models=path_to_models, currency_metabolites=currency_metabolites)
            if strain_graph is not None:
                if combined_graph is None:
                    combined_graph = strain_graph
                else:
                    combined_graph = nx.compose(combined_graph, strain_graph)
        if convert_to_grakel and combined_graph is not None:
            species_to_graph[sp] = nx_to_grakel(combined_graph)
        else:
            species_to_graph[sp] = combined_graph
    return species_to_graph

def construct_tree_from_distance_matrix(distance_matrix, labels):
    """
    Construct a UPGMA tree from a distance matrix and labels.
    Returns a newick string of the tree.
    """
    tree = upgma(distance_matrix.to_numpy())
    newick_string = tree.to_newick(include_distance=True)
    for i, label in enumerate(labels):
        newick_string = newick_string.replace(f"({i}:", f"(\'{label}\':").replace(f",{i}:", f",\'{label}\':")

    return newick_string

def render_tree(tree, annotations, save=True, output_file="colored_tree.png"):
    """Render and optionally save a colored tree based on annotations."""
    for node in tree.traverse():
        # Skip leaves, handle them separately
        if node.is_leaf():
            color = annotations[node.name]
            style = NodeStyle()
            style["size"] = 14
            style["fgcolor"] = color
            style["hz_line_color"] = color
            style["vt_line_color"] = color
            style["hz_line_width"] = 3
            style["vt_line_width"] = 3
            node.set_style(style)
            continue

        ann = get_leaf_annotations(node, annotations)

        # If subtree is homogeneous → color clade
        if len(ann) == 1:
            only_color = next(iter(ann))
            style = NodeStyle()
            style["fgcolor"] = only_color
            style["size"] = 0  # internal node dot size
            style["vt_line_color"] = only_color
            style["hz_line_color"] = only_color
            style["vt_line_width"] = 3
            style["hz_line_width"] = 3
            node.set_style(style)

    ts = TreeStyle()
    ts.mode = "c"
    ts.show_leaf_name = True
    ts.legend_position = 4

    legend_items = {}
    for phylum, color in PHYLUM_COLORS.items():
        legend_items[color] = phylum  # uniq by color

    for color, label in legend_items.items():
        tf = TextFace(f"  {label}  ", fsize=12)
        tf.background.color = color
        tf.opacity = 0.8
        ts.legend.add_face(tf, column=0)


    # tree.show(tree_style=ts)
    if save:
        tree.render(output_file, w=1200, tree_style=ts)

def run_jaccard(species_to_strains, species_to_ncbi):
    species_to_reactions = {}
    for sp, strains in species_to_strains.items():
        reactions = set()
        for strain in strains:
            strain_reactions = get_strain_reactions(strain)
            if strain_reactions is not None:
                reactions.update([rxn.id for rxn in strain_reactions])
        species_to_reactions[sp] = reactions

    jaccard_distance_matrix = pd.DataFrame(index=species_to_reactions.keys(), columns=species_to_reactions.keys(), dtype=float)
    for sp1, reactions1 in species_to_reactions.items():
        for sp2, reactions2 in species_to_reactions.items():
            jaccard_distance_matrix.at[sp1, sp2] = 1-jaccard_index(reactions1, reactions2)

    jaccard_distance_matrix.to_csv("jaccard_distance_matrix.csv")
    jaccard_distance_matrix = pd.read_csv("jaccard_distance_matrix.csv", index_col=0)

    print("Distance matrix computed, Building UPGMA tree...")
    newick_string = construct_tree_from_distance_matrix(jaccard_distance_matrix, jaccard_distance_matrix.index)

    with open("jaccard_upgma_tree.nwk", "w") as f:
        f.write(newick_string)

    tree = Tree("jaccard_upgma_tree.nwk")

    render_tree(tree, annotations={'\''+sp+'\'': PHYLUM_COLORS[get_phylum_taxid(species_to_ncbi["s__" +sp])] 
                for sp in jaccard_distance_matrix.columns if "s__" +sp in species_to_ncbi}, save=True, output_file="jaccard_colored_tree.png")

def kernel_to_distance(K):
    diag = np.diag(K)
    D = np.sqrt(
        diag[:, None] + diag[None, :] - 2 * K
    )
    return D

def normalize_matrix(matrix):
    """
    Normalize a matrix to values between 0 and 1.

    Parameters
    ----------
    matrix : array-like
        2D list or NumPy array.

    Returns
    -------
    np.ndarray
        Normalized matrix with values in [0, 1].
    """
    matrix = np.asarray(matrix, dtype=float)
    min_val = matrix.min()
    max_val = matrix.max()

    if max_val == min_val:
        # Avoid division by zero: return zeros
        return np.zeros_like(matrix)

    return (matrix - min_val) / (max_val - min_val)


def run_wl(species_to_strains, species_to_ncbi, path_to_models="models/", currency_metabolites=None, prefix="wl"):
    species_to_graph = map_species_to_graph(species_to_strains, path_to_models=path_to_models, currency_metabolites=currency_metabolites, nodes_as_reactions=True, convert_to_grakel=True)
    pickle.dump(species_to_graph, open(f"{prefix}_species_to_graph.pkl", "wb"))
    species_to_graph = pickle.load(open(f"{prefix}_species_to_graph.pkl", "rb"))

    wl_distance_matrix = wl_graph_kernel_similarity(list(species_to_graph.values()))
    wl_distance_matrix = kernel_to_distance(wl_distance_matrix)
    wl_distance_matrix = normalize_matrix(wl_distance_matrix)
    wl_distance_matrix = pd.DataFrame(wl_distance_matrix, index=species_to_graph.keys(), columns=species_to_graph.keys())
    

    wl_distance_matrix.to_csv(f"{prefix}_distance_matrix.csv")
    wl_distance_matrix = pd.read_csv(f"{prefix}_distance_matrix.csv", index_col=0)

    
    print("Distance matrix computed, Building UPGMA tree...")
    newick_string = construct_tree_from_distance_matrix(wl_distance_matrix, wl_distance_matrix.index)

    with open(f"{prefix}_upgma_tree.nwk", "w") as f:
        f.write(newick_string)

    tree = Tree(f"{prefix}_upgma_tree.nwk")

    render_tree(tree, annotations={'\''+sp+'\'': PHYLUM_COLORS[get_phylum_taxid(species_to_ncbi["s__" +sp])] 
                for sp in wl_distance_matrix.columns if "s__" +sp in species_to_ncbi}, save=True, output_file=f"{prefix}_colored_tree.png")

def run_graph_intersection(species_to_strains, species_to_ncbi, path_to_models="models/", currency_metabolites=None, prefix="graph_intersection"):
    species_to_graph = map_species_to_graph(species_to_strains, path_to_models=path_to_models, currency_metabolites=currency_metabolites, nodes_as_reactions=True)

    # Compute pairwise intersections
    distance_matrix = pd.DataFrame(index=species_to_graph.keys(), columns=species_to_graph.keys(), dtype=int)
    for sp1, graph1 in species_to_graph.items():
        for sp2, graph2 in species_to_graph.items():
            distance_matrix.at[sp1, sp2] = 1- graph_intersection_index(graph1, graph2)

    distance_matrix.to_csv(f"{prefix}_distance_matrix.csv")
    distance_matrix = pd.read_csv(f"{prefix}_distance_matrix.csv", index_col=0)

    newick_string = construct_tree_from_distance_matrix(distance_matrix, distance_matrix.index)

    with open(f"{prefix}_upgma_tree.nwk", "w") as f:
        f.write(newick_string)

    tree = Tree(f"{prefix}_upgma_tree.nwk")

    render_tree(tree, annotations={'\''+sp+'\'': PHYLUM_COLORS[get_phylum_taxid(species_to_ncbi["s__" +sp])] 
                for sp in distance_matrix.columns if "s__" +sp in species_to_ncbi}, save=True, output_file=f"{prefix}_colored_tree.png")
    
def run_source_sink(species_to_strains, species_to_ncbi, path_to_models="models/", currency_metabolites=None, prefix="source_sink"):
    species_to_graph = map_species_to_graph(species_to_strains, path_to_models=path_to_models, currency_metabolites=currency_metabolites, nodes_as_reactions=False)
    pickle.dump(species_to_graph, open(f"{prefix}_species_to_graph.pkl", "wb"))
    species_to_graph = pickle.load(open(f"{prefix}_species_to_graph.pkl", "rb"))

    distance_matrix = pd.DataFrame(index=species_to_graph.keys(), columns=species_to_graph.keys(), dtype=int)
    for sp1, graph1 in species_to_graph.items():
        for sp2, graph2 in species_to_graph.items():
            distance_matrix.at[sp1, sp2] = 1- source_sink_index(graph1, graph2)

    distance_matrix.to_csv(f"{prefix}_distance_matrix.csv")
    distance_matrix = pd.read_csv(f"{prefix}_distance_matrix.csv", index_col=0)

    newick_string = construct_tree_from_distance_matrix(distance_matrix, distance_matrix.index)

    with open(f"{prefix}_upgma_tree.nwk", "w") as f:
        f.write(newick_string)

    tree = Tree(f"{prefix}_upgma_tree.nwk")

    render_tree(tree, annotations={'\''+sp+'\'': PHYLUM_COLORS[get_phylum_taxid(species_to_ncbi["s__" +sp])] 
                for sp in distance_matrix.columns if "s__" +sp in species_to_ncbi}, save=True, output_file=f"{prefix}_colored_tree.png")
    
def run_ged(species_to_strains, species_to_ncbi, path_to_models="models/", currency_metabolites=None, prefix="ged"):
    species_to_graph = map_species_to_graph(species_to_strains, path_to_models=path_to_models, currency_metabolites=currency_metabolites, nodes_as_reactions=False)

    pickle.dump(species_to_graph, open(f"{prefix}_species_to_graph.pkl", "wb"))
    species_to_graph = pickle.load(open(f"{prefix}_species_to_graph.pkl", "rb"))

    distance_matrix = pd.DataFrame(index=species_to_graph.keys(), columns=species_to_graph.keys(), dtype=int)
    for sp1, graph1 in species_to_graph.items():
        print(f"Computing distances for species: {sp1}")
        for sp2, graph2 in species_to_graph.items():
            distance_matrix.at[sp1, sp2] = graph_edit_distance_labeled(graph1, graph2, timeout=2)

    newick_string = construct_tree_from_distance_matrix(distance_matrix, distance_matrix.index)

    with open(f"{prefix}_upgma_tree.nwk", "w") as f:
        f.write(newick_string)

    tree = Tree(f"{prefix}_upgma_tree.nwk")

    render_tree(tree, annotations={'\''+sp+'\'': PHYLUM_COLORS[get_phylum_taxid(species_to_ncbi["s__" +sp])] 
                for sp in distance_matrix.columns if "s__" +sp in species_to_ncbi}, save=True, output_file=f"{prefix}_colored_tree.png")

    
if __name__ == "__main__":
    # Ignore all UserWarnings
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # metadata_path = "bac120_metadata_r207.tsv"

    # gtdb_species_list = pd.read_csv("train/microbiome.csv").set_index("SampleID").columns.tolist()

    # # Load metadata and build mapping
    # metadata_df = load_gtdb_metadata(metadata_path)
    # species_to_ncbi = build_species_to_ncbi_map(metadata_df)



    # # Convert GTDB → NCBI
    # converted = convert_species_list(gtdb_species_list, species_to_ncbi)

    # print(f"Number of GTDB species converted: {len(converted)}")

    # agora_organisms = pd.read_csv("vmh_species.tsv", sep="\t")

    # species_to_strains = {}


    # agora_strains = agora_organisms["ncbiid"].dropna().astype(int).tolist()
    # print(f"Number of Agora strains: {len(agora_strains)}")
    # for strain in agora_strains:
    #     species_taxid = get_species_taxid(strain)
    #     if species_taxid is not None:
    #         species_to_strains.setdefault(species_taxid, []).append(strain)

    

    # species_to_strains = {sp: agora_organisms[agora_organisms["ncbiid"].isin(species_to_strains[taxid])]["organism"].astype(str).tolist()
    #                        for sp, taxid in converted.items() if taxid in species_to_strains}

    # print(f"Number of GTDB species with strains in Agora: {len(species_to_strains)}")

    # pickle.dump(species_to_strains, open("species_to_strains.pkl", "wb"))
    # pickle.dump(species_to_ncbi, open("species_to_ncbi.pkl", "wb"))
    species_to_strains = pickle.load(open("species_to_strains.pkl", "rb"))
    species_to_ncbi = pickle.load(open("species_to_ncbi.pkl", "rb"))
    currency_metabolites = {
        'h2o[c]', 'atp[c]', 'adp[c]', 'pi[c]', 'ppi[c]', 'h[c]', 'nad[c]', 'nadh[c]', 
        'nadp[c]', 'nadph[c]', 'co2[c]', 'o2[c]', 'fadh2[c]', 'fad[c]', 'amp[c]'
    }

    #run_jaccard(species_to_strains, species_to_ncbi)
    # run_wl(species_to_strains, species_to_ncbi, path_to_models="models/")
    # run_wl(species_to_strains, species_to_ncbi, currency_metabolites=currency_metabolites, prefix="wl_currency")
    # run_graph_intersection(species_to_strains, species_to_ncbi, path_to_models="models/", currency_metabolites=currency_metabolites, prefix="graph_intersection_currency")
    # run_graph_intersection(species_to_strains, species_to_ncbi, path_to_models="models/", currency_metabolites=None, prefix="graph_intersection")
    # run_source_sink(species_to_strains, species_to_ncbi, path_to_models="models/", currency_metabolites=None, prefix="source_sink")
    # run_source_sink(species_to_strains, species_to_ncbi, path_to_models="models/", currency_metabolites=currency_metabolites, prefix="source_sink_currency")
    # run_ged(species_to_strains, species_to_ncbi, path_to_models="models/", currency_metabolites=None, prefix="ged") # this might take up to 48 hours
    # run_ged(species_to_strains, species_to_ncbi, path_to_models="models/", currency_metabolites=currency_metabolites, prefix="ged_currency") # this might take up to 48 hours

    # ncbi = NCBITaxa()
    # phylo_tree = ncbi.get_topology([species_to_ncbi["s__" + sp] for sp in species_to_strains.keys()])

    # for leaf in phylo_tree.get_leaves():
    #     taxid = int(leaf.name)
    #     species_name = None
    #     for sp, ncbi_id in species_to_ncbi.items():
    #         if ncbi_id == taxid:
    #             species_name = '\'' + sp[3:] + '\''
    #             break
    #     if species_name is not None:
    #         leaf.name = species_name
    #     else:
    #         leaf.name = '\'' + str(taxid) + '\''

    # phylo_tree.write(format=1, outfile="ncbi_phylo_tree.nwk")

    

    # render_tree(Tree("ncbi_phylo_tree.nwk"), annotations={'\''+sp+'\'': PHYLUM_COLORS[get_phylum_taxid(species_to_ncbi["s__" + sp])] 
    #             for sp in species_to_strains.keys()}, save=True, output_file="ncbi_colored_tree.png")


    distance_matrices_names = [n for n in Path(".").glob("*_distance_matrix.csv")]
    mantel_matrix = pd.DataFrame(index=distance_matrices_names, columns=distance_matrices_names, dtype=float)
    for mat1 in distance_matrices_names:
        for mat2 in distance_matrices_names:
            dm1 = pd.read_csv(mat1, index_col=0)
            dm2 = pd.read_csv(mat2, index_col=0)
            result = mantel.test(dm1.to_numpy(), dm2.to_numpy(), method='pearson', perms=10000, tail='two-tail')
            mantel_matrix.loc[mat1, mat2] = result.r

    mantel_matrix.to_csv("mantel_correlation_matrix.csv")


    


