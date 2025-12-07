import pandas as pd
import numpy as np
from ete3 import NCBITaxa, Tree, TreeStyle, NodeStyle, TextFace
import cobra
import warnings
from pathlib import Path
import requests
from biotite.sequence.phylo import upgma
from grakel import Graph
from grakel.kernels import WeisfeilerLehman, VertexHistogram, GraphletSampling
import networkx as nx
import pickle
import mantel
import netlsd
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from scipy.spatial.distance import pdist, squareform


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
    print("Building metabolite-reaction network for strain:", strain_name)

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

def graphlet_sampling_kernel_similarity(graphs):
    """
    Compute the Graphlet Sampling graph kernel between two graphs.
    Each graph is represented as a Grakel Graph object.
    Returns the similarity score.
    """
    print("Computing Graphlet Sampling graph kernel similarity...")
    gs_kernel = GraphletSampling()
    K = gs_kernel.fit_transform(graphs)
    return K

def netlsd_embedding(graphs, time_scales=np.logspace(-2, 2, 250)):
    """
    Compute the NetLSD embeddings for a list of networkx Graphs.
    Returns a list of embeddings.
    """
    print("Computing NetLSD embeddings...")
    embeddings = []
    for G in graphs:
        emb = netlsd.heat(G, timescales=time_scales)
        embeddings.append(emb)
    return np.vstack(embeddings)

def wl_relabel(G, n_iter=2):
    labels = {n: str(G.nodes[n].get("label", "0")) for n in G.nodes()}
    all_labels = []

    for _ in range(n_iter):
        new_labels = {}
        for n in G.nodes():
            neigh = sorted(labels[v] for v in G.neighbors(n))
            new_label = labels[n] + "_" + "_".join(neigh)
            new_labels[n] = new_label
            all_labels.append(new_label)
        labels = new_labels

    return all_labels

def graph2vec(graphs, dim=128, wl_iter=2, epochs=20):
    documents = []
    for i, G in enumerate(graphs):
        words = wl_relabel(G, wl_iter)
        documents.append(TaggedDocument(words, [i]))

    model = Doc2Vec(
        vector_size=dim,
        window=5,
        min_count=1,
        workers=4,
        epochs=epochs
    )

    model.build_vocab(documents)
    model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)

    embeddings = np.vstack([model.dv[i] for i in range(len(graphs))])
    return embeddings

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

def normalize_distance_matrix(distance_matrix):
    """Normalize a distance matrix to [0, 1]."""
    min_val = distance_matrix.values.min()
    max_val = distance_matrix.values.max()
    normalized = (distance_matrix - min_val) / (max_val - min_val)
    return normalized

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
    """
    Convert a similarity kernel matrix to a distance matrix.
    D(i, j) = sqrt(K(i, i) + K(j, j) - 2 * K(i, j))
    """
    n = K.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = np.sqrt(K[i, i] + K[j, j] - 2 * K[i, j])
    return D
    

def run_wl(species_to_strains, species_to_ncbi, path_to_models="models/", currency_metabolites=None, prefix="wl"):
    try:
        species_to_graph = pickle.load(open(f"{prefix}_species_to_graph.pkl", "rb"))
    except FileNotFoundError:
        species_to_graph = map_species_to_graph(species_to_strains, path_to_models=path_to_models, currency_metabolites=currency_metabolites, nodes_as_reactions=True, convert_to_grakel=True)
        pickle.dump(species_to_graph, open(f"{prefix}_species_to_graph.pkl", "wb"))

    wl_distance_matrix = wl_graph_kernel_similarity(list(species_to_graph.values()))
    wl_distance_matrix = kernel_to_distance(wl_distance_matrix)
    wl_distance_matrix = pd.DataFrame(wl_distance_matrix, index=species_to_graph.keys(), columns=species_to_graph.keys())
    wl_distance_matrix = normalize_distance_matrix(wl_distance_matrix)

    wl_distance_matrix.to_csv(f"{prefix}_distance_matrix.csv")
    wl_distance_matrix = pd.read_csv(f"{prefix}_distance_matrix.csv", index_col=0)

    
    print("Distance matrix computed, Building UPGMA tree...")
    newick_string = construct_tree_from_distance_matrix(wl_distance_matrix, wl_distance_matrix.index)

    with open(f"{prefix}_upgma_tree.nwk", "w") as f:
        f.write(newick_string)

    tree = Tree(f"{prefix}_upgma_tree.nwk")

    render_tree(tree, annotations={'\''+sp+'\'': PHYLUM_COLORS[get_phylum_taxid(species_to_ncbi["s__" +sp])] 
                for sp in wl_distance_matrix.columns if "s__" +sp in species_to_ncbi}, save=True, output_file=f"{prefix}_colored_tree.png")
    
def run_graphlet_sampling(species_to_strains, species_to_ncbi, path_to_models="models/", currency_metabolites=None, prefix="graphlet_sampling"):
    try:
        species_to_graph = pickle.load(open(f"{prefix}_species_to_graph.pkl", "rb"))
    except FileNotFoundError:
        species_to_graph = map_species_to_graph(species_to_strains, path_to_models=path_to_models, currency_metabolites=currency_metabolites, nodes_as_reactions=True, convert_to_grakel=True)
        pickle.dump(species_to_graph, open(f"{prefix}_species_to_graph.pkl", "wb"))

    gs_distance_matrix = graphlet_sampling_kernel_similarity(list(species_to_graph.values()))
    gs_distance_matrix = kernel_to_distance(gs_distance_matrix)
    gs_distance_matrix = pd.DataFrame(gs_distance_matrix, index=species_to_graph.keys(), columns=species_to_graph.keys())
    

    gs_distance_matrix.to_csv(f"{prefix}_distance_matrix.csv")
    gs_distance_matrix = pd.read_csv(f"{prefix}_distance_matrix.csv", index_col=0)

    
    print("Distance matrix computed, Building UPGMA tree...")
    newick_string = construct_tree_from_distance_matrix(gs_distance_matrix, gs_distance_matrix.index)

    with open(f"{prefix}_upgma_tree.nwk", "w") as f:
        f.write(newick_string)

    tree = Tree(f"{prefix}_upgma_tree.nwk")

    render_tree(tree, annotations={'\''+sp+'\'': PHYLUM_COLORS[get_phylum_taxid(species_to_ncbi["s__" +sp])] 
                for sp in gs_distance_matrix.columns if "s__" +sp in species_to_ncbi}, save=True, output_file=f"{prefix}_colored_tree.png")
    
def run_netlsd(species_to_strains, species_to_ncbi, path_to_models="models/", currency_metabolites=None, prefix="netlsd"):
    try:
        species_to_graph = pickle.load(open(f"{prefix}_species_to_graph.pkl", "rb"))
    except FileNotFoundError:
        species_to_graph = map_species_to_graph(species_to_strains, path_to_models=path_to_models, currency_metabolites=currency_metabolites, nodes_as_reactions=False)
        pickle.dump(species_to_graph, open(f"{prefix}_species_to_graph.pkl", "wb"))

    embeddings = netlsd_embedding(list(species_to_graph.values()))
    distance_matrix = pdist(embeddings, metric='euclidean')
    distance_matrix = squareform(distance_matrix)
    distance_matrix = pd.DataFrame(distance_matrix, index=species_to_graph.keys(), columns=species_to_graph.keys(), dtype=float)

    distance_matrix.to_csv(f"{prefix}_distance_matrix.csv")
    distance_matrix = pd.read_csv(f"{prefix}_distance_matrix.csv", index_col=0)

    newick_string = construct_tree_from_distance_matrix(distance_matrix, distance_matrix.index)

    with open(f"{prefix}_upgma_tree.nwk", "w") as f:
        f.write(newick_string)

    tree = Tree(f"{prefix}_upgma_tree.nwk")

    render_tree(tree, annotations={'\''+sp+'\'': PHYLUM_COLORS[get_phylum_taxid(species_to_ncbi["s__" +sp])] 
                for sp in distance_matrix.columns if "s__" +sp in species_to_ncbi}, save=True, output_file=f"{prefix}_colored_tree.png")
    
def run_graph2vec(species_to_strains, species_to_ncbi, path_to_models="models/", currency_metabolites=None, prefix="graph2vec"):
    try:
        species_to_graph = pickle.load(open(f"{prefix}_species_to_graph.pkl", "rb"))
    except FileNotFoundError:
        species_to_graph = map_species_to_graph(species_to_strains, path_to_models=path_to_models, currency_metabolites=currency_metabolites, nodes_as_reactions=True)
        pickle.dump(species_to_graph, open(f"{prefix}_species_to_graph.pkl", "wb"))

    embeddings = graph2vec(list(species_to_graph.values()))
    similarity_matrix = cosine_similarity(embeddings)
    for i in range(similarity_matrix.shape[0]):
        for j in range(similarity_matrix.shape[1]):
            if similarity_matrix[i, j] > 1.0:
                similarity_matrix[i, j] = 1.0 # for some reason the diagonal values are slightly > 1.0 fml i hate sklearn
    distance_matrix = np.sqrt(2 - 2 * similarity_matrix)
    distance_matrix = pd.DataFrame(distance_matrix, index=species_to_graph.keys(), columns=species_to_graph.keys(), dtype=float)

    distance_matrix.to_csv(f"{prefix}_distance_matrix.csv")
    distance_matrix = pd.read_csv(f"{prefix}_distance_matrix.csv", index_col=0)

    newick_string = construct_tree_from_distance_matrix(distance_matrix, distance_matrix.index)

    with open(f"{prefix}_upgma_tree.nwk", "w") as f:
        f.write(newick_string)

    tree = Tree(f"{prefix}_upgma_tree.nwk")

    render_tree(tree, annotations={'\''+sp+'\'': PHYLUM_COLORS[get_phylum_taxid(species_to_ncbi["s__" +sp])] 
                for sp in distance_matrix.columns if "s__" +sp in species_to_ncbi}, save=True, output_file=f"{prefix}_colored_tree.png")    

def run_graph_intersection(species_to_strains, species_to_ncbi, path_to_models="models/", currency_metabolites=None, prefix="graph_intersection"):
    try:
        species_to_graph = pickle.load(open(f"{prefix}_species_to_graph.pkl", "rb"))
    except FileNotFoundError:
        species_to_graph = map_species_to_graph(species_to_strains, path_to_models=path_to_models, currency_metabolites=currency_metabolites, nodes_as_reactions=True, convert_to_grakel=False)
        pickle.dump(species_to_graph, open(f"{prefix}_species_to_graph.pkl", "wb"))

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
    try:
        species_to_graph = pickle.load(open(f"{prefix}_species_to_graph.pkl", "rb"))
    except FileNotFoundError:
        species_to_graph = map_species_to_graph(species_to_strains, path_to_models=path_to_models, currency_metabolites=currency_metabolites, nodes_as_reactions=True)
        pickle.dump(species_to_graph, open(f"{prefix}_species_to_graph.pkl", "wb"))

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
    
    
def get_equivalent_phylo_leaf_name(sp, species_to_ncbi, phylo_tree):
    if sp in phylo_tree.get_leaf_names():
        corresponding_leaf_name = '\'' + sp + '\''
    else:
        taxid = species_to_ncbi["s__" + sp]
        for leaf in phylo_tree.get_leaves():
            if species_to_ncbi["s__" + leaf.name.replace("'", "")] == taxid:
                corresponding_leaf_name = leaf.name
                break
    return corresponding_leaf_name

def search_gtdb_nodes(taxonomy, tree, name):
    search_term = f"s__{name}"
    accessions = taxonomy[taxonomy["taxonomy"].str.contains(search_term, regex=False)]["accession"].tolist()
    for acc in accessions:
        leaf = tree.search_nodes(name=acc)
        if leaf:
            return leaf[0]
    return None

def compute_path_to_root(leaf):
    path = []
    node = leaf
    while node:
        path.append(node)
        node = node.up
    return path


    
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

    try:
        species_to_graph = pickle.load(open(f"ag_reduction_species_to_graph.pkl", "rb"))
    except FileNotFoundError:
        species_to_graph = map_species_to_graph(species_to_strains, path_to_models="./models/", currency_metabolites=currency_metabolites, nodes_as_reactions=False)
        pickle.dump(species_to_graph, open(f"ag_reduction_species_to_graph.pkl", "wb"))

    species_to_metabolites = {}
    for sp, G in species_to_graph.items():
        species_to_metabolites[sp] = set()
        for node in G.nodes():
            species_to_metabolites[sp].add(node)

    mutual_metabolites = set.intersection(*species_to_metabolites.values())

    #run_jaccard(species_to_strains, species_to_ncbi)
    # run_wl(species_to_strains, species_to_ncbi, path_to_models="models/")
    # run_wl(species_to_strains, species_to_ncbi, currency_metabolites=currency_metabolites, prefix="wl_currency")
    # run_wl(species_to_strains, species_to_ncbi, currency_metabolites=mutual_metabolites, prefix="wl_agressive")
    # run_netlsd(species_to_strains, species_to_ncbi, path_to_models="models/")
    # run_netlsd(species_to_strains, species_to_ncbi, path_to_models="models/", currency_metabolites=currency_metabolites, prefix="netlsd_currency")
    # run_netlsd(species_to_strains, species_to_ncbi, path_to_models="models/", currency_metabolites=mutual_metabolites, prefix="netlsd_agressive")
    # run_graph_intersection(species_to_strains, species_to_ncbi, path_to_models="models/", currency_metabolites=mutual_metabolites, prefix="graph_intersection_agressive")
    # run_source_sink(species_to_strains, species_to_ncbi, path_to_models="models/", currency_metabolites=mutual_metabolites, prefix="source_sink_agressive")
    # run_graph2vec(species_to_strains, species_to_ncbi, path_to_models="models/")
    # run_graph2vec(species_to_strains, species_to_ncbi, path_to_models="models/", currency_metabolites=mutual_metabolites, prefix="graph2vec_currency")
    # run_graphlet_sampling(species_to_strains, species_to_ncbi, path_to_models="models/")
    # run_graphlet_sampling(species_to_strains, species_to_ncbi, path_to_models="models/", currency_metabolites=mutual_metabolites, prefix="graphlet_sampling_currency")
    # run_graph_intersection(species_to_strains, species_to_ncbi, path_to_models="models/", currency_metabolites=currency_metabolites, prefix="graph_intersection_currency")
    # run_graph_intersection(species_to_strains, species_to_ncbi, path_to_models="models/", currency_metabolites=None, prefix="graph_intersection")
    # run_source_sink(species_to_strains, species_to_ncbi, path_to_models="models/", currency_metabolites=None, prefix="source_sink")
    # run_source_sink(species_to_strains, species_to_ncbi, path_to_models="models/", currency_metabolites=currency_metabolites, prefix="source_sink_currency")

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

    # phylo_tree = Tree("ncbi_phylo_tree.nwk")
    # phylo_tree_distance_matrix = pd.DataFrame(index=species_to_strains.keys(), columns=species_to_strains.keys(), dtype=float)
    # for sp1 in species_to_strains.keys():
    #     corresponding_leaf1_name = get_equivalent_phylo_leaf_name(sp1, species_to_ncbi, phylo_tree)
    #     for sp2 in species_to_strains.keys():
    #         corresponding_leaf2_name = get_equivalent_phylo_leaf_name(sp2, species_to_ncbi, phylo_tree)
    #         node1 = phylo_tree.search_nodes(name=corresponding_leaf1_name)[0]
    #         node2 = phylo_tree.search_nodes(name=corresponding_leaf2_name)[0]
    #         distance = phylo_tree.get_distance(node1, node2)
    #         phylo_tree_distance_matrix.at[sp1, sp2] = distance

    # phylo_tree_distance_matrix.to_csv("ncbi_phylo_tree_distance_matrix.csv")
    
    # phylo_tree_upgma = construct_tree_from_distance_matrix(phylo_tree_distance_matrix, phylo_tree_distance_matrix.index)
    # with open("ncbi_phylo_tree_upgma.nwk", "w") as f:
    #     f.write(phylo_tree_upgma)
    # render_tree(Tree("ncbi_phylo_tree_upgma.nwk"), annotations={'\''+sp+'\'': PHYLUM_COLORS[get_phylum_taxid(species_to_ncbi["s__" + sp])] 
    #             for sp in phylo_tree_distance_matrix.columns if "s__" +sp in species_to_ncbi}, save=True, output_file="ncbi_phylo_tree_upgma_colored.png")

    gtdb_phylo_tree = Tree("bac120_r207.nwk", format=1, quoted_node_names=True)
    taxonomy = pd.read_csv("bac120_taxonomy_r207.tsv", sep="\t", header=None,
                       names=["accession", "taxonomy"])
    try:
        species_to_paths = pickle.load(open("species_to_paths.pkl", "rb"))
    except FileNotFoundError:
        species_to_paths = {}
        for s in species_to_strains:
            node = search_gtdb_nodes(taxonomy, gtdb_phylo_tree, s)
            species_to_paths[s] = compute_path_to_root(node)
        pickle.dump(species_to_paths, open("species_to_paths.pkl", "wb"))

    print(f"Finished mapping {len(species_to_paths)} species to paths")

    gtdb_phylo_distance_matrix = pd.DataFrame(index=species_to_strains.keys(), columns=species_to_strains.keys(), dtype=float)
    finished = set()
    for sp1 in species_to_strains.keys():
        p1 = species_to_paths[sp1]
        set_p1 = set(p1)
        for sp2 in species_to_strains.keys():
            if sp2 in finished:
                continue
            p2 = species_to_paths[sp2]

            # Find lowest common ancestor by walking up the paths
            lca = next(n for n in p2 if n in set_p1)

            # Distances = dist(leaf1→LCA) + dist(leaf2→LCA)
            d1 = sum(node.dist for node in p1[:p1.index(lca)])
            d2 = sum(node.dist for node in p2[:p2.index(lca)])
            gtdb_phylo_distance_matrix.loc[sp1][sp2] = d1 + d2
            gtdb_phylo_distance_matrix.loc[sp2][sp1] = d1 + d2
        finished.add(sp1)
        print(f"finished {len(finished)} out of {len(species_to_strains.keys())}")

    gtdb_phylo_distance_matrix.to_csv("gtdb_phylo_tree_distance_matrix.csv")

    gtdb_phylo_tree_upgma = construct_tree_from_distance_matrix(gtdb_phylo_distance_matrix, gtdb_phylo_distance_matrix.index)
    with open("gtdb_phylo_tree_upgma.nwk", "w") as f:
        f.write(gtdb_phylo_tree_upgma)
    render_tree(Tree("gtdb_phylo_tree_upgma.nwk"), annotations={'\''+sp+'\'': PHYLUM_COLORS[get_phylum_taxid(species_to_ncbi["s__" + sp])] 
                for sp in gtdb_phylo_distance_matrix.columns if "s__" +sp in species_to_ncbi}, save=True, output_file="gtdb_phylo_tree_upgma_colored.png")


    distance_matrices_names = [n for n in Path(".").glob("*_distance_matrix.csv")]
    mantel_matrix = pd.DataFrame(index=distance_matrices_names, columns=distance_matrices_names, dtype=float)
    for mat1 in distance_matrices_names:
        for mat2 in distance_matrices_names:
            print(f"Computing Mantel test between {mat1} and {mat2}...")
            dm1 = pd.read_csv(mat1, index_col=0)
            dm2 = pd.read_csv(mat2, index_col=0)
            result = mantel.test(dm1.to_numpy(), dm2.to_numpy(), method='pearson', perms=10000, tail='two-tail')
            mantel_matrix.loc[mat1, mat2] = result.r

    mantel_matrix.to_csv("mantel_correlation_matrix.csv")


    


