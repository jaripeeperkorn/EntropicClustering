import math
import networkx as nx
from collections import defaultdict

def graph_density(activity_counts, edge_counts):
    """
    Computes the density of the directed graph.
    """
    num_nodes = len(activity_counts)
    num_edges = len(edge_counts)
    
    if num_nodes < 2:
        return 0  # A single node or empty graph has no meaningful density
    
    max_edges = num_nodes * (num_nodes - 1)  # Directed graph max edges
    return num_edges / max_edges

def graph_entropy(activity_counts, edge_counts):
    """
    Computes the Structural Graph Entropy of the graph based on transition probabilities.
    More speciically we calculate Markov-based entropy or Shannon entropy for random walks on a graph.
    """
    outgoing_counts = defaultdict(int)
    for (src, _), count in edge_counts.items():
        outgoing_counts[src] += count
    
    entropy = 0
    for (src, dest), count in edge_counts.items():
        prob = count / outgoing_counts[src]  # Transition probability
        entropy -= prob * math.log2(prob)
    
    return entropy

def cyclomatic_complexity(activity_counts, edge_counts):
    """
    Computes the cyclomatic complexity of the directed graph.
    """
    num_nodes = len(activity_counts)
    num_edges = len(edge_counts)
    
    # Assuming a single connected component for now (adjust if needed)
    return num_edges - num_nodes + 1 if num_nodes > 0 else 0

def num_strongly_connected_components(activity_counts, edge_counts):
    """
    Computes the number of strongly connected components (SCCs) in the directed graph.
    """
    G = nx.DiGraph()
    G.add_nodes_from(activity_counts.keys())
    G.add_edges_from(edge_counts.keys())
    
    return nx.number_strongly_connected_components(G)