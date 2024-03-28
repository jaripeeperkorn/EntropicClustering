import networkx as nx
import pm4py
import copy
from collections import defaultdict


def get_variant_log(log, order=True):
    """
    Get the variants of a given event log, together with the occurences.

    Parameters:
    - log (pm4py.objects.log.obj.EventLog): The event log for which to compute the variants.
    - order (bool, optional): Whether to order the variants by frequency. Defaults to True.

    Returns:
    - dict: A dictionary where the keys are the variant/trace tuples and the values are their frequencies.

    """
    variants = pm4py.stats.get_variants_as_tuples(log)
    if order == True:
        return dict(sorted(variants.items(), key=lambda x: x[1], reverse=True))
    else:
        return variants


def add_start_end(t):
    """
    Add start and end markers to a given trace (tuple). We use this so we can just use a graph instead of having to define initial probabilities.

    Parameters:
    - t (tuple): The trace (tuple) to which start and end markers will be added.

    Returns:
    - tuple: The modified tuple with start and end markers added.
    """
    return ('BOS',) + t + ('EOS',)


def get_dfg(variant_log):
    """
    Creates a dfg by calculating the counts of activities and edges in the variant log.

    Args:
        variant_log (dict): A dictionary where the keys are variants (sequences of activities) and the values are the occurrences of each variant in the log.

    Returns:
        activity_counts (dict): A dictionary where the keys are activities and the values are the total counts of each activity in the variant log.
        edge_counts (dict): A dictionary where the keys are pairs of activities representing edges and the values are the total counts of each edge in the variant log.
    """
    logdummy = copy.deepcopy(variant_log)
    variant_log_with_start_end = {add_start_end(key): value for key, value in logdummy.items()}
    activity_counts = defaultdict(int)
    edge_counts = defaultdict(int)
    
    for variant, occurrence in variant_log_with_start_end.items():
        for i in range(len(variant) - 1):
            current_activity, next_activity = variant[i], variant[i + 1]
            activity_counts[current_activity] += occurrence
            edge_counts[(current_activity, next_activity)] += occurrence
        #count last activity
        activity_counts[variant[-1]] += occurrence
    
    return dict(activity_counts), dict(edge_counts)

def update_dfg(activity_counts, edge_counts, new_trace, occurrence):
    """
    Update the Directly-Follows Graph (DFG) with a new trace.

    Parameters:
    - activity_counts (defaultdict): A dictionary containing the counts of each activity in the DFG.
    - edge_counts (defaultdict): A dictionary containing the counts of each edge in the DFG.
    - new_trace (tuple): The new trace to be added to the DFG.
    - occurrence (int): The number of times the new trace occurred.

    Returns:
    - activity_counts (defaultdict): The updated activity counts after adding the new trace.
    - edge_counts (defaultdict): The updated edge counts after adding the new trace.
    """
    new_trace_with_start_end = add_start_end(new_trace)
    for i in range(len(new_trace_with_start_end) - 1):
        current_activity = new_trace_with_start_end[i]
        next_activity = new_trace_with_start_end[i + 1]
        # Add new activity and edge if they do not exist
        # else add occurence to counts
        if current_activity not in activity_counts:
            activity_counts[current_activity] = occurrence
        else:
            activity_counts[current_activity] += occurrence
        if (current_activity, next_activity) not in edge_counts:
            edge_counts[(current_activity, next_activity)] = occurrence
        else:
            edge_counts[(current_activity, next_activity)] += occurrence

    #add end count
    activity_counts[new_trace_with_start_end[-1]] += occurrence

    return activity_counts, edge_counts

def get_probability(activity_counts, edge_counts, trace):
    """
    Calculate the probability of a given trace to be replayed by a graph.

    Parameters:
    - activity_counts (dict): A dictionary containing the counts of each activity in the graph.
    - edge_counts (dict): A dictionary containing the counts of each edge in the graph.
    - trace (tuple): The trace for which the probability needs to be calculated.

    Returns:
    - float: The probability of the given trace in the graph.
    """
    total_probability = 1.0
    trace_with_start_end = add_start_end(trace)
    for i in range(len( trace_with_start_end) - 1): 
        current_activity =  trace_with_start_end[i]
        next_activity =  trace_with_start_end[i + 1]      
        # Calculate the probability of taking the edge from current_activity to next_activity
        outgoing_edges = sum(edge_counts.get((current_activity, other_element), 0) for other_element in activity_counts.keys())
        if outgoing_edges > 0:
            edge_probability = edge_counts.get((current_activity, next_activity), 0) / outgoing_edges
            total_probability *= edge_probability
        else:
            # If no outgoing edges, set probability to 0
            total_probability = 0.0
            break
    return total_probability



import networkx as nx
import matplotlib.pyplot as plt

def visualize_graph(node_counts, edge_counts):
    """
    Visualizes a directed graph using NetworkX and Matplotlib.

    Parameters:
    - node_counts (dict): A dictionary mapping nodes to their counts.
    - edge_counts (dict): A dictionary mapping edges to their counts.

    Returns:
    None
    """
    # Create an empty directed graph
    G = nx.DiGraph()

    # Add nodes to the graph
    for node, count in node_counts.items():
        G.add_node(node, count=count)

    # Add edges to the graph
    for edge, count in edge_counts.items():
        node1, node2 = edge
        G.add_edge(node1, node2, count=count)

    # Draw the graph
    pos = nx.spring_layout(G)  # Position nodes using Fruchterman-Reingold force-directed algorithm

    # Calculate node sizes and edge widths based on counts
    max_node_count = max(node_counts.values())
    max_edge_count = max(edge_counts.values())
    node_sizes = [100 + 300 * (data['count'] / max_node_count) for node, data in G.nodes(data=True)]
    edge_widths = [0.5 + 1.5 * (data['count'] / max_edge_count) for _, _, data in G.edges(data=True)]

    nx.draw(G, pos, with_labels=True, node_size=node_sizes, width=edge_widths, edge_color='gray', node_color='skyblue', font_size=10, font_weight='bold', arrows=True)

    # Draw node count annotations
    for node, (x, y) in pos.items():
        plt.text(x, y + 0.05, s=str(node_counts[node]), fontsize=8, ha='center', bbox=dict(facecolor='white', edgecolor='none', alpha=0.5))

    # Draw edge count annotations
    edge_labels = nx.get_edge_attributes(G, 'count')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.show()


def filter_log_with_vl(log, variantlog):
    variants = list(variantlog.keys())
    filtered = pm4py.filtering.filter_variants(log, variants)
    return filtered
