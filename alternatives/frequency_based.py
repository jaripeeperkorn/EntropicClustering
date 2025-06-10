import copy
from sklearn.cluster import KMeans
from sklearn import preprocessing
import numpy as np

from entroclus import utils as utils

def get_alphabet(trace_list):
    """
    Return a list of unique activities present in the given trace list.

    Parameters:
    - trace_list (list): A list of traces, where each trace is represented as a list of activities.

    Returns:
    - list: A list of unique activities present in the trace list.
    """
    activities = set([act for trace in trace_list for act in trace])
    return list(activities)

def apply_integer_map(log, map):
    """
    Apply an integer map to a log.

    Parameters:
    - log (list): The log to apply the integer map to. Each element in the log is a list of elements.
    - map (dict): The integer map to apply. The keys are the original elements in the log and the values are the corresponding integer values to be mapped to.

    Returns:
    - list: The log with the integer map applied. Each element in the log is a list of integers.
    """
    return [[map[a] for a in t] for t in log]

def frequency_based_clustering(input_variant_log, num_clus, version='k-means++', distance='normalized'):
    """
    Perform frequency-based clustering on a variant log.

    Parameters:
    - input_variant_log (dict): The variant log to perform clustering on. The keys are the variant names and the values are the corresponding traces.
    - num_clus (int): The number of clusters to create.
    - version (str, optional): The initialization method for K-means clustering. Defaults to 'k-means++'.
    - distance (str, optional): The distance metric to use for clustering. Can be 'euclidian' or 'normalized'. Defaults to 'normalized'.

    Returns:
    - list: A list of dictionaries representing the clusters. Each dictionary contains the variant names as keys and the corresponding traces as values.
    """
    variant_log = copy.deepcopy(input_variant_log)
    keys = list(variant_log.keys())
    voc = get_alphabet(keys)
    string_to_index = {string: index for index, string in enumerate(voc)}
    temp_log = apply_integer_map(keys, string_to_index)
    count_log = np.zeros((len(temp_log), len(voc)), dtype=int)
    for i, trace in enumerate(temp_log):
        for act in trace:
            count_log[i, act] += 1
    if distance == 'euclidian':
        kmeans = KMeans(n_clusters=num_clus, init=version).fit(count_log)
    if distance == 'normalized':
        #proportional to cosine distance
        kmeans = KMeans(n_clusters=num_clus, init=version).fit(preprocessing.normalize(count_log))
    labels = kmeans.labels_
    clusters = [{} for _ in range(num_clus)]
    for i, key in enumerate(keys):
        clusters[labels[i]][key] = variant_log[key] 
    return clusters

def cluster(log, num_clus, outputshape='log', version='k-means++', distance='normalized'):
    """
    Perform clustering on a log.

    Parameters:
    - log (list): The log to perform clustering on.
    - num_clus (int): The number of clusters to create.
    - outputshape (str, optional): The shape of the output. Can be 'log' or 'variant_log'. Defaults to 'log'.
    - version (str, optional): The initialization method for K-means clustering. Defaults to 'k-means++'.
    - distance (str, optional): The distance metric to use for clustering. Can be 'euclidian' or 'normalized'. Defaults to 'normalized'.

    Returns:
    - list: A list of logs representing the clusters. Each log contains the traces belonging to a cluster.
    - or
    - list: A list of dictionaries representing the clusters. Each dictionary contains the variant names as keys and the corresponding traces as values.

    Raises:
    - ValueError: If the outputshape parameter is not 'log' or 'variant_log'.
    """
    variant_log_input = utils.get_variant_log(log)
    clusters_vl = frequency_based_clustering(variant_log_input, num_clus, version=version, distance=distance)
    if outputshape == 'log':
        return [utils.filter_log_with_vl(log, cluster_vl) for cluster_vl in clusters_vl]
    elif outputshape == 'variant_log':
        return clusters_vl
    else:
        raise ValueError("Output has to be 'log' or 'variant_log'.")