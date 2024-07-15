from ..entroclus import utils as utils

import copy, random

def get_random_clusters_equisized(variant_log_input, num_clus): 
    """
    Returns random equisized clusters based on the number of variants (not trace counts).

    Parameters:
    - variant_log_input (dict): A dictionary containing variant log data.
    - num_clus (int): The number of clusters to create.

    Returns:
    - random_clusters (list): A list of dictionaries representing the random equisized clusters. Each dictionary contains variant log data for a specific cluster.
    """
    variant_log = copy.deepcopy(variant_log_input)
    keys = list(variant_log.keys())
    random.shuffle(keys)
    # Calculate the size of each clusters
    sub_dict_size = len(keys) // num_clus
    remainder = len(keys) % num_clus
    # Divide the keys into k sublists
    sub_lists = [keys[i * sub_dict_size: (i + 1) * sub_dict_size] for i in range(num_clus)]
    for i in range(remainder):
        sub_lists[i].append(keys[-(i + 1)])
    # Create clusters
    random_clusters = [dict((key, variant_log[key]) for key in sub_list) for sub_list in sub_lists]
    return random_clusters

def get_random_clusters(variant_log_input, num_clus):
    """
    Returns random clusters based on the number of variants.

    Parameters:
    - variant_log_input (dict): A dictionary containing variant log data.
    - num_clus (int): The number of clusters to create.

    Returns:
    - random_clusters (list): A list of dictionaries representing the random clusters. Each dictionary contains variant log data for a specific cluster.
    """
    variant_log = copy.deepcopy(variant_log_input)
    keys = list(variant_log.keys())
    clusters = [[] for _ in range(num_clus)]
    for key in keys:
        random_cluster = random.choice(clusters)  # Choose a random cluster
        random_cluster.append(key) 
    # Create clusters
    random_clusters = [{key: variant_log[key] for key in cluster} for cluster in clusters]
    return random_clusters

def cluster(log, num_clus, variant='equisized', outputshape='log'):
    """
    Cluster the given log data into a specified number of clusters.

    Parameters:
    - log (list): A list of log data.
    - num_clus (int): The number of clusters to create.
    - variant (str, optional): The variant of clustering to use. Default is 'equisized'.
    - outputshape (str, optional): The shape of the output. Default is 'log'.

    Returns:
    - clustered_data (list): A list of clustered log data. The shape of the output depends on the value of 'outputshape' parameter.

    Raises:
    - ValueError: If the value of 'outputshape' parameter is not 'log' or 'variant_log'.

    """
    variant_log_input = utils.get_variant_log(log)
    if variant == 'equisized':
        clusters_vl = get_random_clusters_equisized(variant_log_input, num_clus)
    else:
        clusters_vl = get_random_clusters(variant_log_input, num_clus)
    if outputshape == 'log':
        return [utils.filter_log_with_vl(log, cluster_vl) for cluster_vl in clusters_vl]
    elif outputshape == 'variant_log':
        return clusters_vl
    else:
        raise ValueError("Output has to be 'log' or 'variant_log'.")