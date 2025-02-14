from entroclus import utils as utils
from entroclus import entropic_relevance as entropic_relevance
from entroclus import entropic_clustering_utils as entropic_clustering_utils
import copy

def add_and_remove_variant(clusters, variant_log, variant, occurrence, cluster_index):
    """
    Add a variant to a specific cluster and remove it from the variant log.

    Parameters:
    - clusters (list of dictionaries): The list of clusters, where each cluster is represented as a dictionary.
    - variant_log (dictionary): The variant log, where each variant is a key and its occurrence is the value.
    - variant (any): The variant to be added to the cluster.
    - occurrence (int): The occurrence of the variant.
    - cluster_index (int): The index of the cluster where the variant should be added.

    Returns:
    - clusters (list of dictionaries): The updated list of clusters after adding the variant to the specified cluster.
    - variant_log (dictionary): The updated variant log after removing the variant.
    """
    clusters[cluster_index].update({variant:occurrence})
    del variant_log[variant]
    return clusters, variant_log

def entropic_clustering_VL(variant_log_input, num_clusters, initialization = '++', opt = 'trace'):
    """
    Perform entropic clustering on a given variant log.

    Parameters:
    - variant log (dictionary): The variant log to be clustered.
    - num_clusters (int): The number of clusters to create.
    - initialization (str, optional): The initialization method for selecting initial seeds. Defaults to '++'.
    - opt (str, optional): The optimization method for calculating ER. Can be 'full_cluster' or 'trace'. Defaults to 'trace'.

    Returns:
    - list: A list of clusters, where each cluster is a dictionary containing variants and their occurrences.

    Then, it selects initial seeds for clustering using the 'get_seeds' function from the 'entropic_clustering_utils' module. 
    Next, it initializes the clusters and variant log using the 'initialize_clusters' function from the 'entropic_clustering_utils' module.

    After initialization, the function iterates through each variant in the variant log and calculates the best cluster to add the variant to based on 
    the lowest ER value. The ER value is calculated using the 'get_ER' function from the 'entropic_relevance' module. The variant is then added to the 
    best cluster and the variant log is updated accordingly. The DFG of the best cluster is also updated using the 'update_dfg' function from the 
    'utils' module.

    Finally, the function returns the list of clusters.
    """
    variant_log = copy.deepcopy(variant_log_input)
    seeds = entropic_clustering_utils.get_seeds(variant_log, num_clusters, version=initialization)
    print("seeds obtained")
    clusters, variant_log = entropic_clustering_utils.intialize_clusters(variant_log, seeds)
    variant_log_dum = copy.deepcopy(variant_log) #needed because within the loop, the size of the dictionary can not change
    dfgs_clusters = [utils.get_dfg(clus) for clus in clusters]
    for variant, occurrence in variant_log_dum.items():
        best_ER = 99999.0
        best_cluster_index = 0
        for k in range(0, len(clusters)):
            #add variant to each cluster and calculate ER
            curr_clus = copy.deepcopy(clusters[k])
            curr_clus.update({variant:occurrence})
            #print(copy.deepcopy(dfgs_clusters[k]))
            curr_activity_counts, curr_edge_counts = copy.deepcopy(dfgs_clusters[k])[0], copy.deepcopy(dfgs_clusters[k])[1]
            curr_activity_counts, curr_edge_counts = utils.update_dfg(curr_activity_counts, curr_edge_counts, variant, occurrence)
            if opt == 'full_cluster':
                curr_ER = entropic_relevance.get_ER(curr_clus, curr_activity_counts, curr_edge_counts)
            elif opt == 'trace':
                tracelog = {variant: occurrence}
                curr_ER = entropic_relevance.get_ER(tracelog, curr_activity_counts, curr_edge_counts)
            else:
                raise ValueError("opt has to be 'full_cluster' or 'trace'")
            if curr_ER < best_ER:
                best_ER = curr_ER
                best_cluster_index = k
        #actually add variant to cluster with optimal ER (lowest)
        clusters, variant_log = add_and_remove_variant(clusters, variant_log, variant, occurrence, best_cluster_index)
        #update dfg of cluster
        dfgs_clusters[best_cluster_index] = utils.update_dfg(dfgs_clusters[best_cluster_index][0], dfgs_clusters[best_cluster_index][1], variant, occurrence)
    return clusters

def entropic_clustering(log, num_clusters, initialization = '++', opt = 'trace'):
    """
    Perform entropic clustering on a given event log.

    Parameters:
    - log (pm4py.objects.log.obj.EventLog): The event log to be clustered.
    - num_clusters (int): The number of clusters to create.
    - initialization (str, optional): The initialization method for selecting initial seeds. Defaults to '++'.
    - opt (str, optional): The optimization method for calculating ER. Can be 'full_cluster' or 'trace'. Defaults to 'trace'.

    Returns:
    - list: A list of clusters, where each cluster is a dictionary containing variants and their occurrences.

    This function performs entropic clustering on a given event log. It first obtains the variant log using the 'get_variant_log' function from the 'utils' 
    module. 

    """
    variant_log_input = utils.get_variant_log(log)
    print("variant_log obtained")
    return entropic_clustering_VL(variant_log_input, num_clusters, initialization = initialization, opt = opt)

def get_worst_cluster_and_remove(clusters):
    """
    Get the worst cluster from a list of clusters based on their Entropic Relevance (ER) values and remove it from the list.

    Parameters:
    - clusters (list): A list of variant log dictionaries representing clusters.

    Returns:
    - tuple: A tuple containing the worst cluster and the updated list of clusters.
    """
    current_highest = 0.0 #ER needs to be minimized
    for c in clusters: #c is a variant log dictionaries, clusters a list of these
        c_activity_counts, c_edge_counts = utils.get_dfg(c)
        c_ER = entropic_relevance.get_ER(c, c_activity_counts, c_edge_counts)
        if c_ER > current_highest:
            current_highest = c_ER
            worst = c
    clusters.remove(worst)
    return worst, clusters

def add_clusters(clusters, new_clusters):
    """
    Add clusters to an existing list of clusters.

    Parameters:
    - clusters (list): The existing list of clusters.
    - new_clusters (list): The new clusters to be added.

    Returns:
    - list: The updated list of clusters.
    """
    clusters_updated = clusters
    for c in new_clusters:
        clusters_updated.append(c)
    return clusters_updated

def entropic_clustering_split_VL(variant_log_input, num_clusters, initialization = '++', opt = 'trace'):
    """
    Hierarchical variant.
    Perform entropic clustering on a given event log and split clusters to create the desired number of clusters.

    Parameters:
    - log (pm4py.objects.log.obj.EventLog): The event log to be clustered.
    - num_clusters (int): The number of clusters to create.
    - initialization (str, optional): The initialization method for selecting initial seeds. Defaults to '++'.
    - opt (str, optional): The optimization method for calculating ER. Can be 'full_cluster' or 'trace'. Defaults to 'trace'.

    Returns:
    - list: A list of clusters, where each cluster is a dictionary containing variants and their occurrences.
"""
    variant_log = copy.deepcopy(variant_log_input)
    clusters = entropic_clustering_VL(variant_log, 2, initialization, opt)
    if num_clusters>2:
        i = 2
        while i<num_clusters:
            to_be_split_cluster, clusters = get_worst_cluster_and_remove(copy.deepcopy(clusters))
            new_clusters = entropic_clustering_VL(to_be_split_cluster, 2, initialization, opt)
            clusters = add_clusters(clusters, new_clusters)
            i += 1
    return clusters


def entropic_clustering_split(log, num_clusters, initialization = '++', opt = 'trace'):
    """
    Hierarchical variant.
    Perform entropic clustering on a given event log and split clusters to create the desired number of clusters.

    Parameters:
    - log (pm4py.objects.log.obj.EventLog): The event log to be clustered.
    - num_clusters (int): The number of clusters to create.
    - initialization (str, optional): The initialization method for selecting initial seeds. Defaults to '++'.
    - opt (str, optional): The optimization method for calculating ER. Can be 'full_cluster' or 'trace'. Defaults to 'trace'.

    Returns:
    - list: A list of clusters, where each cluster is a dictionary containing variants and their occurrences.
"""

    clusters = entropic_clustering(log, 2, initialization, opt)
    if num_clusters>2:
        i = 2
        while i<num_clusters:
            to_be_split_cluster, clusters = get_worst_cluster_and_remove(copy.deepcopy(clusters))
            new_clusters = entropic_clustering_VL(to_be_split_cluster, 2, initialization, opt)
            clusters = add_clusters(clusters, new_clusters)
            i += 1
    return clusters
