from entroclus import utils as utils
from entroclus import entropic_relevance as entropic_relevance

import random
import copy


def pairwise_ER(trace1, trace2, norm=False):
    """
    Calculate the pairwise Entropic Relevance (ER) value between two traces.

    Parameters:
    - trace1 (list): The first trace.
    - trace2 (list): The second trace.

    Returns:
    - float: The pairwise ER value between the two traces.
    """
    log = {tuple(trace1): 1, tuple(trace2): 1}
    activity_counts, edge_counts = utils.get_dfg(log)
    if norm == True:
        ER = entropic_relevance.get_ER_normalized(log, activity_counts, edge_counts)
    else:
        ER = entropic_relevance.get_ER(log, activity_counts, edge_counts)
    return ER



def get_distances_to_closest_seed(variant_log, seeds, distance_function="ER"):
    """
    Calculate the distances between each variant in the variant_log and the closest seed in the seeds list.

    Args:
        variant_log (list): A list of variants.
        seeds (list): A list of seeds.
        distance_function (str, optional): The distance function to use. Defaults to "ER".

    Returns:
        minimal_distances (dictionary): A disctionary that for each variants contains the minimal distances between it and the closest seed.
    """
    minimal_distances = {}
    #Compare distance for each variant to each seed (cluster), select minimum, add to a dictionary of distances
    if distance_function == 'ER':
        for variant in variant_log:
            min_distance = min(pairwise_ER(variant, seed, norm=False) for seed in seeds)
            minimal_distances[variant] = min_distance
    elif distance_function == 'ER_norm':
        for variant in variant_log:
            min_distance = min(pairwise_ER(variant, seed, norm=True) for seed in seeds)
            minimal_distances[variant] = min_distance
    return minimal_distances


def sample_seed_distance_based(distances_dict):
    """
    Selects a variant based on the distances provided in the distances_dict.

    Parameters:
    - distances_dict (dict): A dictionary where the keys represent the variants and the values represent the distances.

    Returns:
    - selected_variant: The variant selected based on the distances.
    """
    keys_variants = list(distances_dict.keys())
    weights = [distance**2 for distance in distances_dict.values()]
    selected_variant = random.choices(keys_variants, weights=weights, k=1)[0]
    return selected_variant


def get_seeds(variant_log, num_clusters, version= "++"):
    """
    Get seeds for clustering based on the variant log.

    Args:
        variant_log (dict): A dictionary representing the variant log.
        num_clusters (int): The number of clusters/seeds to generate.
        version (str, optional): The version of seed selection. Defaults to "++".

    Returns:
        list: A list of seeds for clustering.

    Notes:
        - The variant log should be a dictionary where the keys represent the variants and the values the counts.
        - The version parameter can take the following values:
            - "++": kmeans++ based seed selection.
            - "++_norm": kmeans++ based seed selection with normalized distances.
            - "random": random seed selection.
    """
    temp_variant_log = copy.deepcopy(variant_log)
    seeds = []
    if version == "++": #kmeans++ based seed selection
        #add first seed randomly
        first_seed = random.choice(list(temp_variant_log.keys()))
        seeds.append(copy.deepcopy(first_seed))
        del temp_variant_log[first_seed]
        while len(seeds) < num_clusters:
            #use inverse of the distance to closest seed to sample next seed
            new_seed = sample_seed_distance_based(get_distances_to_closest_seed(temp_variant_log, seeds, distance_function="ER"))
            seeds.append(copy.deepcopy(new_seed))
            del temp_variant_log[new_seed]
    elif version == "++_norm":
        #add first seed randomly
        first_seed = random.choice(list(temp_variant_log.keys()))
        seeds.append(copy.deepcopy(first_seed))
        del temp_variant_log[first_seed]
        while len(seeds) < num_clusters:
            #use inverse of the distance to closest seed to sample next seed
            new_seed = sample_seed_distance_based(get_distances_to_closest_seed(temp_variant_log, seeds, distance_function="ER_norm"))
            seeds.append(copy.deepcopy(new_seed))
            del temp_variant_log[new_seed]
    elif version == "random":
        seeds = random.sample(list(temp_variant_log.keys()), num_clusters)
    else:
        raise ValueError("verion has to be '++' or '++_norm' or 'random'")
    return seeds

def intialize_clusters(variant_log, seeds):
    """
    Initialize clusters based on a variant log and a list of seed variants.

    Parameters:
    - variant_log (list): A list of variants.
    - seeds (list): A list of seed variants.

    Returns:
    - clusters (list): A list of clusters, where each cluster is a list of variants.
    - variant_log (list): The updated variant log after removing the seed variants.
    """
    vl_temp = copy.deepcopy(variant_log)
    clusters = []
    for seed in seeds:
        clusters.append({seed:copy.deepcopy(vl_temp[seed])})
        del vl_temp[seed]
    return clusters, vl_temp
