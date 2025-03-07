

import os

import pm4py
import pandas as pd
import copy

from entroclus import entropic_clustering as entropic_clustering
from alternatives import frequency_based
from alternatives import random_clustering
from alternatives import trace2vec_fixed

def get_clusters(log, n_clus, method = 'entropic_clustering'):
    #for now just used ++ variant not random etc
    if method == 'entropic_clustering_++':
        clusters = entropic_clustering.cluster(log, n_clus, outputshape='log', variant='regular', initialization = '++', opt = 'trace')
    elif method == 'entropic_clustering_++_norm':
        clusters = entropic_clustering.cluster(log, n_clus, outputshape='log', variant='regular', initialization = '++_norm', opt = 'trace')
    elif method == 'entropic_clustering_randominit':
        clusters = entropic_clustering.cluster(log, n_clus, outputshape='log', variant='regular', initialization = 'random', opt = 'trace')
    elif method == 'entropic_clustering_split_++':
        clusters = entropic_clustering.cluster(log, n_clus, outputshape='log', variant='split', initialization = '++', opt = 'trace')
    elif method == 'entropic_clustering_split_++_norm':
        clusters = entropic_clustering.cluster(log, n_clus, outputshape='log', variant='split', initialization = '++_norm', opt = 'trace')
    elif method == 'entropic_clustering_split_randominit':
        clusters = entropic_clustering.cluster(log, n_clus, outputshape='log', variant='split', initialization = 'random', opt = 'trace')
    elif method == 'frequency_based':
        clusters = frequency_based.cluster(log, n_clus, outputshape='log', version='k-means++', distance='normalized')
    elif method == 'random_clustering':
        clusters = random_clustering.cluster(log, n_clus, variant='random', outputshape='log')
    elif method == 'trace2vec_based':
        clusters = trace2vec_fixed.cluster(log, n_clus, cluster_version='k-means++', distance='normalized', vector_size=None, window_size=2, min_count=1, dm=0, epochs=200, outputshape='log')
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    return clusters

def get_clusters_all_methods(log_location, n_clusters):
    log = pm4py.read_xes('datasets/'+log_location)
    original_log = copy.deepcopy(log)

    methods = ['trace2vec_based',
               'entropic_clustering_++', 'entropic_clustering_++_norm', 'entropic_clustering_randominit', 
               'entropic_clustering_split_++', 'entropic_clustering_split_++_norm', 'entropic_clustering_split_randominit', 
               'frequency_based', 'random_clustering']

    for method in methods:
        log = copy.deepcopy(original_log)
        clusters = get_clusters(log, n_clusters, method)

        # Create a directory to store cluster logs if it doesn't exist
        save_dir = f"experimental_results_one_cluster_size/clusters/{log_location.replace('.xes','').replace('.gz','')}/{method}"
        os.makedirs(save_dir, exist_ok=True)

        for cluster_index in range(0, n_clusters):
            cluster = clusters[cluster_index]
            cluster_log_location = f"{save_dir}/cluster_{cluster_index + 1}.xes"
            print("Saving logs to directory:", cluster_log_location)
            pm4py.write.write_xes(cluster, cluster_log_location)
    

#when ER doesn't go down we use graph entropy for both BPIC13 and BPIC12

elbow_points = {'Helpdesk': 4,
                'RTFM': 4,
                'BPIC13_incidents': 5,
                'BPIC13_closedproblems': 5,
                'BPIC15': 4,
                'Hospital_Billing': 5,
                'BPIC12': 4,
                'Sepsis': 6}           

#to_run = ['Helpdesk']
to_run = ['Helpdesk', 'RTFM', 'BPIC13_incidents', 'BPIC13_closedproblems', 'BPIC15', 'Hospital_Billing', 'BPIC12', 'Sepsis']


for log in to_run:
    get_clusters_all_methods(log+'.xes', elbow_points[log])