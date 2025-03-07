import pm4py
import pandas as pd
import copy

from entroclus import entropic_clustering as entropic_clustering
from alternatives import frequency_based
from alternatives import random_clustering
from alternatives import trace2vec_based




import os


def test_all_methods(log_location, n_clusters):
    log = pm4py.read_xes('datasets/'+log_location)
    original_log = copy.deepcopy(log)

    methods = ['entropic_clustering', 'entropic_clustering_split', 'frequency_based', 'random_clustering']
    
    for method in methods:
        log = copy.deepcopy(original_log)
        clusters = get_clusters(log, n_clusters, method)

        # Create a directory to store cluster logs if it doesn't exist
        save_dir = f"experimental_results_elbow/clusters/{log_location.replace('.xes','').replace('.gz','')}/{method}/{n_clusters}"
        os.makedirs(save_dir, exist_ok=True)

        for cluster_index in range(0, n_clusters):
            cluster = clusters[cluster_index]
            # Save the individual cluster log to file
            cluster_log_location = f"{save_dir}/cluster_{cluster_index + 1}.xes"
            print("Saving logs to directory:", cluster_log_location)
            pm4py.write.write_xes(cluster, cluster_log_location)


def get_clusters(log, n_clus, method = 'entropic_clustering'):
    #for now just used ++ variant not random etc
    if method == 'entropic_clustering':
        clusters = entropic_clustering.cluster(log, n_clus, outputshape='log', variant='regular', initialization = '++', opt = 'trace')
    elif method == 'entropic_clustering_split':
        clusters = entropic_clustering.cluster(log, n_clus, outputshape='log', variant='split', initialization = '++', opt = 'trace')
    elif method == 'frequency_based':
        clusters = frequency_based.cluster(log, n_clus, outputshape='log', version='k-means++', distance='normalized')
    elif method == 'random_clustering':
        clusters = random_clustering.cluster(log, n_clus, variant='random', outputshape='log')
    elif method == 'trace2vec_based':
        clusters = trace2vec_based.cluster(log, n_clus, cluster_version='k-means++', distance='normalized', 
                                          vector_size=None, window_size=2, min_count=1, dm=0, epochs=200, outputshape='log')
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    return clusters




#for i in range(2, 10):
#    test_all_methods('Helpdesk.xes', i)

#for i in range(2, 10):
#    test_all_methods('RTFM.xes', i)

#for i in range(2, 10):
#    test_all_methods('BPIC13_incidents.xes', i)

#for i in range(2, 10):
#    test_all_methods('BPIC13_closedproblems.xes', i)


#for i in range(2, 10):
#    test_all_methods_no_alignments('Hospital_Billing.xes', i)

#for i in range(2, 10):
#    test_all_methods_no_alignments('BPIC15.xes', i)

#for i in range(2, 10):
#    test_all_methods_no_alignments('BPIC12.xes', i)

#for i in range(2, 10):
#    test_all_methods_no_alignments('Sepsis.xes', i)
