from entroclus import entropic_clustering as entropic_clustering
from alternatives import frequency_based
from alternatives import random_clustering
from alternatives import trace2vec_based

from evaluation import metrics

import pm4py

def get_clusters(log, n_clus, type = 'entropic_clustering'):
    #for now just used ++ variant not random etc
    if type == 'entropic_clustering':
        clusters = entropic_clustering.cluster(log, n_clus, outputshape='log', variant='regular', initialization = '++', opt = 'trace')
    if type == 'entropic_clustering_split':
        clusters = entropic_clustering.cluster(log, n_clus, outputshape='log', variant='split', initialization = '++', opt = 'trace')
    if type == 'frequency_based':
        clusters = frequency_based.cluster(log, n_clus, outputshape='log', version='k-means++', distance='normalized')
    if type == 'random_clustering':
        clusters = random_clustering.cluster(log, n_clus, variant='random', outputshape='log')
    if type == 'trace2vec_based':
        clusters = trace2vec_based.cluster(log, n_clus, cluster_version='k-means++', distance='normalized', 
                                          vector_size=None, window_size=2, min_count=0, dm=0, epochs=200, outputshape='log')
    return clusters

def do_one_experiment(log, n_clus, type = 'entropic_clustering'):
    baseline = metrics.get_non_stochastic_metrics(log)
    print("Baseline: ", baseline)
    baseline_stoch = metrics.get_stochastic_metrics(log)
    print("Baseline: ", baseline_stoch)
    baseline_graph_simplicity = metrics.get_graph_simplicity_metrics(log)
    print("Baseline: ", baseline_graph_simplicity)
    clusters = get_clusters(log, n_clus, type)
    for i in range(0,n_clus):
        print("Cluster: ", metrics.get_non_stochastic_metrics(clusters[i]))
        print("Cluster: ", metrics.get_stochastic_metrics(clusters[i]))
        print("Cluster: ", metrics.get_graph_simplicity_metrics(clusters[i]))

def test_all_types(log, n_clus):
    types = ['entropic_clustering', 'entropic_clustering_split', 'frequency_based', 'random_clustering', 'trace2vec_based']
    for type in types:
        print("########################---", type, "---########################")
        do_one_experiment(log, n_clus, type = type)
    full_results = metrics.get_non_stochastic_metrics(log)
    print("Full log: ", full_results)
    full_results_stoch = metrics.get_stochastic_metrics(log)
    print("Full log: ", full_results_stoch)

log = pm4py.read_xes('Helpdesk.xes')
do_one_experiment(log, 3, type = 'entropic_clustering')

#test_all_types(log, 3)