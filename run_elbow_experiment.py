import pm4py
import pandas as pd
import copy

from entroclus import entropic_clustering as entropic_clustering
from alternatives import frequency_based
from alternatives import random_clustering
from alternatives import trace2vec_based

from evaluation import metrics


import os


def test_all_methods(log_location, n_clusters):
    log = pm4py.read_xes('datasets/'+log_location)
    original_log = copy.deepcopy(log)
    baseline = metrics.get_non_stochastic_metrics(log)
    baseline_stoch = metrics.get_stochastic_metrics(log)
    baseline_graph_simplicity = metrics.get_graph_simplicity_metrics(log)

    columns = ['method', 'cluster', 'num_traces', 'replay_fitness', 'replay_precision', 'align_fitness', 'align_precision', 'ER', 'tade_fitness', 'graph_density', 'graph_entropy']
    df = pd.DataFrame(columns=columns)

    baseline_results = ['full_log', 0, len(log),
                        baseline['replay_fitness'], baseline['replay_precision'], baseline['align_fitness'], baseline['align_precision'], 
                        baseline_stoch['ER'], baseline_stoch['tade_fitness'], 
                        baseline_graph_simplicity['graph_density'], baseline_graph_simplicity['graph_entropy']]
    df.loc[df.shape[0]] = baseline_results
    print(df)

    #methods = ['entropic_clustering', 'entropic_clustering_split', 'frequency_based', 'random_clustering', 'trace2vec_based']
    methods = ['entropic_clustering', 'entropic_clustering_split', 'frequency_based', 'random_clustering']
    
    for method in methods:
        log = copy.deepcopy(original_log)
        clusters = get_clusters(log, n_clusters, method)

        total_traces = 0
        weighted_sums = {col: 0 for col in columns[3:]}  # Initialize dictionary to store weighted sums of metrics used for avergaging

        # Create a directory to store cluster logs if it doesn't exist
        save_dir = f"experimental_results_clusters/{log_location.replace('.xes','').replace('.gz','')}/{method}/{n_clusters}"
        os.makedirs(save_dir, exist_ok=True)

        for cluster_index in range(0, n_clusters):
            cluster = clusters[cluster_index]
            num_traces = len(cluster)
            results_general = metrics.get_non_stochastic_metrics(cluster)
            results_stochastic = metrics.get_stochastic_metrics(cluster)
            results_graph_simplicity = metrics.get_graph_simplicity_metrics(cluster)
            cluster_results = [method, cluster_index, num_traces,
                               results_general['replay_fitness'], results_general['replay_precision'], results_general['align_fitness'], results_general['align_precision'], 
                               results_stochastic['ER'], results_stochastic['tade_fitness'], 
                               results_graph_simplicity['graph_density'], results_graph_simplicity['graph_entropy']]
            df.loc[df.shape[0]] = cluster_results

            # Save the individual cluster log to file
            cluster_log_location = f"{save_dir}/cluster_{cluster_index + 1}.xes"
            print("Saving logs to directory:", cluster_log_location)
            pm4py.write.write_xes(cluster, cluster_log_location)

            # Accumulate weighted sums for each metric
            total_traces += num_traces
            weighted_sums['replay_fitness'] += results_general['replay_fitness'] * num_traces
            weighted_sums['replay_precision'] += results_general['replay_precision'] * num_traces
            weighted_sums['align_fitness'] += results_general['align_fitness'] * num_traces
            weighted_sums['align_precision'] += results_general['align_precision'] * num_traces
            weighted_sums['ER'] += results_stochastic['ER'] * num_traces
            weighted_sums['tade_fitness'] += results_stochastic['tade_fitness'] * num_traces
            weighted_sums['graph_density'] += results_graph_simplicity['graph_density'] * num_traces
            weighted_sums['graph_entropy'] += results_graph_simplicity['graph_entropy'] * num_traces

        # Now compute the weighted average
        weighted_average = {key: weighted_sums[key] / total_traces for key in weighted_sums}

        # Add the method and average row to the dataframe
        weighted_average['method'] = method
        weighted_average['cluster'] = 'weighted_average'
        weighted_average = pd.Series(weighted_average)
        df.loc[df.shape[0]] = weighted_average

    logname = log_location.replace('.xes','').replace('.gz','')
    results_loc = f'experimental_results/{logname}_{str(n_clusters)}_results.csv'
    df.to_csv(results_loc, index=False)



    
def test_all_methods_no_alignments(log_location, n_clusters):
    log = pm4py.read_xes('datasets/'+log_location)
    original_log = copy.deepcopy(log)
    baseline = metrics.get_non_stochastic_metrics_no_alignments(log)
    baseline_stoch = metrics.get_stochastic_metrics(log)
    baseline_graph_simplicity = metrics.get_graph_simplicity_metrics(log)

    columns = ['method', 'cluster', 'num_traces', 'replay_fitness', 'replay_precision', 'ER', 'tade_fitness', 'graph_density', 'graph_entropy']
    df = pd.DataFrame(columns=columns)

    baseline_results = ['full_log', 0, len(log),
                        baseline['replay_fitness'], baseline['replay_precision'], 
                        baseline_stoch['ER'], baseline_stoch['tade_fitness'], 
                        baseline_graph_simplicity['graph_density'], baseline_graph_simplicity['graph_entropy']]
    df.loc[df.shape[0]] = baseline_results
    print(df)

    #methods = ['entropic_clustering', 'entropic_clustering_split', 'frequency_based', 'random_clustering', 'trace2vec_based']
    methods = ['entropic_clustering', 'entropic_clustering_split', 'frequency_based', 'random_clustering']

    
    for method in methods:
        log = copy.deepcopy(original_log)
        clusters = get_clusters(log, n_clusters, method)

        total_traces = 0
        weighted_sums = {col: 0 for col in columns[3:]}  # Initialize dictionary to store weighted sums of metrics used for avergaging

        # Create a directory to store cluster logs if it doesn't exist
        save_dir = f"experimental_results_clusters/{log_location.replace('.xes','').replace('.gz','')}/{method}/{n_clusters}"
        os.makedirs(save_dir, exist_ok=True)

        for cluster_index in range(0, n_clusters):
            cluster = clusters[cluster_index]
            num_traces = len(cluster)
            results_general = metrics.get_non_stochastic_metrics_no_alignments(cluster)
            results_stochastic = metrics.get_stochastic_metrics(cluster)
            results_graph_simplicity = metrics.get_graph_simplicity_metrics(cluster)
            cluster_results = [method, cluster_index, num_traces,
                               results_general['replay_fitness'], results_general['replay_precision'], 
                               results_stochastic['ER'], results_stochastic['tade_fitness'], 
                               results_graph_simplicity['graph_density'], results_graph_simplicity['graph_entropy']]
            df.loc[df.shape[0]] = cluster_results

            # Save the individual cluster log to file
            cluster_log_location = f"{save_dir}/cluster_{cluster_index + 1}.xes"
            print("Saving logs to directory:", cluster_log_location)
            pm4py.write.write_xes(cluster, cluster_log_location)

            # Accumulate weighted sums for each metric
            total_traces += num_traces
            weighted_sums['replay_fitness'] += results_general['replay_fitness'] * num_traces
            weighted_sums['replay_precision'] += results_general['replay_precision'] * num_traces
            weighted_sums['ER'] += results_stochastic['ER'] * num_traces
            weighted_sums['tade_fitness'] += results_stochastic['tade_fitness'] * num_traces
            weighted_sums['graph_density'] += results_graph_simplicity['graph_density'] * num_traces
            weighted_sums['graph_entropy'] += results_graph_simplicity['graph_entropy'] * num_traces

        # Now compute the weighted average
        weighted_average = {key: weighted_sums[key] / total_traces for key in weighted_sums}

        # Add the method and average row to the dataframe
        weighted_average['method'] = method
        weighted_average['cluster'] = 'weighted_average'
        weighted_average = pd.Series(weighted_average)
        df.loc[df.shape[0]] = weighted_average

    logname = log_location.replace('.xes','').replace('.gz','')
    results_loc = f'experimental_results/{logname}_{str(n_clusters)}_results.csv'
    df.to_csv(results_loc, index=False)
    


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


#! Should we also consider ++norm and random initialization? Maybe only for one number of clusters then...
#! Also probably best to remove the TADE fitness?

#for i in range(2, 10):
#    test_all_methods('Helpdesk.xes', i)

#for i in range(2, 10):
#    test_all_methods('RTFM.xes', i)

#for i in range(2, 10):
#    test_all_methods('BPIC13_incidents.xes', i)

#for i in range(2, 10):
#    test_all_methods('BPIC13_closedproblems.xes', i)

#! from here onwards we did not include alignments as they are too computationally expensive
#! also no trace2vec → there were some issues

#for i in range(2, 10):
#    test_all_methods_no_alignments('Hospital_Billing.xes', i)

#for i in range(2, 10):
#    test_all_methods_no_alignments('BPIC15.xes', i)

for i in range(2, 10):
    test_all_methods_no_alignments('BPIC12.xes', i)

for i in range(2, 10):
    test_all_methods_no_alignments('Sepsis.xes', i)
