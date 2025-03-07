from evaluation import metrics
import pm4py
import pandas as pd

def test_all_methods(log_location, n_clus):
    log = pm4py.read_xes('datasets/'+log_location)
    print(log)
    baseline_stoch = metrics.get_stochastic_metrics(log)
    baseline_graph_simplicity = metrics.get_graph_simplicity_metrics(log)

    columns = ['method', 'cluster', 'num_traces', 'ER', 'graph_density', 'graph_entropy']

    df = pd.DataFrame(columns=columns)

    baseline_results = ['full_log', 0, log["case:concept:name"].nunique(), baseline_stoch['ER'], baseline_graph_simplicity['graph_density'], baseline_graph_simplicity['graph_entropy']]
    print(baseline_results)
    
    df.loc[df.shape[0]] = baseline_results
    print(df)

    methods = ['entropic_clustering', 'entropic_clustering_split', 'frequency_based', 'random_clustering']

    for method in methods:
        save_dir = f"experimental_results_elbow/clusters/{log_location.replace('.xes','').replace('.gz','')}/{method}/{n_clus}"
        total_traces = 0
        weighted_sums = {col: 0 for col in columns[3:]}  # Initialize dictionary to store weighted sums of metrics used for avergaging
        for cluster_index in range(0,n_clus):
            cluster_log_location = f"{save_dir}/cluster_{cluster_index + 1}.xes"
            cluster = pm4py.read_xes(cluster_log_location)
            num_traces = cluster["case:concept:name"].nunique()
            results_stochastic = metrics.get_stochastic_metrics(cluster)
            results_graph_simplicity = metrics.get_graph_simplicity_metrics(cluster)
            cluster_results = [method, cluster_index, num_traces,
                               results_stochastic['ER'], results_graph_simplicity['graph_density'], results_graph_simplicity['graph_entropy']]
            df.loc[df.shape[0]] = cluster_results

            # Accumulate weighted sums for each metric
            total_traces += num_traces
            weighted_sums['ER'] += results_stochastic['ER'] * num_traces
            weighted_sums['graph_density'] += results_graph_simplicity['graph_density'] * num_traces
            weighted_sums['graph_entropy'] += results_graph_simplicity['graph_entropy'] * num_traces
        # Now compute the weighted average
        weighted_average = {key: weighted_sums[key] / total_traces for key in weighted_sums}

        # Add the method and average row to the dataframe
        weighted_average['method'] = method
        weighted_average['cluster'] = 'weighted_average'
        weighted_average['num_traces'] = total_traces
        weighted_average = pd.Series(weighted_average)
        df.loc[df.shape[0]] = weighted_average

    logname = log_location.replace('.xes','').replace('.gz','')
    results_loc = f'experimental_results_elbow/results/{logname}_{str(n_clus)}_results.csv'
    df.to_csv(results_loc, index=False)


for i in range(2, 10):
    test_all_methods('Helpdesk.xes', i)

for i in range(2, 10):
    test_all_methods('RTFM.xes', i)

for i in range(2, 10):
    test_all_methods('BPIC13_incidents.xes', i)

for i in range(2, 10):
    test_all_methods('BPIC13_closedproblems.xes', i)


for i in range(2, 10):
    test_all_methods('Hospital_Billing.xes', i)

for i in range(2, 10):
    test_all_methods('BPIC15.xes', i)

for i in range(2, 10):
    test_all_methods('BPIC12.xes', i)

for i in range(2, 10):
    test_all_methods('Sepsis.xes', i)
