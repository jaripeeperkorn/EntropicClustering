from evaluation import metrics
import pm4py
import pandas as pd
import copy



def evaluate_all_methods(logname, n_clusters):
    #! actitrac to add trhough ProM manually
    #todo add automatic java calls for actitrac or implement in python
    methods = ['trace2vec_based',
               'entropic_clustering_++', 'entropic_clustering_++_norm', 'entropic_clustering_randominit', 
               'entropic_clustering_split_++', 'entropic_clustering_split_++_norm', 'entropic_clustering_split_randominit', 
               'frequency_based', 'random_clustering',
               'actitrac_freq', 'actitrac_dist']
    
    columns =  ['method', 'cluster', 'num_traces', 
                'replay_fitness', 'replay_precision', 
                'align_fitness', 'align_precision', 
                'ER', 
                'graph_density', 'graph_entropy']
    
    df = pd.DataFrame(columns=columns)
    log = pm4py.read_xes('datasets/'+logname)

    baseline_PN = metrics.get_non_stochastic_metrics(log)
    baseline_stoch = metrics.get_stochastic_metrics(log)
    baseline_graph_simplicity = metrics.get_graph_simplicity_metrics(log)

    baseline_results = ['full_log', 0, len(log), 
                        baseline_PN['replay_fitness'], baseline_PN['replay_precision'], 
                        baseline_PN['align_fitness'], baseline_PN['align_precision'], 
                        baseline_stoch['ER'], 
                        baseline_graph_simplicity['graph_density'], baseline_graph_simplicity['graph_entropy']]
    
    df.loc[df.shape[0]] = baseline_results
    print(df)

    for method in methods:
        total_traces = 0
        weighted_sums = {col: 0 for col in columns[3:]}  # Initialize dictionary to store weighted sums of metrics used for avergaging
        for clus in range(0, n_clusters):
            curr_cluster_loc = 'experimental_results_one_cluster_size/clusters/'+logname.replace('.xes','').replace('.gz','')+'/'+method+'/cluster_'+str(clus+1)+'.xes'
            curr_cluster = pm4py.read_xes(curr_cluster_loc)
            num_traces = len(curr_cluster)
            curr_PN = metrics.get_non_stochastic_metrics(curr_cluster)
            curr_stoch = metrics.get_stochastic_metrics(curr_cluster)
            curr_graph_simplicity = metrics.get_graph_simplicity_metrics(curr_cluster)
            curr_results = [method, clus+1, len(curr_cluster), 
                            curr_PN['replay_fitness'], curr_PN['replay_precision'], 
                            curr_PN['align_fitness'], curr_PN['align_precision'], 
                            curr_stoch['ER'], 
                            curr_graph_simplicity['graph_density'], curr_graph_simplicity['graph_entropy']]
            df.loc[df.shape[0]] = curr_results

            # Accumulate weighted sums for each metric
            total_traces += num_traces
            for col in columns[3:]:
                weighted_sums[col] += num_traces * curr_results[columns.index(col)]
            
        # Calculate averages
        averages = [method, 'average', total_traces]
        for col in columns[3:]:
            averages.append(weighted_sums[col] / total_traces)
        df.loc[df.shape[0]] = averages
        print(df)
    results_loc = f'experimental_results_one_cluster_size/results/{logname.replace('.xes','').replace('.gz','')}_results.csv'
    df.to_csv(results_loc, index=False)
