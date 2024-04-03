import pm4py
import entroclus.utils as utils
import entroclus.entropic_relevance as entropic_relevance
import pandas as pd

def get_scores_original_log(loglocation):
    log = pm4py.read_xes(loglocation)
    print(len(log))
    columns = ['fitness_tbr', 'precision_tbr', 'fitness_alignments', 'precision_alignments', 'ER']
    df = pd.DataFrame(columns=columns)
    net, im, fm = pm4py.discover_petri_net_inductive(log, activity_key='concept:name', case_id_key='case:concept:name', timestamp_key='time:timestamp')
    fitness_tbr = pm4py.fitness_token_based_replay(log, net, im, fm, activity_key='concept:name', case_id_key='case:concept:name', timestamp_key='time:timestamp')['log_fitness']
    precision_tbr = pm4py.precision_token_based_replay(log, net, im, fm, activity_key='concept:name', case_id_key='case:concept:name', timestamp_key='time:timestamp')
    fitness_alignments = pm4py.fitness_alignments(log, net, im, fm, activity_key='concept:name', case_id_key='case:concept:name', timestamp_key='time:timestamp')['log_fitness']
    precision_alignments = pm4py.precision_alignments(log, net, im, fm, activity_key='concept:name', case_id_key='case:concept:name', timestamp_key='time:timestamp')
    variant_log = utils.get_variant_log(log) 
    activity_counts, edge_counts = utils.get_dfg(variant_log)
    ER = entropic_relevance.get_ER(variant_log, activity_counts, edge_counts)
    results = [fitness_tbr, precision_tbr, fitness_alignments, precision_alignments, ER]
    df.loc[len(df)] = results
    print(df)
    filename = loglocation.split("/")[-1]
    filename = filename.replace('.xes', '')
    filename = filename.replace('.gz', '')
    df.to_csv('Experiment/Results/'+filename+'/Results_full_log.csv', index=False)

def get_scores_clusters(filename, n_clus):
    location_clusters = 'Experiment/Clusters/'+filename+'/'+str(n_clus)+'/'
    methods_list = ['ER_trace_only_pp', 'ER_trace_only_ppnorm', 'ER_trace_only_random','ER_trace_only_pp_split', 'ER_trace_only_ppnorm_split', 'ER_trace_only_random_split', 'frequency_normalized', 'random', 'trace2vec','Actitrac_freq', 'Actitrac_dist']
    for method in methods_list:
        #columns = ['cluster', 'size', 'fitness_tbr', 'precision_tbr', 'fitness_alignments', 'precision_alignments', 'ER']
        columns = ['cluster','size', 'fitness_tbr', 'precision_tbr', 'ER']
        total_results = [0.0]*len(columns)
        total_size = 0
        df = pd.DataFrame(columns=columns)
        for i in range(1,n_clus+1):
            log = pm4py.read_xes(location_clusters+method+'/Cluster'+str(i)+'.xes')
            size_cluster = len(log)
            net, im, fm = pm4py.discover_petri_net_inductive(log, activity_key='concept:name', case_id_key='case:concept:name', timestamp_key='time:timestamp')
            fitness_tbr = pm4py.fitness_token_based_replay(log, net, im, fm, activity_key='concept:name', case_id_key='case:concept:name', timestamp_key='time:timestamp')['log_fitness']
            precision_tbr = pm4py.precision_token_based_replay(log, net, im, fm, activity_key='concept:name', case_id_key='case:concept:name', timestamp_key='time:timestamp')
            #fitness_alignments = pm4py.fitness_alignments(log, net, im, fm, activity_key='concept:name', case_id_key='case:concept:name', timestamp_key='time:timestamp')['log_fitness']
            #precision_alignments = pm4py.precision_alignments(log, net, im, fm, activity_key='concept:name', case_id_key='case:concept:name', timestamp_key='time:timestamp')
            variant_log = utils.get_variant_log(log) 
            activity_counts, edge_counts = utils.get_dfg(variant_log)
            ER = entropic_relevance.get_ER(variant_log, activity_counts, edge_counts)
            #results = [i, size_cluster, fitness_tbr, precision_tbr, fitness_alignments, precision_alignments, ER]
            results = [i, size_cluster, fitness_tbr, precision_tbr, ER]
            df.loc[len(df)] = results
            total_results = [a + size_cluster*b for a,b in zip(total_results, results)]
            total_size += size_cluster
        print(total_size)
        average_results = [v/total_size for v in total_results]
        average_results[0] = 'average'
        df.loc[len(df)] = average_results
        print(df)
        df.to_csv('Experiment/Results/'+filename+'/'+str(n_clus)+'/Results_'+method+'.csv', index=False)

         


get_scores_original_log('Experiment/Datasets/Helpdesk.xes')
get_scores_clusters('Helpdesk', 3)

get_scores_original_log('Experiment/Datasets/RTFM.xes')
get_scores_clusters('RTFM', 3)

'''
get_scores_original_log('Experiment/Datasets/Sepsis.xes.gz')
get_scores_clusters('Sepsis', 6)
'''

