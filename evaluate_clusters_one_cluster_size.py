    #! actitrac to add with ProM manually
    #todo add automatic java calls for actitrac or implement in python

from evaluation import metrics
import pm4py
import pandas as pd
import os



def evaluate_all_methods(logname, n_clusters, align=True):
    """
    methods = ['trace2vec_based',
               'entropic_clustering_++', 'entropic_clustering_++_norm', 'entropic_clustering_randominit', 
               'entropic_clustering_split_++', 'entropic_clustering_split_++_norm', 'entropic_clustering_split_randominit', 
               'frequency_based', 'random_clustering',
               'actitrac_freq', 'actitrac_dist']
    """

    methods = ['trace2vec_based',
               'entropic_clustering_++', 'entropic_clustering_++_norm', 'entropic_clustering_randominit', 
               'entropic_clustering_split_++', 'entropic_clustering_split_++_norm', 'entropic_clustering_split_randominit', 
               'frequency_based', 'random_clustering']
    

    
    columns =  ['method', 'cluster', 'num_traces', 
                'replay_fitness', 'replay_precision', 
                'align_fitness', 'align_precision',
                'simplicity', 
                'ER', 
                'graph_density', 'graph_entropy']
    
    results_loc = f"experimental_results_one_cluster_size/results/{logname.replace('.xes', '').replace('.gz', '')}_results.csv"

    # Load existing results if available
    if os.path.exists(results_loc):
        print(f"Results file {logname} exists, loading existing results.")
        df = pd.read_csv(results_loc)
    else:
        print(f"Results file {logname} does not exist, creating new DataFrame.")
        df = pd.DataFrame(columns=columns)

    # Check if baseline results already exist
    if not (df['method'] == 'full_log').any():
        print("Baseline results do not exist, calculating.")
        log = pm4py.read_xes('datasets/'+logname)
        if align==True:
            baseline_PN = metrics.get_non_stochastic_metrics(log)
        else:
            baseline_PN = metrics.get_non_stochastic_metrics_no_alignments(log)
        baseline_stoch = metrics.get_stochastic_metrics(log)
        baseline_graph_simplicity = metrics.get_graph_simplicity_metrics(log)

        baseline_results = ['full_log', 0, len(log), 
                            baseline_PN['replay_fitness'], baseline_PN['replay_precision'], 
                            baseline_PN['align_fitness'], baseline_PN['align_precision'],
                            baseline_PN['simplicity'], 
                            baseline_stoch['ER'], 
                            baseline_graph_simplicity['graph_density'], baseline_graph_simplicity['graph_entropy']]
        df.loc[df.shape[0]] = baseline_results
        df.to_csv(results_loc, index=False)  # Save after baseline
    else:
        print("Baseline results already exist.")
        
    print(df)

    for method in methods:
        if (df['method'] == method).any():  # Skip if already computed
            print(f"Skipping {method}, results already exist.")
            continue

        print(method)
        total_traces = 0
        weighted_sums = {col: 0 for col in columns[3:]}  # Initialize dictionary to store weighted sums of metrics used for avergaging

        for clus in range(0, n_clusters):
            curr_cluster_loc = f"experimental_results_one_cluster_size/clusters/{logname.replace('.xes','').replace('.gz','')}/{method}/cluster_{str(clus+1)}.xes"
            curr_cluster = pm4py.read_xes(curr_cluster_loc)
            num_traces = len(curr_cluster)
            if align==True:
                curr_PN = metrics.get_non_stochastic_metrics(curr_cluster)
            else:
                curr_PN = metrics.get_non_stochastic_metrics_no_alignments(curr_cluster)
            curr_stoch = metrics.get_stochastic_metrics(curr_cluster)
            curr_graph_simplicity = metrics.get_graph_simplicity_metrics(curr_cluster)
            curr_results = [method, clus+1, len(curr_cluster), 
                            curr_PN['replay_fitness'], curr_PN['replay_precision'], 
                            curr_PN['align_fitness'], curr_PN['align_precision'],
                            curr_PN['simplicity'],
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
        df.to_csv(results_loc, index=False)  # Save after averages
        print(df)

    #df.to_csv(results_loc, index=False)

    # Generate LaTeX Table for the results
    generate_latex_table(df, logname)

def generate_latex_table(df, logname):
    """
    This function generates a LaTeX table comparing the baseline and weighted averages for each method.
    """
    latex_code = f"""
\\begin{{table}}[ht]
\\centering
\\begin{{tabular}}{{|l|l|l|l|l|l|l|l|l|l|l|}} 
\\hline
Method & Replay Fitness & Replay Precision & Align Fitness & Align Precision & Simplicity & ER & Graph Density & Graph Entropy \\\\
\\hline
"""

    # Add Baseline Results
    baseline_row = df.iloc[0]  # First row should be the baseline (full log)
    latex_code += f"Full Log & {baseline_row['replay_fitness']:.3f} & {baseline_row['replay_precision']:.3f} & {baseline_row['align_fitness']:.3f} & {baseline_row['align_precision']:.3f} & {baseline_row['simplicity']:.3f} & {baseline_row['ER']:.3f} & {baseline_row['graph_density']:.3f} & {baseline_row['graph_entropy']:.3f} \\\\ \n"

    # Add Weighted Averages for each method
    for method in df['method'].unique():
        if method != 'full_log':  # Skip the baseline row
            method_rows = df[df['method'] == method]
            weighted_avg_row = method_rows.iloc[-1]  # Last row for weighted averages
            latex_code += f"{method} & {weighted_avg_row['replay_fitness']:.3f} & {weighted_avg_row['replay_precision']:.3f} & {weighted_avg_row['align_fitness']:.3f} & {weighted_avg_row['align_precision']:.3f} & {weighted_avg_row['simplicity']:.3f} & {weighted_avg_row['ER']:.3f} & {weighted_avg_row['graph_density']:.3f} & {weighted_avg_row['graph_entropy']:.3f} \\\\ \n"

    latex_code += """
\\hline
\\end{tabular}
\\caption{Metrics comparison for different clustering methods}
\\label{tab:metrics_comparison}
\\end{table}
"""
    latex_filename = f"experimental_results_one_cluster_size/results/{logname.replace('.xes', '').replace('.gz', '')}_latex_table.tex"
    
    with open(latex_filename, 'w') as f:
        f.write(latex_code)
    
    print(f"LaTeX table saved to {latex_filename}")


#This function was purely added because of the prder we ran the experiments
def add_actitrac_methods(logname, n_clusters, align=True):
    actitrac_methods = ['actitrac_freq', 'actitrac_dist']

    results_loc = f"experimental_results_one_cluster_size/results/{logname.replace('.xes', '').replace('.gz', '')}_results.csv"
    # Load existing results if available
    if os.path.exists(results_loc):
        print("Results file exists, loading existing results.")
        df = pd.read_csv(results_loc)
    else:
        raise ValueError("The resutls file does not exist for this log.")

    columns = ['method', 'cluster', 'num_traces', 
                'replay_fitness', 'replay_precision', 
                'align_fitness', 'align_precision',
                'simplicity', 
                'ER', 
                'graph_density', 'graph_entropy']
    
    for method in actitrac_methods:

        if (df['method'] == method).any():  # Skip if already computed
            print(f"Skipping {method}, results already exist.")
            continue

        print(method)
        total_traces = 0
        weighted_sums = {col: 0 for col in columns[3:]}
        for clus in range(0, n_clusters):
            curr_cluster_loc = f"experimental_results_one_cluster_size/clusters/{logname.replace('.xes','').replace('.gz','')}/{method}/cluster_{str(clus+1)}.xes"
            curr_cluster = pm4py.read_xes(curr_cluster_loc)
            num_traces = len(curr_cluster)
            if align==True:
                curr_PN = metrics.get_non_stochastic_metrics(curr_cluster)
            else:
                curr_PN = metrics.get_non_stochastic_metrics_no_alignments(curr_cluster)
            curr_stoch = metrics.get_stochastic_metrics(curr_cluster)
            curr_graph_simplicity = metrics.get_graph_simplicity_metrics(curr_cluster)
            curr_results = [method, clus+1, len(curr_cluster), 
                            curr_PN['replay_fitness'], curr_PN['replay_precision'], 
                            curr_PN['align_fitness'], curr_PN['align_precision'],
                            curr_PN['simplicity'],
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
        df.to_csv(results_loc, index=False)  # Save after averages
        print(df)
    #df.to_csv(results_loc, index=False)

    # Generate LaTeX Table for the results
    generate_latex_table(df, logname)



#when ER doesn't go down we use graph entropy for both BPIC13 and BPIC12

elbow_points = {'Helpdesk': 4,
                'RTFM': 4,
                'BPIC13_incidents': 5,
                'BPIC13_closedproblems': 5,
                'BPIC15': 4,
                'Hospital_Billing': 5,
                'BPIC12': 4,
                'Sepsis': 6}       

#to_run = ['Helpdesk', 'RTFM', 'BPIC13_incidents', 'BPIC13_closedproblems', 'Hospital_Billing', 'Sepsis']
#for logname in to_run:
#    evaluate_all_methods(logname+'.xes', elbow_points[logname])

#to_run = ['BPIC15', 'BPIC12']
#for logname in to_run:
#    evaluate_all_methods(logname+'.xes', elbow_points[logname], align=False)


to_run = ['Helpdesk', 'RTFM', 'BPIC13_incidents', 'BPIC13_closedproblems', 'Hospital_Billing', 'Sepsis']

for logname in to_run:
    add_actitrac_methods(logname+'.xes', elbow_points[logname])

#to_run = ['BPIC15', 'BPIC12']

#for logname in to_run: 
#    add_actitrac_methods(logname+'.xes', elbow_points[logname], align=False)