import pandas as pd
import os
import numpy as np
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
import matplotlib.pyplot as plt
from Orange.evaluation import compute_CD, graph_ranks

#Needs Orange3 older version (like 3.30)

#just used for plotting
logs_replacement = {
    'full_log': r"$\mathtt{full\ log}$",
    'trace2vec_based': r"$\mathtt{trace2vec_{km++}}$",
    'entropic_clustering_++': r"$\mathtt{EC_{++}}$",
    'entropic_clustering_++_norm': r"$\mathtt{EC_{++norm}}$",
    'entropic_clustering_randominit': r"$\mathtt{EC_{rand}}$",
    'entropic_clustering_split_++': r"$\mathtt{EC\text{-}split_{++}}$",
    'entropic_clustering_split_++_norm': r"$\mathtt{EC\text{-}split_{++norm}}$",
    'entropic_clustering_split_randominit': r"$\mathtt{EC\text{-}split_{rand}}$",
    'frequency_based': r"$\mathtt{frequency_{km++}}$",
    'random_clustering': r"$\mathtt{random}$",
    'actitrac_freq': r"$\mathtt{actitrac_{freq}}$",
    'actitrac_dist': r"$\mathtt{actitrac_{dist}}$"
}



# Define datasets and clustering techniques
datasets_all = ['Helpdesk', 'RTFM', 'BPIC13_incidents', 'BPIC13_closedproblems', 'Hospital_Billing', 'Sepsis', 'BPIC12', 'BPIC15']
datasets_no_align = ['Helpdesk', 'RTFM', 'BPIC13_incidents', 'BPIC13_closedproblems', 'Hospital_Billing', 'Sepsis']

clustering_methods = ['full_log', 'trace2vec_based', 'entropic_clustering_++', 'entropic_clustering_++_norm',
                      'entropic_clustering_randominit', 'entropic_clustering_split_++', 'entropic_clustering_split_++_norm',
                      'entropic_clustering_split_randominit', 'frequency_based', 'random_clustering', 'actitrac_freq', 'actitrac_dist']

metrics = ['replay_fitness', 'replay_precision', 'align_fitness', 'align_precision', 'simplicity', 'ER', 'graph_density', 'graph_entropy']
maximize_metrics = ['replay_fitness', 'replay_precision', 'align_fitness', 'align_precision', 'simplicity']
minimize_metrics = ['ER', 'graph_density', 'graph_entropy']

# Path to results directory
results_dir = "experimental_results_one_cluster_size/results"

# Dictionary to store rankings
total_rankings = {metric: {method: [] for method in clustering_methods} for metric in metrics}

# Process each dataset
for metric in metrics:
    datasets = datasets_no_align if metric in ['align_fitness', 'align_precision'] else datasets_all

    for dataset in datasets:
        file_path = os.path.join(results_dir, f"{dataset}_results.csv")
        if not os.path.exists(file_path):
            print(f"Warning: Results file for {dataset} not found.")
            continue

        df = pd.read_csv(file_path)
        df = df[(df['cluster'] == '0') | (df['cluster'] == 'average')]

        # Rank clustering techniques for the current metric
        ascending = metric in minimize_metrics
        ranked_df = df[['method', metric]].dropna().sort_values(by=metric, ascending=ascending)
        ranked_df['rank'] = range(1, len(ranked_df) + 1)

        for _, row in ranked_df.iterrows():
            total_rankings[metric][row['method']].append(row['rank'])
# Now, print the average ranks to examine if full_log is consistently ranked poorly
print(total_rankings)

# Compute average ranking for each method
average_rankings = {metric: {} for metric in metrics}
friedman_results = {}  
nemenyi_results = {}  
cd_plot_data = {}  # Store rankings for CD plots

for metric in metrics:
    ranking_matrix = []
    
    for method in clustering_methods:
        if total_rankings[metric][method]:
            avg_rank = sum(total_rankings[metric][method]) / len(total_rankings[metric][method])
            average_rankings[metric][method] = avg_rank
            ranking_matrix.append(total_rankings[metric][method])  
        else:
            average_rankings[metric][method] = None  
    
    # Perform Friedman's test if there is enough data
    if len(ranking_matrix) > 2:  
        friedman_stat, p_value = friedmanchisquare(*ranking_matrix)
        friedman_results[metric] = (friedman_stat, p_value)

        
        # Perform Nemenyi post-hoc test if Friedman test is significant
        if p_value < 0.05:
            nemenyi_pvals = sp.posthoc_nemenyi_friedman(np.array(ranking_matrix).T)
            nemenyi_results[metric] = nemenyi_pvals

            # Store rankings for CD plot
            avg_ranks = [sum(ranks) / len(ranks) for ranks in ranking_matrix]
            cd_plot_data[metric] = (clustering_methods, avg_ranks, len(ranking_matrix[0]))  # (Methods, Average Ranks, #Datasets)

    else:
        friedman_results[metric] = "NaN"

print(average_rankings)

# Add Friedman test results as separate rows (Chi-Squared and p-value in different rows)
friedman_row_chi2 = pd.DataFrame({metric: [f"{friedman_stat:.2f}" if friedman_stat != "Not Computed" else "Not Computed"] for metric, (friedman_stat, _) in friedman_results.items()}, index=["Friedman Chi²"])
friedman_row_pvalue = pd.DataFrame({metric: [f"{p_value:.3f}" if p_value != "Not Computed" else "Not Computed"] for metric, (_, p_value) in friedman_results.items()}, index=["p-value"])

# Convert results to a DataFrame
final_ranking_df = pd.DataFrame(average_rankings)
final_ranking_df = final_ranking_df.round(2)

# Add Friedman test results
final_ranking_df = pd.concat([final_ranking_df, friedman_row_chi2, friedman_row_pvalue])


# Print and save results
print(final_ranking_df)

# Save results as a LaTeX table
latex_table = final_ranking_df.to_latex(float_format=lambda x: f"{x:.2f}", escape=False)
with open(os.path.join(results_dir, "average_ranking_table.tex"), "w") as f:
    f.write(latex_table)

print("LaTeX table saved as average_ranking_table.tex")

# Save Nemenyi test results if available
for metric, pvals in nemenyi_results.items():
    pvals.to_csv(os.path.join(results_dir, f"nemenyi_{metric}.csv"))
    print(f"Nemenyi post-hoc test results saved for {metric}.")

# Generate CD plots
# Generate CD plots
for metric, rankings in average_rankings.items():
    if metric in friedman_results and "NaN" not in friedman_results[metric]:  
        _, p_value = friedman_results[metric] 

        if p_value < 0.05:  # Only plot if Friedman test is significant
            print(f"Creating CD plot for {metric}...")

            avranks = list(rankings.values())  # Extract average ranks
            num_datasets = len(datasets_no_align) if metric in ['align_fitness', 'align_precision'] else len(datasets_all)

            cd = compute_CD(avranks, num_datasets)  # Compute Critical Difference

            # Increase figure size & adjust text
            plt.figure(figsize=(10, 4))

            # Apply LaTeX names
            latex_method_names = [logs_replacement[method] for method in clustering_methods]

            graph_ranks(avranks, latex_method_names, cd=cd, width=8, textspace=1.5)

            plt.title(f"Critical Difference Plot for {metric}", fontsize=14)
            plt.subplots_adjust(left=0.25, right=0.75, top=0.85, bottom=0.15)
            plt.savefig(os.path.join(results_dir, f"cd_plot_{metric}.png"))
            plt.close()

