import pandas as pd
import os
import numpy as np
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
import matplotlib.pyplot as plt

# try import Orange plotting utilities, but tolerate failure (some envs lack compiled extensions)
try:
    from Orange.evaluation import compute_CD, graph_ranks
    has_orange_eval = True
except Exception as e:
    # Orange not available or failed to import (C extensions missing). Disable CD plotting.
    print(f"Warning: Orange.evaluation unavailable, skipping CD plots (import error: {e}).")
    has_orange_eval = False

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

# Path to results directory — make it absolute and relative to this script so printed paths are consistent
base_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(base_dir, "experimental_results_one_cluster_size", "results")
results_dir = os.path.normpath(results_dir)

# optional: helpful debug message when folder is missing
if not os.path.isdir(results_dir):
    print(f"Warning: results_dir does not exist: {results_dir}")

# Dictionary to store rankings
total_rankings = {metric: {method: [] for method in clustering_methods} for metric in metrics}

# Also store the raw ranking_matrix per metric so we can run post-hoc later
ranking_matrix_per_metric = {}

# Process each dataset
for metric in metrics:
    datasets = datasets_no_align if metric in ['align_fitness', 'align_precision'] else datasets_all

    for dataset in datasets:
        file_path = os.path.join(results_dir, f"{dataset}_results.csv")
        if not os.path.exists(file_path):
            print(f"Warning: Results file for {dataset} not found.")
            print(file_path)
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
holm_corrected_pvals = {}
nemenyi_results = {}
cd_plot_data = {}  # Store rankings for CD plots

# Perform Friedman tests and store ranking matrices per metric
for metric in metrics:
    ranking_matrix = []

    for method in clustering_methods:
        if total_rankings[metric][method]:
            avg_rank = sum(total_rankings[metric][method]) / len(total_rankings[metric][method])
            average_rankings[metric][method] = avg_rank
            ranking_matrix.append(total_rankings[metric][method])
        else:
            average_rankings[metric][method] = None

    # Save ranking matrix for later post-hoc if available
    ranking_matrix_per_metric[metric] = ranking_matrix

    # Perform Friedman's test if there is enough data
    if len(ranking_matrix) > 2:
        try:
            friedman_stat, p_value = friedmanchisquare(*ranking_matrix)
            friedman_results[metric] = (friedman_stat, p_value)
        except Exception as e:
            print(f"Friedman test failed for {metric}: {e}")
            friedman_results[metric] = (None, None)
    else:
        friedman_results[metric] = (None, None)

print(average_rankings)

# Holm correction across all metrics' Friedman's p-values
# Collect metrics with computed p-values
metrics_with_p = [m for m, (stat, p) in friedman_results.items() if p is not None]
pvals = [friedman_results[m][1] for m in metrics_with_p]

def holm_correction(metrics_list, pvalues):
    # Return dict metric -> holm-adjusted p-value
    m = len(pvalues)
    if m == 0:
        return {}
    # sort by p ascending
    sorted_idx = sorted(range(m), key=lambda i: pvalues[i])
    sorted_p = [pvalues[i] for i in sorted_idx]
    adj_sorted = [0.0] * m
    # Holm: adj_p_i = min(1, (m - i) * p_(i)) with monotonicity enforced
    for i, p in enumerate(sorted_p):
        adj = min(1.0, (m - i) * p)
        adj_sorted[i] = adj
    # enforce non-decreasing (monotonicity)
    for i in range(1, m):
        if adj_sorted[i] < adj_sorted[i - 1]:
            adj_sorted[i] = adj_sorted[i - 1]
    # map back to metrics
    adj_by_metric = {}
    for rank_pos, idx in enumerate(sorted_idx):
        adj_by_metric[metrics_list[idx]] = adj_sorted[rank_pos]
    return adj_by_metric

holm_corrected_pvals = holm_correction(metrics_with_p, pvals)

# Use Holm-adjusted p-values to decide whether to run Nemenyi and produce CD plots
for metric in metrics:
    stat, raw_p = friedman_results.get(metric, (None, None))
    holm_p = holm_corrected_pvals.get(metric, None)

    if holm_p is not None and holm_p < 0.05:
        # Run Nemenyi post-hoc using the stored ranking matrix
        ranking_matrix = ranking_matrix_per_metric.get(metric, [])
        if ranking_matrix and len(ranking_matrix) > 2:
            try:
                nemenyi_pvals = sp.posthoc_nemenyi_friedman(np.array(ranking_matrix).T)
                nemenyi_results[metric] = nemenyi_pvals

                # Store rankings for CD plot
                avg_ranks = [sum(ranks) / len(ranks) for ranks in ranking_matrix]
                cd_plot_data[metric] = (clustering_methods, avg_ranks, len(ranking_matrix[0]))  # (Methods, Average Ranks, #Datasets)
            except Exception as e:
                print(f"Nemenyi failed for {metric}: {e}")

# Add Friedman test results as separate rows (Chi-Squared and p-value in different rows)
# Handle missing values gracefully
friedman_row_chi2 = pd.DataFrame({metric: [f"{friedman_results[metric][0]:.2f}" if (friedman_results[metric][0] is not None) else "Not Computed"] for metric in metrics}, index=["Friedman Chi²"])
friedman_row_pvalue = pd.DataFrame({metric: [f"{friedman_results[metric][1]:.3f}" if (friedman_results[metric][1] is not None) else "Not Computed"] for metric in metrics}, index=["p-value (Friedman)"])

# Convert results to a DataFrame
final_ranking_df = pd.DataFrame(average_rankings)
final_ranking_df = final_ranking_df.round(2)

# Add Friedman test results
final_ranking_df = pd.concat([final_ranking_df, friedman_row_chi2, friedman_row_pvalue])

# Also add Holm-adjusted p-values row for clarity
holm_row = pd.DataFrame({metric: [f"{holm_corrected_pvals[metric]:.3f}" if metric in holm_corrected_pvals else "Not Computed"] for metric in metrics}, index=["p-value (Holm)"])
final_ranking_df = pd.concat([final_ranking_df, holm_row])

# Print and save results
print(final_ranking_df)

# Save results as a LaTeX table
latex_table = final_ranking_df.to_latex(float_format=lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else str(x), escape=False)
with open(os.path.join(results_dir, "average_ranking_table_EXTRA.tex"), "w") as f:
    f.write(latex_table)

print("LaTeX table saved as average_ranking_table_EXTRA.tex")

# Save Nemenyi test results if available
for metric, pvals in nemenyi_results.items():
    pvals.to_csv(os.path.join(results_dir, f"nemenyi_{metric}_EXTRA.csv"))
    print(f"Nemenyi post-hoc test results saved for {metric}.")

# Generate CD plots (only if Orange evaluation utilities are available)
for metric, rankings in average_rankings.items():
    holm_p = holm_corrected_pvals.get(metric, None)
    if holm_p is not None and holm_p < 0.05:
        if not has_orange_eval:
            print(f"Skipping CD plot for {metric}: Orange.evaluation not available.")
            continue

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
        plt.savefig(os.path.join(results_dir, f"cd_plot_{metric}_EXTRA.png"))
        plt.close()

