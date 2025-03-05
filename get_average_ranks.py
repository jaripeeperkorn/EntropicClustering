import pandas as pd
import os

# Define datasets and clustering techniques

#datasets_no_align = ['Helpdesk', 'RTFM', 'BPIC13_incidents', 'BPIC13_closedproblems', 'Hospital_Billing', 'Sepsis', 'BPIC12', 'BPIC15']
datasets_no_align = ['Helpdesk', 'RTFM', 'BPIC13_incidents', 'BPIC13_closedproblems', 'Hospital_Billing', 'Sepsis']
datasets_all = ['Helpdesk', 'RTFM', 'BPIC13_incidents', 'BPIC13_closedproblems', 'Hospital_Billing', 'Sepsis']


clustering_methods = ['full_log', 'trace2vec_based', 'entropic_clustering_++', 'entropic_clustering_++_norm',
                      'entropic_clustering_randominit', 'entropic_clustering_split_++', 'entropic_clustering_split_++_norm',
                      'entropic_clustering_split_randominit', 'frequency_based', 'random_clustering', 'actitrac_freq', 'actitrac_dist']

metrics = ['replay_fitness', 'replay_precision', 'align_fitness', 'align_precision', 'simplicity', 'ER', 'graph_density', 'graph_entropy']
maximize_metrics = ['replay_fitness', 'replay_precision', 'align_fitness', 'align_precision', 'simplicity']
minimize_metrics = ['ER',  'graph_density', 'graph_entropy']

# Path to results directory
results_dir = "experimental_results_one_cluster_size/results"

# Dictionary to store rankings
total_rankings = {metric: {method: [] for method in clustering_methods} for metric in metrics}

# Process each dataset

for metric in metrics:
    # Choose dataset list based on metric
    datasets = datasets_no_align if metric in ['align_fitness', 'align_precision'] else datasets_all

    for dataset in datasets:
        file_path = os.path.join(results_dir, f"{dataset}_results.csv")
        if not os.path.exists(file_path):
            print(f"Warning: Results file for {dataset} not found.")
            continue

        # Read CSV
        df = pd.read_csv(file_path)

        # Only keep the weighted averages
        df = df[(df['cluster'] == '0') | (df['cluster'] == 'average')]

        # Rank clustering techniques for the current metric
        ascending = metric in minimize_metrics  # Ascending for minimization metrics
        ranked_df = df[['method', metric]].dropna().sort_values(by=metric, ascending=ascending)
        ranked_df['rank'] = range(1, len(ranked_df) + 1)

        for _, row in ranked_df.iterrows():
            total_rankings[metric][row['method']].append(row['rank'])

# Compute average ranking for each method
average_rankings = {metric: {} for metric in metrics}
for metric in metrics:
    for method in clustering_methods:
        if total_rankings[metric][method]:
            average_rankings[metric][method] = sum(total_rankings[metric][method]) / len(total_rankings[metric][method])
        else:
            average_rankings[metric][method] = None  # No ranking data

# Convert results to a DataFrame
final_ranking_df = pd.DataFrame(average_rankings)
final_ranking_df = final_ranking_df.round(2)
print(final_ranking_df)

# Save results as a LaTeX table
latex_table = final_ranking_df.to_latex(float_format="%.2f")
with open(os.path.join(results_dir, "average_ranking_table.tex"), "w") as f:
    f.write(latex_table)

print("LaTeX table saved as average_ranking_table.tex")