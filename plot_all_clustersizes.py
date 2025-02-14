import pandas as pd
import matplotlib.pyplot as plt
import os
import itertools
import re

def extract_cluster_count(filename):
    """Extracts the number of clusters from the filename."""
    match = re.search(r'_(\d+)_results.csv', filename)
    return int(match.group(1)) if match else None

def plot_metric_for_all_methods(csv_files, metric_name, output_dir='experimental_results_plots'):
    """
    Plots the weighted average of a specific metric across methods and number of clusters.
    The first point (cluster=1) is set to the baseline value from 'full_log'.

    Args:
        csv_files (list): List of CSV file paths containing the results.
        metric_name (str): The metric to plot (e.g., 'replay_fitness', 'align_precision').
        output_dir (str): Directory where the plots will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    method_data = {}  # Stores method-wise data
    baseline_values = {}  # Stores full_log values for each metric

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        cluster_count = extract_cluster_count(csv_file)

        if cluster_count is None:
            continue  # Skip invalid files

        # Extract baseline (full_log) for cluster 1
        full_log_row = df[df['method'] == 'full_log']
        if not full_log_row.empty:
            baseline_values = full_log_row.iloc[0].to_dict()  # Store as dictionary

        # Filter for weighted average rows and remove 'full_log'
        df_weighted = df[(df['cluster'] == 'weighted_average') & (df['method'] != 'full_log')]

        for method in df_weighted['method'].unique():
            method_df = df_weighted[df_weighted['method'] == method]
            metric_value = method_df[metric_name].values[0] if not method_df.empty else None

            if method not in method_data:
                method_data[method] = {
                    'num_clusters': [1],  # Always start with 1
                    'metric_values': [baseline_values.get(metric_name, 0)]  # Default to 0 if missing
                }

            if metric_value is not None:
                method_data[method]['num_clusters'].append(cluster_count)
                method_data[method]['metric_values'].append(metric_value)

    # Define dynamic colors and markers
    markers = itertools.cycle(['o', 's', '^', 'D', 'x', '*', 'p', 'h'])
    colors = itertools.cycle(plt.cm.get_cmap("tab10").colors)

    plt.figure(figsize=(10, 6))
    
    for method, data in method_data.items():
        color = next(colors)
        marker = next(markers)
        plt.plot(data['num_clusters'], data['metric_values'], label=method, 
                 color=color, marker=marker, markersize=6, linestyle='-')

    plt.xlabel('Number of Clusters')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} vs Number of Clusters')
    plt.legend(title='Clustering Methods', loc='best')
    plt.grid(True)
    
    plot_filename = os.path.join(output_dir, f'{metric_name}_vs_clusters.pdf')
    plt.savefig(plot_filename, format='pdf', dpi=300)
    plt.close()
    print(f"Saved: {plot_filename}")

def generate_plots_from_results(logname, max_clusters, output_dir='experimental_results_plots'):
    """
    Generates plots for each metric from pre-saved CSV results.

    Args:
        logname (str): Base name of the log file (e.g., 'Helpdesk', 'Sepsis').
        max_clusters (int): Maximum number of clusters to evaluate.
        output_dir (str): Directory where the plots will be saved.
    """
    metrics = [
        'replay_fitness', 'replay_precision', 'align_fitness', 'align_precision', 
        'ER', 'tade_fitness', 'graph_density', 'graph_entropy'
    ]

    csv_files = [f'experimental_results/{logname}_{n_clusters}_results.csv' for n_clusters in range(2, max_clusters + 1)]
    valid_csv_files = [file for file in csv_files if os.path.exists(file)]

    if not valid_csv_files:
        print(f"No valid CSV files found for {logname}.")
        return

    for metric in metrics:
        plot_metric_for_all_methods(valid_csv_files, metric, output_dir)

# Example usage:
generate_plots_from_results('Helpdesk', 10)
