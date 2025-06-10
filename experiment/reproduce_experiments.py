import get_elbow_results as get_elbow
import plot_elbow_experiment as plot_elbow_results
import get_clusters_one_cluster_size as get_clusters
import evaluate_clusters_one_cluster_size as evaluate_clusters

def run_elbow_and_evaluate(logname):
    """
    Run the elbow detection and evaluation for a given log.
    """
    # Get elbow results
    get_elbow.test_all_methods(logname, n_clus = 10)
    # Plot elbow results
    plot_elbow_results.generate_plots_from_results(logname, max_clusters = 10, output_dir='experimental_results_elbow/plots')


def elbow_run_all_logs(lst_logs):
    """
    Run elbow detection and evaluation for all logs in the provided list.
    """
    for logname in lst_logs:
        print(f"Processing log: {logname}")
        run_elbow_and_evaluate(logname)


def run_experiment_one_log(logname, n_clus, align=True):
    """
    Run the clustering and evaluation for a specific log and number of clusters.
    
    Args:
        logname (str): Name of the event log.
        n_clus (int): Number of clusters to evaluate.
    """
    # Get clusters for the specified number of clusters
    get_clusters.get_clusters_all_methods(logname, n_clus)
    evaluate_clusters.evaluate_all_methods(logname, n_clus, align=align)
    
def run_experiment_all_logs(lst_logs):
    """
    Run the clustering and evaluation for all logs in the provided list.
    
    Args:
        lst_logs (list): List of event log names.
        n_clus (int): Number of clusters to evaluate.
    """
    #already determined elbow points for each log
    shortcut_elbow_points = {
        'Helpdesk': 4,
        'RTFM': 4,
        'BPIC13_incidents': 5,
        'BPIC13_closedproblems': 5,
        'BPIC15': 4,
        'Hospital_Billing': 5,
        'BPIC12': 4,
        'Sepsis': 6
    }
    for logname in lst_logs:
        n_clus = shortcut_elbow_points[logname]
        print(f"Processing log: {logname} with {n_clus} clusters")
        if logname == 'BPIC12' or logname == 'BPIC15':
            # For BPIC12, we run the experiment with alignment turned off
            run_experiment_one_log(logname, n_clus, align=False)
        else:
            run_experiment_one_log(logname, n_clus, align=True)

def main():
    """
    Main function to run the experiments.
    """
    # Uncomment the following line to run elbow detection and evaluation for all logs
    # elbow_run_all_logs(event_log_names)
    
    # Run clustering and evaluation for all logs
    event_log_names = ['Helpdesk', 'RTFM', 'BPIC13_incidents', 'BPIC13_closedproblems', 'Hospital_Billing', 'Sepsis', 'BPIC12', 'BPIC15']
    run_experiment_all_logs(event_log_names)