# Model-driven Stochastic Trace Clustering

## Description

This is the github repository containing the code, experiments and experimental results used in the paper:
> Model-driven Stochastic Trace Clustering
> Jari Peeperkorn, Johannes De Smedt & Jochen De Weerdt

## Using the code

The folder `entroclus` contains the code for the clustering algorithm.

`alternatives` contains all the alternative clustering algorithms used in the experiments (with the exception of ActiTraC).

```tutorials to be added```


## Structure

```
├── entroclus/                      # The main files needed for our model-driven clustering apporach
│   ├── entropic_clustering.py          # The script used for running our method
│   ├── entropic_clustering_variants.py # Every alternative of entroclus
│   ├── entropic_clustering_utils.py    # Utilities such as initialization
│   ├── entropic_relevance.py           # Custom python script to run our version of ER
│   └── utils.py                        # Containing extra utilities such as DFG discovery
├── alternatives/                   # Alternative clustering algorithms (except ActiTraC)
│   ├── frequency_based.py              # Frequency-based clustering
│   ├── random_clustering.py            # Random clustering
│   ├── trace2vec_based.py              # Trace2Vec-based clustering
│   └── trace2vec_fixed.py              # Trace2Vec with fixed parameters
├── evaluation/                     # Evaluation metrics and utilities
│   ├── graph_simplicity_metrics.py     # Graph simplicity metrics
│   ├── metrics.py                      # General metrics
│   ├── stochastic_PN.py                # Stochastic Petri Net evaluation
│   └── tade_conformance.py             # TADE conformance metrics
├── experiment/                     # All scripts and files needed for the experiment + output
│   ├── datasets/                        # Datasets used in the experiments
│   ├── experimental_results_elbow/      # Results for elbow experiments
│   ├── experimental_results_one_cluster_size/ # Results for fixed cluster size experiments
│   ├── evaluate_clusters_one_cluster_size.py  # Evaluation script
│   ├── get_average_ranks.py             # Compute average ranks
│   ├── get_average_ranks_statistical_test.py # Statistical tests on ranks
│   ├── get_clusters_one_cluster_size.py # Get clusters for fixed size
│   ├── get_elbow_results.py             # Get elbow method results
│   ├── get_log_stats.py                 # Log statistics
│   ├── plot_elbow_experiment.py         # Plotting elbow experiment results
│   ├── reproduce_experiments.py         # Script to rerun all experiments
│   ├── run_elbow_experiment.py          # Run elbow experiment
├── requirements.txt                # Python dependencies
├── README.md                       # This file
```

## Reproducing the experiment

1. Install the required dependencies:

```powershell
pip install -r requirements.txt
```

2. To reproduce the main experiments, run the following script:

```powershell
python experiment/reproduce_experiments.py
```

This will execute all experiments and generate the results in the `experiment/experimental_results_elbow/` and `experiment/experimental_results_one_cluster_size/` directories.

## Citing

If you use this code or results in your research, please cite the following paper:

> Model-driven Stochastic Trace Clustering
> Jari Peeperkorn, Johannes De Smedt & Jochen De Weerdt

## Contact

For questions or issues, please contact Jari Peeperkorn (jari.peeperkorn@kuleuven.be).
