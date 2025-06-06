import pm4py
from entroclus import utils as utils
from entroclus import entropic_relevance as entropic_relevance

from evaluation.tade_conformance import TADE

from evaluation import graph_simplicity_metrics as gsm



def get_non_stochastic_metrics(log, discovery = 'inductive'):
    '''
    Calculate non-stochastic metrics based on token-based replay and alignments for a given log or cluster.

    Parameters:
    - log: Event log or cluster for analysis
    - discovery:  The name of the discovery algorithm used (default inductive miner)

    Returns:
    - Dictionary containing replay fitness, replay precision, alignment fitness, and alignment precision
    '''
    #discovery of a petri net with inductive miner
    
    if discovery == 'inductive':
        #! to do CHECK NOISE THRESHOLD VALUE
        net, im, fm = pm4py.discover_petri_net_inductive(log, noise_threshold = 0.2, activity_key='concept:name', case_id_key='case:concept:name', timestamp_key='time:timestamp')
    if discovery == 'alpha':
        net, im, fm = pm4py.discover_petri_net_alpha(log, activity_key='concept:name', case_id_key='case:concept:name', timestamp_key='time:timestamp')
    if discovery == 'ilp':
        net, im, fm = pm4py.discover_petri_net_ilp(log, activity_key='concept:name', case_id_key='case:concept:name', timestamp_key='time:timestamp')

    simplicity = pm4py.algo.evaluation.simplicity.algorithm.apply(net)
    
    fitness_tbr = pm4py.fitness_token_based_replay(log, net, im, fm, activity_key='concept:name', case_id_key='case:concept:name', timestamp_key='time:timestamp')['log_fitness']
    precision_tbr = pm4py.precision_token_based_replay(log, net, im, fm, activity_key='concept:name', case_id_key='case:concept:name', timestamp_key='time:timestamp')

    fitness_alignments = pm4py.fitness_alignments(log, net, im, fm, activity_key='concept:name', case_id_key='case:concept:name', timestamp_key='time:timestamp')['log_fitness']
    precision_alignments = pm4py.precision_alignments(log, net, im, fm, activity_key='concept:name', case_id_key='case:concept:name', timestamp_key='time:timestamp')
    return {'replay_fitness': fitness_tbr, 'replay_precision': precision_tbr, 'align_fitness': fitness_alignments, 'align_precision': precision_alignments, 'simplicity': simplicity}


def get_stochastic_metrics(log):
    """
    Calculate the Entropic Relevance (ER) metric for a given log or cluster.

    Parameters:
    - log (pm4py.objects.log.obj.EventLog): The event log or cluster for which to calculate the ER metric.

    Returns:
    - dict: A dictionary containing the ER metric value.
    """
    variant_log = utils.get_variant_log(log)
    #! for now it does not really make sense to split this into entropic precision and fitness (as the automata coming from log and model is the same)
    #Getting ER for log and DFG wiht full behavior, no filter
    activity_counts, edge_counts = utils.get_dfg(variant_log)
    ER = entropic_relevance.get_ER(variant_log, activity_counts, edge_counts)

    return {'ER': ER}


def get_graph_simplicity_metrics(log):
    """
    Calculate the graph simplicity metrics for a given log or cluster.

    Parameters:
    - log (pm4py.objects.log.obj.EventLog): The event log or cluster for which to calculate the graph simplicity metrics.

    Returns:
    - dict: A dictionary containing the graph simplicity metrics.
    """
    variant_log = utils.get_variant_log(log)
    activity_counts, edge_counts = utils.get_dfg(variant_log)

    # Calculate the graph simplicity metrics
    graph_density = gsm.graph_density(activity_counts, edge_counts)
    graph_entropy = gsm.graph_entropy(activity_counts, edge_counts)
    return {'graph_density': graph_density, 'graph_entropy': graph_entropy}


def get_non_stochastic_metrics_no_alignments(log, discovery = 'inductive'):
    '''
    Calculate non-stochastic metrics based on token-based replay and alignments for a given log or cluster.

    Parameters:
    - log: Event log or cluster for analysis
    - discovery:  The name of the discovery algorithm used (default inductive miner)

    Returns:
    - Dictionary containing replay fitness, replay precision, alignment fitness, and alignment precision
    '''
    #discovery of a petri net with inductive miner
    
    if discovery == 'inductive':
        #! to do CHECK NOISE THRESHOLD VALUE
        net, im, fm = pm4py.discover_petri_net_inductive(log, noise_threshold = 0.2, activity_key='concept:name', case_id_key='case:concept:name', timestamp_key='time:timestamp')
    if discovery == 'alpha':
        net, im, fm = pm4py.discover_petri_net_alpha(log, activity_key='concept:name', case_id_key='case:concept:name', timestamp_key='time:timestamp')
    if discovery == 'ilp':
        net, im, fm = pm4py.discover_petri_net_ilp(log, activity_key='concept:name', case_id_key='case:concept:name', timestamp_key='time:timestamp')

    simplicity = pm4py.algo.evaluation.simplicity.algorithm.apply(net)
    
    fitness_tbr = pm4py.fitness_token_based_replay(log, net, im, fm, activity_key='concept:name', case_id_key='case:concept:name', timestamp_key='time:timestamp')['log_fitness']
    precision_tbr = pm4py.precision_token_based_replay(log, net, im, fm, activity_key='concept:name', case_id_key='case:concept:name', timestamp_key='time:timestamp')

    #fitness_alignments = pm4py.fitness_alignments(log, net, im, fm, activity_key='concept:name', case_id_key='case:concept:name', timestamp_key='time:timestamp')['log_fitness']
    #precision_alignments = pm4py.precision_alignments(log, net, im, fm, activity_key='concept:name', case_id_key='case:concept:name', timestamp_key='time:timestamp')

    # We store the value as -999.0 to indicate that it was not calculated - but have a value so we can still rank and need to adapt less of the code
    fitness_alignments = -999.0
    precision_alignments = -999.0

    return {'replay_fitness': fitness_tbr, 'replay_precision': precision_tbr, 'align_fitness': fitness_alignments, 'align_precision': precision_alignments, 'simplicity': simplicity}

    

