import math
from entroclus import utils as utils

def get_ER(variant_log, activity_counts, edge_counts):
    """
    Calculate the Entropic Relevance (ER) value for a given variant log, activity counts, and edge counts (last two together dfg).
    #! This is the average ER over all traces in the log

    Parameters:
    - variant_log (dict): A dictionary where the keys are variants (sequences of activities) and the values are the occurrences of each variant in the log.
    - activity_counts (dict): A dictionary containing the counts of each activity in the graph.
    - edge_counts (dict): A dictionary containing the counts of each edge in the graph.

    Returns:
    - float: The ER value for the given variant log, activity counts, and edge counts.
    """
    ER_sum = 0.0
    total_occurences = 0
    for variant, occurrence in variant_log.items():
        prob = utils.get_probability(activity_counts, edge_counts, variant)
        #use this for the logs where probabilities get too small
        prob = max(prob, 1e-10)
        ER_sum += (-math.log(prob, 2))*occurrence
        total_occurences += occurrence
    ER = ER_sum/total_occurences
    return ER

def get_ER_sum(variant_log, activity_counts, edge_counts):
    """
    Calculate the Entropic Relevance (ER) value for a given variant log, activity counts, and edge counts (last two together dfg).
    #! This is the total ER over all traces in the log

    Parameters:
    - variant_log (dict): A dictionary where the keys are variants (sequences of activities) and the values are the occurrences of each variant in the log.
    - activity_counts (dict): A dictionary containing the counts of each activity in the graph.
    - edge_counts (dict): A dictionary containing the counts of each edge in the graph.

    Returns:
    - float: The ER value for the given variant log, activity counts, and edge counts.
    """
    ER_sum = 0.0
    for variant, occurrence in variant_log.items():
        prob = utils.get_probability(activity_counts, edge_counts, variant)
        #use this for the logs where probabilities get too small
        prob = max(prob, 1e-10)
        ER_sum += (-math.log(prob, 2))*occurrence
    return ER_sum

def get_ER_normalized(variant_log, activity_counts, edge_counts):
    """
    Calculate the normalized Entropic Relevance (ER) value for a given variant log, activity counts, and edge counts. it corrects the ER function by adjusting it to not 
    take into account the inherent decrease in probability introduced by loops. We therefore deduct the ER score of each trace, on a dfg mined on only that trace itself
    We might want to use this to make sure  that when taking the highest pairwise ER distance we do no necessarily prefer traces with loops. 

    Parameters:
    - variant_log (dict): A dictionary where the keys are variants (sequences of activities) and the values are the occurrences of each variant in the log.
    - activity_counts (dict): A dictionary containing the counts of each activity in the graph.
    - edge_counts (dict): A dictionary containing the counts of each edge in the graph.

    Returns:
    - float: The normalized ER value.
    
    """
    ER_sum = 0.0
    total_occurences = 0
    for variant, occurrence in variant_log.items():
        # Calculate the replay probability of the trace with the real dfg 
        prob = utils.get_probability(activity_counts, edge_counts, variant)
        # Get a new dfg, which is only discovered using the varint, used for normalization
        act_counts_var, edge_count_var = utils.get_dfg({variant:1})
        # The probability of this dfg is the maximal probability possible for this trace when using dfg's, not always 1 because of loops
        maximal_prob = utils.get_probability(act_counts_var, edge_count_var, variant)
        # Get normalized probability, subtracting the minimal ER at the end (obtained with maximal probability) would be the same
        prob_norm = prob/maximal_prob
        
        ER_sum += (-math.log(prob_norm, 2))*occurrence
        total_occurences += occurrence
    ER = ER_sum/total_occurences
    return ER