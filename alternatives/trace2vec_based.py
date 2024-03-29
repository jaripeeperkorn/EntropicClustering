import pandas as pd

import numpy as np
import math

import copy
from sklearn.cluster import KMeans
from sklearn import preprocessing

import gensim
from gensim.models.doc2vec import TaggedDocument

import entroclus.utils as utils

def sequences_from_dataframe(df):
    """
    Converts a dataframe into a list of sequences.

    Parameters:
    - df (pandas.DataFrame): The input dataframe.

    Returns:
    - sequences (list): A list of sequences, where each sequence is a list of concept names.

    Note:
    - The dataframe should have a column named 'case:concept:name' to group the sequences together.
    - The dataframe should be sorted by the 'time:timestamp' column before calling this function.
    - The sequences are extracted based on the order of 'time:timestamp' within each group.
    """    
    #we need this function because in recent version of pm4py they have switched to df format
    # Sort the dataframe by 'time:timestamp'
    #! for some reason ordering made it bug, probably because of formatting
    #! xes importer pm4py automatically puts cases together and orders on timestamp, but better to fix this in future to make sure
    #df_sorted = df.sort_values(by='time:timestamp')
    df_sorted = df
    # Group by 'case:concept:name' to keep sequences together
    grouped = df_sorted.groupby('case:concept:name')
    # Initialize an empty list to store sequences
    sequences = []
    # Iterate through each group
    for group_name, group_df in grouped:
        # Sort the group by 'time:timestamp'
        group_df_sorted = group_df.sort_values(by='time:timestamp')  
        # Extract the sequence of concept names for the current group
        sequence = list(group_df_sorted['concept:name'])
        # Append the sequence to the list of sequences
        sequences.append(sequence)
    return sequences

def get_alphabet(log):
    """
    Return a list of unique activities in the log.

    Parameters:
    - log (list): A list of traces, where each trace is a list of activities.

    Returns:
    - list: A list of unique activities in the log.
    """
    activities = set([act for trace in log for act in trace])
    return list(activities)
def add_tags(log):
    """
    Add tags to each log sequence in the input log.

    Parameters:
    - log (list): A list of log sequences.

    Returns:
    - taggedlog (list): A list of TaggedDocument objects, where each object represents a log sequence with an associated tag.

    Example:
    >>> log = [['A', 'B', 'C'], ['D', 'E', 'F']]
    >>> add_tags(log)
    [TaggedDocument(words=['A', 'B', 'C'], tags=['ABC']), TaggedDocument(words=['D', 'E', 'F'], tags=['DEF'])]
    """
    taggedlog = []
    for j in range(len(log)):
        ID = str('')
        for i in range(len(log[j])):
            ID = ID + log[j][i].replace(" ", "")
        trace_id = [ID]
        td = TaggedDocument(log[j], trace_id)
        taggedlog.append(td)
    return(taggedlog)

def get_tags(log):
    """
    Return a list of tags generated from a log.

    Parameters:
    - log (list): A list of lists representing the log.

    Returns:
    - list: A list of tags generated from the log.

    Example:
    >>> log = [['A', 'B', 'C'], ['D', 'E', 'F']]
    >>> get_tags(log)
    ['ABC', 'DEF']
    """
    tags = []
    for j in range(len(log)):
        ID = str('')
        for i in range(len(log[j])):
            ID = ID + log[j][i].replace(" ", "")
        tags.append(ID)
    return(tags)

def train_model(taggedlog, vector_size, window_size, min_count, dm, epochs):
    """
    Train a Doc2Vec model on a tagged log.

    Parameters:
    - taggedlog (list): A list of tagged log sequences.
    - vector_size (int): The dimensionality of the feature vectors.
    - window_size (int): The maximum distance between the current and predicted word within a sentence.
    - min_count (int): Ignores all words with total frequency lower than this.
    - dm (int): The training algorithm. Set to 1 for distributed memory (PV-DM), 0 for distributed bag of words (PV-DBOW).
    - epochs (int): Number of iterations (epochs) over the corpus.

    Returns:
    - model (gensim.models.Doc2Vec): The trained Doc2Vec model.

    Note:
    - The default values for vector_size, window_size, min_count, dm, and epochs may not necessarily make sense for all use cases.
    - Make sure to preprocess the tagged log before passing it to this function.
    """
    model = gensim.models.Doc2Vec(taggedlog, vector_size=vector_size, window=window_size, 
                                  min_count=min_count, dm = dm)
    model.train(taggedlog, total_examples=len(taggedlog), epochs=epochs)
    return model

def cluster_t2v(input_variant_log, log, num_clus, cluster_version, distance, vector_size, window_size, min_count, dm, epochs):
    """
    Cluster the input variant log using the Trace2Vec approach.

    Parameters:
    - input_variant_log (dict): A dictionary representing the input variant log, where the keys are the trace IDs and the values are the corresponding traces.
    - log (pandas.DataFrame): The input log dataframe.
    - num_clus (int): The number of clusters to create.
    - cluster_version (str): The version of the clustering algorithm to use. Currently, only 'k-means++' is supported.
    - distance (str): The distance metric to use for clustering. Currently, only 'normalized' is supported.
    - vector_size (int): The dimensionality of the feature vectors. If None, it will be automatically calculated based on the size of the alphabet.
    - window_size (int): The maximum distance between the current and predicted word within a sentence for training the Doc2Vec model.
    - min_count (int): Ignores all words with total frequency lower than this for training the Doc2Vec model.
    - dm (int): The training algorithm for the Doc2Vec model. Set to 1 for distributed memory (PV-DM), 0 for distributed bag of words (PV-DBOW).
    - epochs (int): Number of iterations (epochs) over the corpus for training the Doc2Vec model.

    Returns:
    - clusters (list): A list of dictionaries representing the clusters. Each dictionary contains the trace IDs as keys and the corresponding traces as values.

    Note:
    - If vector_size is not provided, it will be automatically calculated based on the size of the alphabet in the input variant log.
    - The default values for vector_size, window_size, min_count, dm, and epochs may not necessarily make sense for all use cases.
    """
    variant_log = copy.deepcopy(input_variant_log)
    keys = list(variant_log.keys())

    if vector_size == None:
        voc = get_alphabet(keys)
        vector_size = math.ceil(len(voc)**0.5)

    log_list = sequences_from_dataframe(log)
    log_tagged = add_tags(log_list)
    print("preprocessing done")
    model = train_model(log_tagged, vector_size = vector_size, window_size=window_size, min_count=min_count, dm=dm, epochs=epochs)
    print("training done") 
    tags_variants = get_tags(keys)
    vector_log_variants = [model.dv[tag] for tag in tags_variants]

    #todo add other options?
    if cluster_version == 'k-means++':
        if distance == 'normalized':
            kmeans = KMeans(n_clusters=num_clus, init=cluster_version).fit(preprocessing.normalize(vector_log_variants))
            labels = kmeans.labels_

    clusters = [{} for _ in range(num_clus)]
    for i, key in enumerate(keys):
        clusters[labels[i]][key] = variant_log[key] 
    return clusters

def cluster(log, num_clus, cluster_version='k-means++', distance='normalized', vector_size=None, window_size=2, min_count=0, dm=0, epochs=200, outputshape='log'):
    """
    Cluster the input log using the Trace2Vec approach.

    Parameters:
    - log (pandas.DataFrame): The input log dataframe.
    - num_clus (int): The number of clusters to create.
    - cluster_version (str): The version of the clustering algorithm to use. Default is 'k-means++'.
    - distance (str): The distance metric to use for clustering. Default is 'normalized'.
    - vector_size (int): The dimensionality of the feature vectors. If None, it will be automatically calculated based on the size of the alphabet.
    - window_size (int): The maximum distance between the current and predicted word within a sentence for training the Doc2Vec model. Default is 2.
    - min_count (int): Ignores all words with total frequency lower than this for training the Doc2Vec model. Default is 0.
    - dm (int): The training algorithm for the Doc2Vec model. Set to 1 for distributed memory (PV-DM), 0 for distributed bag of words (PV-DBOW). Default is 0.
    - epochs (int): Number of iterations (epochs) over the corpus for training the Doc2Vec model. Default is 150.
    - outputshape (str): The shape of the output. Can be 'log' or 'variant_log'. Default is 'log'.

    Returns:
    - clusters (list): A list of logs or dictionaries representing the clusters.

    Note:
    - If vector_size is not provided, it will be automatically calculated based on the size of the alphabet in the input log.
    - The default values for vector_size, window_size, min_count, dm, and epochs may not necessarily make sense for all use cases.
    """
    variant_log_input = utils.get_variant_log(log)
    clusters_vl = cluster_t2v(variant_log_input, log, num_clus, cluster_version, distance, vector_size, window_size, min_count,dm,epochs)
    if outputshape == 'log':
        return [utils.filter_log_with_vl(log, cluster_vl) for cluster_vl in clusters_vl]
    elif outputshape == 'variant_log':
        return clusters_vl
    else:
        raise ValueError("Output has to be 'log' or 'variant_log'.")