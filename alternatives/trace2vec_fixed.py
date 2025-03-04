import pandas as pd
import numpy as np
import math
import copy
from sklearn.cluster import KMeans
from sklearn import preprocessing
import gensim
from gensim.models.doc2vec import TaggedDocument
import pm4py.stats
import hashlib
import pickle
from entroclus import utils as utils


def get_variants(log, order=True):
    variants = pm4py.stats.get_variants_as_tuples(log)
    return dict(sorted(variants.items(), key=lambda x: x[1], reverse=True)) if order else variants
    #print("Variants from log:", variants)  # Debugging line
    #cleaned = {tuple(str(a).replace(" ", "") for a in v): c for v, c in variants.items()}
    #print("Cleaned variants:", cleaned)  # Debugging line
    #return dict(sorted(cleaned.items(), key=lambda x: x[1], reverse=True)) if order else cleaned

def clean_seq(seq):
    return [x.replace(" ", "").strip() for x in seq]

def get_seqs(df):
    """Convert DataFrame to list of ordered event sequences."""
    groups = df.groupby('case:concept:name')
    return [list(g.sort_values('time:timestamp')['concept:name']) for _, g in groups]

def get_alphabet(log):
    """Return unique activities in log."""
    return list(set(a for trace in log for a in trace))

def hash_tag(seq):
    """Generate a short, unique tag for a sequence using a hash."""
    return hashlib.md5("".join(seq).encode()).hexdigest()[:8]  # 8-char hash tag

def tag_seqs(log):
    """Tag sequences for Doc2Vec model with short unique tags."""
    tagged = [TaggedDocument(seq, [hash_tag(seq)]) for seq in log]
    
    # Debugging step - print the generated tags
    print("Tags used for training Doc2Vec:", [t.tags[0] for t in tagged][:10])
    
    return tagged

def get_tags(log):
    """Generate short unique tags for each trace."""
    return [hash_tag(seq) for seq in log]

def train_t2v(tagged, vec_size, win_size, min_ct, dm, epochs):
    """Train Doc2Vec model."""
    model = gensim.models.Doc2Vec(tagged, vector_size=vec_size, window=win_size, min_count=min_ct, dm=dm)
    model.train(tagged, total_examples=len(tagged), epochs=epochs)
    
    # Debugging step - print stored keys in model
    print("Tags stored in trained Doc2Vec model:", model.dv.index_to_key[:10])

    return model

def cluster_t2v(var_log, log, k, clust_type, dist, vec_size, win_size, min_ct, dm, epochs):
    """Cluster log using Trace2Vec."""
    var_log = copy.deepcopy(var_log)
    keys = list(var_log.keys())

    if vec_size is None:
        vec_size = math.ceil(len(get_alphabet(keys))**0.5)

    log_seqs = [list(trace) for trace in keys]  # Use exact traces from variants
    tagged = tag_seqs(log_seqs)  # Ensure training and inference use the same format

    model = train_t2v(tagged, vec_size, win_size, min_ct, dm, epochs)
    
    tags = get_tags(log_seqs)  # Generate tags again for inference
    print("Generated tags for inference:", tags[:10])

    try:
        vectors = [model.dv[tag] for tag in tags]
    except KeyError as e:
        print("KeyError: One of the generated tags is not in the Doc2Vec model!")
        print("Missing tag:", str(e))
        print("Tags stored in model:", model.dv.index_to_key[:10])
        raise

    if clust_type == 'k-means++' and dist == 'normalized':
        labels = KMeans(n_clusters=k, init=clust_type).fit(preprocessing.normalize(vectors)).labels_

    clusters = [{} for _ in range(k)]
    for i, key in enumerate(keys):
        clusters[labels[i]][key] = var_log[key]
    return clusters

def cluster(log, num_clus, cluster_version='k-means++', distance='normalized', vector_size=None, window_size=2, min_count=1, dm=0, epochs=200, outputshape='log'):
    """Cluster log with Trace2Vec."""
    original_log = copy.deepcopy(log)
    var_log = get_variants(log)
    clusters_vl = cluster_t2v(var_log, log, num_clus, cluster_version, distance, vector_size, window_size, min_count, dm, epochs)

    if outputshape == 'log':
        return [utils.filter_log_with_vl(original_log, cluster_vl) for cluster_vl in clusters_vl]
    elif outputshape == 'variant_log':
        return clusters_vl
    else:
        raise ValueError("Output must be 'log' or 'variant_log'.")
