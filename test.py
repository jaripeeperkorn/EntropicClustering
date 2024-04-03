import pm4py
import copy
import matplotlib.pyplot as plt

import entroclus.entropic_clustering as entropic_clustering
import alternatives.random_clustering as random_clustering
import alternatives.frequency_based as frequency_based
import alternatives.trace2vec_based as t2v

#! to do fix the trace2vec order variant problem, issue with sepsis log

def save_clusters(dataloc, n_clus):
    filename = dataloc.split("/")[-1]
    filename = filename.replace('.xes', '')
    filename = filename.replace('.gz', '')
    print(filename)

    log = pm4py.read.read_xes(dataloc)

    location_clusters = 'Experiment/Clusters/'+filename+'/'+str(n_clus)+'/'

    #ER based pp
    clusters = entropic_clustering.cluster(input=copy.deepcopy(log), num_clusters=n_clus, outputshape='log', variant='regular', initialization = '++', opt = 'trace')
    for i in range(0, n_clus):
        pm4py.write.write_xes(clusters[i], location_clusters+'/ER_trace_only_pp/Cluster'+str(i+1)+'.xes')

    #ER based pp_norm
    clusters = entropic_clustering.cluster(input=copy.deepcopy(log), num_clusters=n_clus, outputshape='log', variant='regular', initialization = '++_norm', opt = 'trace')
    for i in range(0, n_clus):
        pm4py.write.write_xes(clusters[i], location_clusters+'/ER_trace_only_ppnorm/Cluster'+str(i+1)+'.xes')

    
    #ER based random
    clusters = entropic_clustering.cluster(input=copy.deepcopy(log), num_clusters=n_clus, outputshape='log', variant='regular', initialization = 'random', opt = 'trace')
    for i in range(0, n_clus):
        pm4py.write.write_xes(clusters[i], location_clusters+'/ER_trace_only_random/Cluster'+str(i+1)+'.xes')

    # SPLIT BASED VARIANT #
    #ER based pp
    clusters = entropic_clustering.cluster(input=copy.deepcopy(log), num_clusters=n_clus, outputshape='log', variant='split', initialization = '++', opt = 'trace')
    for i in range(0, n_clus):
        pm4py.write.write_xes(clusters[i], location_clusters+'/ER_trace_only_pp_split/Cluster'+str(i+1)+'.xes')

    #ER based pp_norm
    clusters = entropic_clustering.cluster(input=copy.deepcopy(log), num_clusters=n_clus, outputshape='log', variant='split', initialization = '++_norm', opt = 'trace')
    for i in range(0, n_clus):
        pm4py.write.write_xes(clusters[i], location_clusters+'/ER_trace_only_ppnorm_split/Cluster'+str(i+1)+'.xes')

    
    #ER based random
    clusters = entropic_clustering.cluster(input=copy.deepcopy(log), num_clusters=n_clus, outputshape='log', variant='split', initialization = 'random', opt = 'trace')
    for i in range(0, n_clus):
        pm4py.write.write_xes(clusters[i], location_clusters+'/ER_trace_only_random_split/Cluster'+str(i+1)+'.xes')

    # Alternatives #
    #frequency based
    clusters = frequency_based.cluster(log=copy.deepcopy(log), num_clus=n_clus, outputshape='log', version='k-means++', distance='normalized')
    for i in range(0, n_clus):
        pm4py.write.write_xes(clusters[i], location_clusters+'/frequency_normalized/Cluster'+str(i+1)+'.xes')

    #random
    clusters = random_clustering.cluster(log=copy.deepcopy(log), num_clus=n_clus, variant='equisized', outputshape='log')
    for i in range(0, n_clus):
        pm4py.write.write_xes(clusters[i], location_clusters+'/random/Cluster'+str(i+1)+'.xes')

    #trace2vec based
    clusters = t2v.cluster(log=copy.deepcopy(log), num_clus=n_clus, cluster_version='k-means++', distance='normalized', vector_size=None, window_size=2, min_count=0, dm=0, epochs=200, outputshape='log')
    for i in range(0, n_clus):
        pm4py.write.write_xes(clusters[i], location_clusters+'/trace2vec/Cluster'+str(i+1)+'.xes')

    
save_clusters('Experiment/Datasets/Helpdesk.xes', 3)
save_clusters('Experiment/Datasets/RTFM.xes', 3)

#save_clusters('Experiment/Datasets/Sepsis.xes.gz', 6)