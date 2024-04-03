import entroclus.utils as utils
import entroclus.entropic_relevance as entropic_relevance
import entroclus.entropic_clustering_variants as entropic_clustering_variants
import entroclus.entropic_clustering as entropic_clustering

import alternatives.random_clustering as random_clustering
import alternatives.frequency_based as frequency_based


import alternatives.trace2vec_based as t2v


import pm4py

log = pm4py.read.read_xes('Helpdesk.xes')

VL = utils.get_variant_log(log)

clusters = t2v.cluster(log, 3)

print(clusters)


#dfg_a, dfg_e = utils.get_dfg(VL)

#utils.visualize_graph(dfg_a, dfg_e)

#print(entropic_relevance.get_ER(VL, dfg_a, dfg_e))

#c = entropic_clustering_variants.entropic_clustering_split(log, 3)


#clusters = entropic_clustering.cluster(log, 3, outputshape='variant_log')

#clusters = frequency_based.cluster(log, 3)

#print(clusters)