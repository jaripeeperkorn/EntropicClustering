import pandas

from entroclus import entropic_clustering_variants as entropic_clustering_variants
from entroclus import utils as utils

def cluster(input, num_clusters, outputshape='log', variant='regular', initialization = '++', opt = 'trace'):
    """
    Cluster the input data using entropic clustering.

    Parameters:
    - input: pandas.core.frame.DataFrame or dict
        The input data to be clustered. If input is a pandas DataFrame, it is assumed to be a pm4py event log. If input is a dictionary, it is assumed to be a variant log.
    - num_clusters: int
        The number of clusters to create.
    - outputshape: str, optional
        The shape of the output. Default is 'log'. Possible values are 'log' and 'variant_log'.
    - variant: str, optional
        The variant of entropic clustering to use. Default is 'regular'. Possible values are 'regular' and 'split'.
    - initialization: str, optional
        The initialization method to use. Default is '++'.
    - opt: str, optional
        The optimization method to use. Default is 'trace'.

    Returns:
    - list or dict
        If outputshape is 'log', a list of filtered logs for each cluster is returned. 
        If outputshape is 'variant_log', a list of dictionaries (variant logs) is returned.

    Raises:
    - ValueError
        If variant is not 'regular' or 'split'.
        If outputshape is not 'log' or 'variant_log'.
        If input is not a pandas DataFrame or a variant log dictionary.
        if input is a variant log dictionary and outputshape is not 'variant_log'
    """
    if isinstance(input,pandas.core.frame.DataFrame) == True:
        if variant == 'regular':
            clusters_vl = entropic_clustering_variants.entropic_clustering(log=input, num_clusters=num_clusters, initialization=initialization, opt=opt)
        elif variant == 'split':
            clusters_vl = entropic_clustering_variants.entropic_clustering_split(log=input, num_clusters=num_clusters, initialization=initialization, opt=opt)
        else:
            raise ValueError("Variant has to be 'regular' or 'split'.")
        if outputshape == 'variant_log':
            return clusters_vl
        elif outputshape == 'log':
            return [utils.filter_log_with_vl(input, cluster_vl) for cluster_vl in clusters_vl]
        else:
            raise ValueError("Output has to be 'log' or 'variant_log'.")
    elif isinstance(input,dict) == True:
        if variant == 'regular':
            clusters_vl = entropic_clustering_variants.entropic_clustering_VL(variant_log_input=input, num_clusters=num_clusters, initialization=initialization, opt=opt)
        elif variant == 'split':
            clusters_vl = entropic_clustering_variants.entropic_clustering_split_VL(variant_log_input=input, num_clusters=num_clusters, initialization=initialization, opt=opt)
        else:
            raise ValueError("Variant has to be 'regular' or 'split'.")
        if outputshape == 'variant_log':
            return clusters_vl
        else:
            raise ValueError("When input is a variant log, output has to be 'variant_log'.")
    else:
        raise ValueError("Input not a pandas dataframe (pm4py event log) or variant log dictionary")
