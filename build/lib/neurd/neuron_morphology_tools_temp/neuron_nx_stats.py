'''

Purpose: To create functions that can 
compute statistics over a graph object
(whether it be a full neuron graph or a subgraph)


'''
import pandas as pd
import time


node_identifier = "u"

def summary_statistic_over_attributes(
    attributes,
    summary_statistic = "mean",
    summary_statisic_args = None,
    weight = None,
    G=None,
    node_df=None,
    verbose = False,
    return_df = False,
    append_statistic_name = False,
    ):
    if node_df is None:
        node_df = xu.node_df(G)
        
    node_df = node_df.query(f"{node_identifier} != 'S0'")
    
    return pu.summary_statistic_over_columns(
        columns = attributes,
        df = node_df,
        summary_statistic = summary_statistic,
        summary_statisic_args = summary_statisic_args,
        verbose = verbose,
        return_df = return_df,
        weight=weight,
        append_statistic_name=append_statistic_name
    )

def n_branches(G):
    return len(nxu.limb_branch_nodes(G))
    
def starting_coordinate(
    G,
    attribute = "endpoint_upstream",
    return_dict = False):
    """
    Purpose: To find the startng centroid of a graph

    Pseudocode: 
    1) Get the most upstream node
    """

    node = nxu.most_upstream_node(G)
    array = G.nodes[node][attribute]
    if not return_dict:
        return array
    else:
        return {f"starting_coordinate_{k}":v for k,v in zip(["x","y","z"],
                                                    array)}

def summary_statistic_over_dynamic_attribute(
    G,
    node,
    attribute,
    prefix = None,
    category_columns = None,
    attribute_summary_dicts = (dict(columns="volume",
                                   summary_statistic = "sum",
                                    summary_statisic_args = None),),
    verbose = False,
    default_value = None,
    debug_time = False,
    ):

    """
    Purpose: To get the dynamic attributes
    dictionary for a node in the graph
    """
    if default_value is None:
        default_value = {}
    
    if attribute_summary_dicts is not None:
        attribute_summary_dicts = list(attribute_summary_dicts)
    
    if debug_time:
        st = time.time()
        
    if prefix is None:
        prefix = attribute.replace("_data","")

    node_dict = G.nodes[node].copy()
    
    if len(node_dict[attribute]) == 0:
        return default_value
    
    
    df = pd.DataFrame.from_records(node_dict[attribute])
    
    if debug_time:
        print(f"Time for attribute dataframe = {time.time() - st}")
        st = time.time()

    return pu.summary_statistics_over_columns_by_category(
        df,
        prefix = prefix,
        category_columns = category_columns,
        attribute_summary_dicts = attribute_summary_dicts,
        add_counts_summary = True,
        verbose = verbose,
        special_count_name = True,
        debug_time = debug_time
    )

def summary_statistic_over_synapses(
    G,
    node,
    verbose = False,
    **kwargs):
    """
    Gets the number and total volume of synapses of
    a certain head/neck/shaft type and presyn/postsyn type
    
    (and does for every super category)
    
    Return
    ------
        Dictionary with sums/counts
        
    Example Output
    --------------
        {'synapse_volume_no_head_postsyn_sum': 48932,
        'n_synapses_no_head_postsyn': 16,
        'synapse_volume_no_head_sum': 48932,
        'n_synapses_no_head': 16,
        'synapse_volume_shaft_postsyn_sum': 186960,
        'n_synapses_shaft_postsyn': 44,
        'synapse_volume_shaft_sum': 186960,
        'n_synapses_shaft': 44,
        'synapse_volume_head_postsyn_sum': 696432,
        'n_synapses_head_postsyn': 65,
        'synapse_volume_head_sum': 696432,
        'n_synapses_head': 65,
        'synapse_volume_neck_postsyn_sum': 4680,
        'n_synapses_neck_postsyn': 3,
        'synapse_volume_neck_sum': 4680,
        'n_synapses_neck': 3,
        'synapse_volume_postsyn_sum': 937004,
        'n_synapses_postsyn': 128,
        'synapse_volume_sum': 937004,
        'n_synapses': 128}
    """
    
    return nxst.summary_statistic_over_dynamic_attribute(
    G,
    node = node,
    attribute = "synapse_data",
    category_columns = ["head_neck_shaft","syn_type"],
    verbose = verbose,
    **kwargs
    )

def add_any_missing_node_features(G,**kwargs):
    return nxf.add_any_missing_node_features(G)

def summary_statistic_over_spines(
    G,
    node,
    verbose = False,
    **kwargs):
    """
    Calculates the total spine volume and number of spines
    
    Return
    ------
        Dictionary with sums/counts
        
    Example Output
    --------------
        {'spine_volume_sum': 26567481267.249374, 'n_spines': 88}
    """
    
    return nxst.summary_statistic_over_dynamic_attribute(
    G,
    node = node,
    attribute = "spine_data",
    category_columns = None,#["head_neck_shaft","syn_type"],
    verbose = verbose,
    **kwargs
    )

def add_summary_statistic_over_dynamic_attributes_to_G(
    G,
    attributes = ("synapses","spines"),
    verbose = False,
    ):
    
    
    for node in nxu.limb_branch_nodes(G):
        st = time.time()
        #nxst = reload(nxst)
        curr_dict = {}
        for s in attributes:
            curr_dict.update(getattr(nxst,f"summary_statistic_over_{s}")(G,node))
            
        G.nodes[node].update(curr_dict)                         
                            
        if verbose:
            print(f"Node {node}: {time.time() - st}")
            
    return G
            
def skeleton_bounding_box(
    G,
    soma_relative = False,
    soma_center = None,
    verbose = False,
    return_dict = False,
    
    ):
    """
    Purpose: To compute the bounding box of a certain limb
    based on the skeleton data

    Pseudocode:
    1) Get the skeleton nodes
    2) Computes the bounding box of the scatter points
    3) If soma_relative flag set --> subtracts the soma
    """


    coordinates = nxu.skeleton_nodes(G)

    if len(coordinates) == 0:
        bbox = -1*np.ones((2,3))
    else:
        bbox = nu.bounding_box(coordinates)
        if soma_relative:
            if soma_center is None:
                soma_center = nxu.soma_center(G) 
            bbox = bbox - soma_center

    if verbose:
        print(f"bbox (soma_relative = {soma_relative}):\n{bbox}")

    if return_dict:
        return dict(
            bbox_x_min=bbox[0][0],
            bbox_y_min=bbox[0][1],
            bbox_z_min=bbox[0][2],
            
            bbox_x_max=bbox[1][0],
            bbox_y_max=bbox[1][1],
            bbox_z_max=bbox[1][2],
        )
        
    return bbox
            
            
# ------------ For computing the overall statistics of a graph ----------#

statistics_survey_attributes_to_sum = [
            'skeletal_length',
            'n_spines',
            'spine_volume_sum',
            'n_synapses',
            'n_synapses_post',
             'n_synapses_pre',
            'n_synapses_head_postsyn',
             'n_synapses_neck_postsyn',
             'n_synapses_no_head_postsyn',
             'n_synapses_shaft_postsyn',
             'n_synapses_spine_postsyn',
             'synapse_volume_shaft_postsyn_sum',
            'synapse_volume_head_postsyn_sum',
            'synapse_volume_neck_postsyn_sum',
            'synapse_volume_no_head_postsyn_sum',
            'synapse_volume_spine_postsyn_sum',
]

statistics_survey_attributes_to_mean = [
            'width_no_spine',
]

statistics_survey_attributes_to_max = [
            "soma_start_angle",
]

statistics_survey_attributes_to_min = [
            "soma_start_angle",
]
def statistics_survey_from_graph(
    G,
    attributes_to_sum=None,
    attributes_to_mean = None,
    attributes_to_max=None,
    attributes_to_min = None,
    soma_center = None,
    compute_compartments = True,
    add_features = True,
    
    # for dividing some features
    divisor = 1_000_000_000,
    attributes_to_divide = dict(
        spine_volume_sum = 1_000_000_000,
        
    ),
    verbose = False,
    
    ):
    """
    Purpose: List of graph statistics 
    that by default would want to measure

    stats that need to be manually calculatd
    - n_nodes

    """
    if add_features:
        G = nxst.add_summary_statistic_over_dynamic_attributes_to_G(G)
        G = nxst.add_any_missing_node_features(G)
        
    
    if attributes_to_sum is None:
        attributes_to_sum = statistics_survey_attributes_to_sum
        
    if attributes_to_mean is None:
        attributes_to_mean = statistics_survey_attributes_to_mean
    
    if attributes_to_max is None:
        attributes_to_max = statistics_survey_attributes_to_max
        
    if attributes_to_min is None:
        attributes_to_min = statistics_survey_attributes_to_min

    if soma_center is None:
            soma_center = nxu.soma_center(G)

    all_data_dicts = []

    compartment_restrictions = [None]
    if compute_compartments:
        compartment_restrictions += ['axon', 'oblique', 'apical', 'apical_shaft','apical_tuft', 'basal']
        
    for comp in compartment_restrictions:
        if comp is None:
            G_stat = G
            prefix = ''
        else:
            G_stat = xu.subgraph_from_node_query(G,f"compartment == '{comp}'")
            prefix = f'{comp}_'
            
        if verbose:
            print(f"--- Working on compartment {comp} ({len(G_stat.nodes())} nodes) ----")

        if len(G_stat.nodes()) == 0:
            data_dict = {}

        else:
            # want to iterate over whole graph and compartment
            data_dict = dict(
                n_branches = nxst.n_branches(G_stat)
            )

            attributes_to_sum_dict = nxst.summary_statistic_over_attributes(
                attributes = attributes_to_sum,
                summary_statistic = "sum",
                G=G_stat,
                return_df=False,
            )
            
            if verbose:
                print(f"attributes_to_sum_dict = {attributes_to_sum_dict}")

            attributes_to_mean_dict = nxst.summary_statistic_over_attributes(
                attributes = attributes_to_mean,
                summary_statistic = "mean",
                weight="skeletal_length",
                G=G_stat,
                return_df=False,
            )

            attributes_to_max_dict = nxst.summary_statistic_over_attributes(
                attributes = attributes_to_max,
                summary_statistic = "max",
                append_statistic_name=True,
                G=G_stat,
                return_df=False,
            )

            attributes_to_min_dict = nxst.summary_statistic_over_attributes(
                attributes = attributes_to_min,
                summary_statistic = "min",
                append_statistic_name=True,
                G=G_stat,
                return_df=False,
            )


            starting_coordinate_dict = nxst.starting_coordinate(G_stat,return_dict = True)
            bbox_dict = nxst.skeleton_bounding_box(G_stat,return_dict = True)

            relative_bbox_dict = nxst.skeleton_bounding_box(
                G_stat,
                soma_relative=True,
                return_dict=True,
                soma_center = soma_center)
            relative_bbox_dict = {f"{k}_soma_relative":v for k,v in relative_bbox_dict.items()}

            data_dict = gu.merge_dicts([
                data_dict,
                attributes_to_sum_dict,
                attributes_to_mean_dict,
                attributes_to_max_dict,
                attributes_to_min_dict,
                starting_coordinate_dict,
                bbox_dict,
                relative_bbox_dict

                ])

            data_dict = {f"{prefix}{k}":v for k,v in data_dict.items()}

        all_data_dicts.append(data_dict)

    all_data_dicts = gu.merge_dicts(all_data_dicts)
    
    if attributes_to_divide is not None:
        for att,divisor in attributes_to_divide.items():
            for k,v in all_data_dicts.items():
                if att in k:
                    all_data_dicts[k] = all_data_dicts[k]/divisor
                
    return all_data_dicts

def skeletal_length_downstream(
    G,
    node,
    include_self=True,
    ):
    
    return xu.sum_downstream_attribute(
        G,
        node,
        attribute="skeletal_length",
        include_self = include_self
    )

#--- from neuron_morphology_tools ---
from . import neuron_nx_feature_processing as nxf
from . import neuron_nx_utils as nxu

#--- from datasci_tools ---
from datasci_tools import general_utils as gu
from datasci_tools import networkx_utils as xu
from datasci_tools import numpy_utils as nu
from datasci_tools import pandas_utils as pu

from . import neuron_nx_stats as nxst
