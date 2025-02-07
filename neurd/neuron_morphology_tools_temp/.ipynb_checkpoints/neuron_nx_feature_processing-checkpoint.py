
import copy
import networkx as nx
from datasci_tools import numpy_dep as np


def features_list(
    G,
    limb_branch_features = True,
    features_to_ignore = ("u",),
    verbose = False):
    """
    Purpose: Find all of the current features
    
    Ex: 
    from neuron_morphology_tools from neurd import neuron_nx_feature_processing as nxf
    nxf.features_list(G)
    """
    if limb_branch_features:
        G = nxu.limb_branch_subgraph(G)
    node_df = xu.node_df(G)
    current_features = list(node_df.columns)
    if features_to_ignore is not None:
        current_features = list(np.setdiff1d(current_features,features_to_ignore))

    if verbose:
        print(f"current_features = {current_features}")
        
    return current_features

def add_node_feature(
    G,
    feature_func,
    feature_name = None,
    nodes=None,
    inplace = True,
    verbose=False,
    default_value = None,
    feature_value_dict = None,
    verbose_loop = False,
    #skip_if_exists = True,
    ):
    
    """
    Puprose: Will apply an feature_func
    to a node based on the current node values
    """
    
    if nodes is None:
        nodes = nxu.limb_branch_nodes(G)
        
    if not inplace:
        G = copy.deepcopy(G)
    
    feature_func = nu.convert_to_array_like(feature_func)
    
#     if verbose:
#         print(f"feature functions")
#         print(f"{[k.__name__ for k in feature_func]}")
        
    def default_func(*args,**kwargs):
        return default_value
    
    for att_func in feature_func:
        if "str" in str(type(att_func)):
            try:
                att_func = getattr(nxf,att_func)
            except:
#                 feature_name = att_func
#                 att_func = default_func
                continue
            
        if feature_name is None:
            curr_name = str(att_func.__name__)
        else:
            curr_name = feature_name

        if verbose:
            print(f"\n\n---Setting {curr_name}, att_func ={att_func}")

        for n in nodes:
            if feature_value_dict is not None:
                attr_value = feature_value_dict.get(n,curr_name,default_value)
            else:
                try:
                    attr_value = att_func(G.nodes[n])
                except:
                    if default_value is not None:
                        attr_value = default_value
                    else:
                        raise Exception("")
            if verbose_loop:
                print(f"For node {n}, {curr_name} = {attr_value}")
            G.nodes[n][curr_name] = attr_value
        
    return G


def skeleton_vector_upstream_x(node_dict):
    return node_dict[f"skeleton_vector_upstream"][0]
def skeleton_vector_upstream_y(node_dict):
    return node_dict[f"skeleton_vector_upstream"][1]
def skeleton_vector_upstream_z(node_dict):
    return node_dict[f"skeleton_vector_upstream"][2]
def skeleton_vector_downstream_x(node_dict):
    return node_dict[f"skeleton_vector_downstream"][0]
def skeleton_vector_downstream_y(node_dict):
    return node_dict[f"skeleton_vector_downstream"][1]
def skeleton_vector_downstream_z(node_dict):
    return node_dict[f"skeleton_vector_downstream"][2]

def skeleton_vector_downstream_theta(node_dict):
    return nu.polar_3D_from_cartesian(*node_dict[f"skeleton_vector_downstream"])[1]
def skeleton_vector_downstream_phi(node_dict):
    return nu.polar_3D_from_cartesian(*node_dict[f"skeleton_vector_downstream"])[2]

def skeleton_vector_upstream_theta(node_dict):
    return nu.polar_3D_from_cartesian(*node_dict[f"skeleton_vector_upstream"])[1]
def skeleton_vector_upstream_phi(node_dict):
    return nu.polar_3D_from_cartesian(*node_dict[f"skeleton_vector_upstream"])[2]

def width_no_spine(node_dict):
    return node_dict["width_new"]["no_spine_median_mesh_center"]

def axon_label(node_dict):
    return int(node_dict["axon_compartment"] == "axon")

def dendrite_label(node_dict):
    return int(node_dict["axon_compartment"] == "dendrite")

def compartment_proof(node_dict):
    return int(nxu.compartment_index_swc_map[node_dict["compartment"]])

def compartment_one_hot(node_dict,compartment):
    compartment = nu.convert_to_array_like(compartment)
    return int(node_dict["compartment"] in compartment)

def apical_label(node_dict):
    return compartment_one_hot(
        node_dict,
        ["apical_tuft",
        "oblique",
        "apical_shaft",
        "apical"])
def basal_label(node_dict):
    return compartment_one_hot(
        node_dict,
        ["basal"])



def feature_clip(
    row_dict,
    feature_name,
    a_min=0,
    a_max=100000,
    ):
    feature = row_dict[feature_name]
    if feature is None:
        return a_max
    
    return np.clip(feature,a_min = a_min,a_max=a_max)

def min_dist_synapses_pre_downstream_clip(
    row_dict):
    
    return feature_clip(
    row_dict,
    feature_name="min_dist_synapses_pre_downstream",
    )

def min_dist_synapses_pre_upstream_clip(
    row_dict):
    
    return feature_clip(
    row_dict,
    feature_name="min_dist_synapses_pre_downstream",
    )

auto_proof_filter_label_map = {
    "valid":0,
    "high_degree_branching_split_locations_before_filter":1,
    "low_degree_branching_split_locations_before_filter":2,
    "width_jump_up_axon_split_locations_before_filter":3,
    "axon_on_dendrite_merges_split_locations_before_filter":4,
    "high_degree_branching_dendrite_split_locations_before_filter":5,
    "width_jump_up_dendrite_split_locations_before_filter":6,
    "double_back_dendrite_split_locations_before_filter":7,
}

def auto_proof_filter_label(row_dict):
    curr_filt = row_dict["auto_proof_filter"]
    if curr_filt is None:
        curr_filt = "valid"
    return auto_proof_filter_label_map[curr_filt]

def merge_label(row_dict,merge_name):
    curr_filt = row_dict["auto_proof_filter"]
    if curr_filt is None:
        curr_filt = "clean" 
    return merge_name in curr_filt
    #return auto_proof_filter_label_map[curr_filt]

def merge_clean(row_dict):
    return merge_label(
        row_dict,
        merge_name="clean")

    
def merge_high_degree_branching_label(row_dict):
    return merge_label(
        row_dict,
        merge_name="high_degree_branching_split")

def merge_low_degree_branching_label(row_dict):
    return merge_label(
        row_dict,
        merge_name="low_degree_branching_split")

def merge_width_jump_up_axon_label(row_dict):
    return merge_label(
        row_dict,
        merge_name="width_jump_up_axon")

def merge_axon_on_dendrite_label(row_dict):
    return merge_label(
        row_dict,
        merge_name="axon_on_dendrite")

def merge_high_degree_branching_dendrite_label(row_dict):
    return merge_label(
        row_dict,
        merge_name="high_degree_branching_dendrite")

def merge_width_jump_up_dendrite_label(row_dict):
    return merge_label(
        row_dict,
        merge_name="width_jump_up_dendrite")

def merge_double_back_dendrite_label(row_dict):
    return merge_label(
        row_dict,
        merge_name="double_back_dendrite")


def merge_width_jump_up_dendrite_label(row_dict):
    return merge_label(
        row_dict,
        merge_name="width_jump_up_dendrite")



def add_skeleton_vector_features(
    G,
    use_polar_coords = True,
    verbose = False,
    **kwargs
    ):
    
    

        
    if not use_polar_coords:
        attr_funcs = [skeleton_vector_upstream_x,
        skeleton_vector_upstream_y,
        skeleton_vector_upstream_z,
        skeleton_vector_downstream_x,
        skeleton_vector_downstream_y,
        skeleton_vector_downstream_z,]
    else:
        if verbose:
            print(f"Using polar coordinates")
        attr_funcs = [skeleton_vector_downstream_theta,
        skeleton_vector_downstream_phi,
        skeleton_vector_upstream_theta,
        skeleton_vector_upstream_phi,]
    G_new_feats = nxf.add_node_feature(
        G,
        feature_func=attr_funcs,
        **kwargs
        )
    
    return G_new_feats

def add_any_missing_node_features(
    G,
    features=None,
    verbose = False,
    inplace = False,
    #default_value = 0,
    ):
    """
    Purpose:
    1) Check that all the features are requested
    2) Generate the features that are not
    """
    if features is None:
        features = features_to_output_for_gnn
    
    curr_features = nxf.features_list(G)
    features_not_computed = np.setdiff1d(features,curr_features)

    if verbose:
        print(f"features_not_computed = {features_not_computed}")

    G_new = nxf.add_node_feature(
        G,
        feature_func=features_not_computed,
        verbose = verbose,
        inplace = inplace,
        #default_value=default_value
    )

    return G_new


features_to_output_for_gnn_old = [
    "n_spines",
    "total_spine_volume",
    "n_synapses_post",
    "n_synapses_pre",
    "n_synapses_head",
    "n_synapses_neck",
    #"parent_skeletal_angle",
    "skeletal_length",
    "skeleton_vector_upstream_theta",
    "skeleton_vector_upstream_phi",
    "skeleton_vector_downstream_theta",
    "skeleton_vector_downstream_phi",
    "width_upstream",
    "width_no_spine",
    "width_downstream",
    ]

features_to_output_for_gnn = [
    # skeletal features
     'skeletal_length',
     'skeleton_vector_upstream_theta',
     'skeleton_vector_upstream_phi',
     'skeleton_vector_downstream_theta',
     'skeleton_vector_downstream_phi',
    
    #width features
     'width_upstream',
     'width_no_spine',
     'width_downstream',
    
    # synapse features
    'n_synapses_post',
    'n_synapses_pre',
    'n_synapses_head_postsyn',
    'n_synapses_neck_postsyn',
    'n_synapses_shaft_postsyn',
    'n_synapses_no_head_postsyn',
    
    # the volumes  of each
    'synapse_volume_shaft_postsyn_sum',
     'synapse_volume_head_postsyn_sum',
     'synapse_volume_no_head_postsyn_sum',
     'synapse_volume_neck_postsyn_sum',
     'synapse_volume_postsyn_sum',
    
    #spine features
    'n_spines',
    'spine_volume_sum',
    
]

features_hierarchical = [
    "soma_start_angle_max",
    "max_soma_volume",
    "n_syn_soma"]

features_to_output_for_gnn_hierarchical = features_to_output_for_gnn + features_hierarchical
    

def filter_G_features(
    G,
    features=None,
    inplace = False,
    verbose = False,
    ):
    """
    Purpose: To reduce a networkx graph to a 
    certain number of features (and all other superflous features are deleted)
    
    Ex: 
    axon_vs_dendrite_features = [
    "mesh_volume",
    "n_spines",
    "total_spine_volume",
    "n_synapses_post",
    "n_synapses_pre",
    #"n_synapse_head",
    #"parent_skeletal_angle",
    "skeletal_length",
    "skeleton_vector_upstream_theta",
    "skeleton_vector_upstream_phi",
    "skeleton_vector_downstream_theta",
    "skeleton_vector_downstream_phi",
    "width_upstream",
    "width_no_spine",
    "width_downstream",
    ]
    
    G_feat_filt = nxf.filter_G_features(
        G,
        features=axon_vs_dendrite_features,
        inplace = False,
        verbose = True,
    )

    xu.node_df(G_feat_filt)
    """


    if not inplace:
        G = copy.deepcopy(G)


    G_ret = nxf.add_any_missing_node_features(
        G,
        features = features,
        verbose = verbose
    )

    if verbose:
        print(f"Number of features after adding missing ones = {len(xu.node_df(G_ret).columns)}")

    G_ret = xu.delete_node_attributes(G_ret,attributes_not_to_delete=features)

    if verbose:
        print(f"Number of features after adding missing ones = {len(xu.node_df(G_ret).columns)}")
    
    return G_ret





#--- from neuron_morphology_tools ---
from . import neuron_nx_utils as nxu

#--- from datasci_tools ---
from datasci_tools import networkx_utils as xu
from datasci_tools import numpy_utils as nu

from . import neuron_nx_feature_processing as nxf
