'''

Purpose: tools that will help process a neuron that
is represented as a networkx graph


'''
from pykdtree.kdtree import KDTree
import copy
import matplotlib.pyplot as plt
import networkx as nx
from datasci_tools import numpy_dep as np
import pandas as pd
import time
import numpy as np

soma_node_name_global = "S0"
node_label = "u"

auto_proof_filter_name_default = "auto_proof_filter"

dynamic_attributes_default = (
 'spine_data',
 'synapse_data',
 'width_data',
  'skeleton_data')


compartments_excitatory = [
    'apical',
    'apical_shaft',
    'apical_tuft',
    'basal',
    'axon',
    'oblique',
]

compartments_inhibitory = [
    'axon',
    'dendrite',
]

compartments = compartments_excitatory + ["dendrite"]

compartment_colors = {'apical': 'blue',
 'apical_shaft': 'aqua',
 'apical_tuft': 'purple',
 'basal': 'brown',
 'axon': 'red',
 'oblique': 'green',
 'dendrite': 'lightsteelblue',
 'apical_total': 'magenta',
 'soma': 'black',
  None:"blue"}


# --- global parameters for vector computations ---
default_upstream_dist_max = 6000
default_scholl_interval = 10_000




def delete_attributes(
    G,
    inplace = True,
    attributes=None,
    verbose = False
    ):
    
    if not inplace:
        G = copy.deepcopy(G)
    
    if attributes is None:
        attributes = list(nxu.dynamic_attributes_default)
    
    return xu.delete_node_attributes(
        G,
        attributes=attributes,
        verbose = verbose)


def name_from_G(G,append=None):
    graph_attr = xu.graph_attr_dict(G)
    segment_id = graph_attr.get("segment_id",None)
    split_index = graph_attr.get("split_index",None)
    nucleus_id = graph_attr.get("nucleus_id",None)
    
    name = f"seg_{segment_id}_{split_index}_nucleus_{nucleus_id}"
    if append is not None:
        name += f"_{append}"
    return name
    
def save_G(
    G,
    filepath=None,
    filename_append = None,
    delete_dynamic_attributes = True,
    verbose = False
    ):
    
    """
    Purpose: To save the graph
    nxu.save_G(G_ax,filename_append="axon_high_fid")
    """
    
    if delete_dynamic_attributes:
        G = nxu.delete_attributes(
            G,
            inplace = False,
            verbose = False)
    if filepath is None:
        filepath = f"./{nxu.name_from_G(G,append=filename_append)}"
    
    return su.compressed_pickle(G,filepath)

def load_G(filepath):
    return su.decompress_pickle(filepath)
    

def skeletal_length_on_path(G,path):
    return np.sum([G.nodes[k]["skeletal_length"] for k in path])

def calculate_soma_distance_to_data_objs(
    G,
    verbose = False,
    data_to_update = ("synapse_data","spine_data","width_data"),                     
    ):
    """
    Purpose: To set the soma distance for all attributes

    Pseudocode: 
    Iterate through all of the nodes with L in name
    1) Find the path back to the soma
    2) Calculate the sum of skeletal length back to the soma
    for each attrbute
    3) Add skeletal length to the upstream dist to get the soma distance

    """
    for node in G.nodes():
        if "L" not in node:
            continue
        if verbose:
            print(f"--working on node {node}--")
        inbetween_node_path = xu.shortest_path(G,"S0",node)[1:-1]
        data_to_update = ("synapse_data","spine_data","width_data")

        soma_dist_on_path = nxu.skeletal_length_on_path(G,inbetween_node_path)

        if verbose:
            print(f"inbetween_node_path = {inbetween_node_path}")
            print(f"soma_dist_on_path = {soma_dist_on_path}")

        for dtype in data_to_update:
            for obj in G.nodes[node][dtype]:
                obj["soma_distance"] = obj["upstream_dist"] + soma_dist_on_path
                
                
def draw_tree(
    G,
    draw_type = "dot",#"twopi" (the circular), "circo" (makes square like)
    node_size=4000,
    font_size = 20,
    font_color = "white",
    figsize=(32,32),
    compartment_colors = True,
    compartment_color_dict = None,
    **kwargs):
    
    if compartment_colors:
        node_color = nxu.node_compartment_color_dict(
            G = G,
            compartment_color_dict=compartment_color_dict
        )
        
        node_color = list(node_color.values())
    else:
        node_color = None
    
    xu.draw_tree(
        G,
        draw_type = draw_type,#"twopi" (the circular), "circo" (makes square like)
        node_size=node_size,
        font_size = font_size,
        font_color = font_color,
        figsize=figsize,
        node_color=node_color,
        **kwargs)
    
skeletal_length_min_starter_branches_global = 700
def small_starter_branches(
    G,
    skeletal_length_min = None,
    soma_node_name = None,
    verbose = False,
    ):
    """
    Purpose: To identify starting nodes may want to remove

    Pseuodocde: 
    1) Look for those with small lengths
    2) Look for those that neighbor the soma
    """
    if skeletal_length_min is None:
        skeletal_length_min = skeletal_length_min_starter_branches_global

    if soma_node_name is None:
        soma_node_name= soma_node_name_global
        
    if nxu.soma_only_graph(G):
        return []

    small_nodes = xu.nodes_from_node_query(G,f"(skeletal_length < {skeletal_length_min})")
    if verbose:
        print(f"small_nodes = {small_nodes}")
    small_soma_neighbors= []
    if len(small_nodes) > 0:
        soma_partners = xu.downstream_nodes(G,soma_node_name)
        if verbose:
            print(f"soma_partners= {soma_partners}")
        small_soma_neighbors = list(set(small_nodes).intersection(set(soma_partners)))
        
    if verbose:
        print(f"small_soma_neighbors= {small_soma_neighbors}")
        
    return small_soma_neighbors



def remove_node(
    G,
    node,
    inplace = False,
    verbose = False,
    maintain_skeleton_connectivity = True,
    remove_all_downstream_nodes = False,
    **kwargs
    ):
    """
    Purpose: To remove a node from the graph
    and to reconnect the downstream children

    Pseudocode: 
    1) Get any downstream nodes 
        a. (if none then just delete and return)
    2) For each downstream node:

    Attributes that need to be changed: 

    # if ask to alter skeleton
    endpoint_upstream --> parent endpoint_upstream
    skeleton_coordinates --> add parent endpoint_upstream

    #the soma information
    'soma_start_vec': array([-0.375219  , -0.91207943, -0.16529313]),
     'soma_start_angle': 24.21}
     
     
    Ez: 
    new_G = nxu.remove_node(
    G,
    node="L0_22",
    inplace = False,
    verbose = True,
    maintain_skeleton_connectivity = True,
        )

    nxu.soma_connected_nodes(new_G)
    new_G.nodes["L0_19"]["skeleton_data"],new_G.nodes["L0_20"]["skeleton_data"]
    
    
    Ex 2: Deleting all downstream nodes
    nodes = ["L1_10","L1_2","L1_8","L0_20"]
    G_del = nxu.remove_node(
        G,
        node=nodes,
        inplace=False,
        verbose = True,
        remove_all_downstream_nodes = True
    )

    """

    if not inplace:
        G = copy.deepcopy(G)

    nodes = nu.convert_to_array_like(node)
    for node in nodes:
        if verbose:
            print(f"--Working on removing node {node}")
        
        if node not in G:
            if verbose:
                print(f"node {node} wasn't in graph nodes so continuing")
            continue
            
        downstream_nodes = xu.downstream_nodes(G,node)

        if verbose:
            print(f"downstream_nodes = {downstream_nodes}")

        if len(downstream_nodes) > 0 and not remove_all_downstream_nodes:
            if node in nxu.soma_connected_nodes(G):
                soma_vals = ["soma_start_vec","soma_start_angle"]
                if verbose:
                    print(f"Adding {soma_vals} to the downstream nodes")
                for att in soma_vals:
                    for n in downstream_nodes:
                        G.nodes[n][att]  =  G.nodes[node][att]

            if maintain_skeleton_connectivity:
                endpoint_upstream = G.nodes[node]["endpoint_upstream"]
                width_upstream = G.nodes[node]["width_new"]["no_spine_median_mesh_center"]
                
                skeletal_length_upstream = G.nodes[node]["skeletal_length"]
                if verbose:
                    print(f"Adding endpoint_upstream {endpoint_upstream} to the downstream nodes skeleton")
                for n in downstream_nodes:
                    G.nodes[n]["endpoint_upstream"]  = endpoint_upstream
                    G.nodes[n]["skeleton_data"] = np.concatenate([[endpoint_upstream],G.nodes[n]["skeleton_data"]])
                    width_upstream_curr = width_upstream
                    for i,(k) in enumerate(G.nodes[n]["width_data"]):
                        if np.isinf(width_upstream):
                            width_upstream_curr = G.nodes[n]["width_data"][0]["width"]
                        
                        G.nodes[n]["width_data"][i]["upstream_dist"] = G.nodes[n]["width_data"][i]["upstream_dist"] + skeletal_length_upstream
                    G.nodes[n]["width_data"] = [dict(upstream_dist=skeletal_length_upstream,
                                                     width = width_upstream_curr)] + G.nodes[n]["width_data"]
                    
        if not remove_all_downstream_nodes:
            xu.remove_node_reattach_children_di(G,node,inplace = True)
        else:
            all_downstream_nodes = xu.all_downstream_nodes(G,node)
            total_nodes_to_delete = [node] 
            if len(all_downstream_nodes) > 0:
                total_nodes_to_delete += list(all_downstream_nodes)
                
            if verbose:
                print(f"Removing all downstream nodes along with node {node}: {total_nodes_to_delete}")
            G = xu.remove_nodes_from(G,total_nodes_to_delete)
            
    return G
    

def remove_small_starter_branches(
    G,
    skeletal_length_min= None,
    inplace = False,
    verbose = False,
    maintain_skeleton_connectivity = True,
    loop_until_fail = True,
    **kwargs):
    """
    Purpose: To remove small starter branches from 
    the graph
    
    Ex: 
    nxu.remove_small_starter_branches(G,verbose = True)
    
    Ex 2:
    new_G = nxu.remove_small_starter_branches(
        G,
        skeletal_length_min = 100000000000,
    )
    """

    if not inplace:
        G = copy.deepcopy(G)

    while True:
        sm_st_branches = nxu.small_starter_branches(
            G,
            verbose = verbose,
            skeletal_length_min = skeletal_length_min,
            **kwargs)

        if verbose:
            print(f"Soma Starter Branches = {sm_st_branches}")

    #     for s_b in sm_st_branches:
    # #         if verbose:
    # #             print(f"Removing {s_b}")
        nxu.remove_node(
            G,
            sm_st_branches,
            verbose= verbose,
            inplace = True,
            maintain_skeleton_connectivity=maintain_skeleton_connectivity,
            **kwargs)

        if not loop_until_fail:
            break
        elif len(sm_st_branches) == 0:
            break
        else:
            continue

    return G

filter_small_starter_branches = remove_small_starter_branches 

def soma_radius(
    G,
    stat = "mean",
    verbose = False
    ):
    """
    Purpose: To find the [stat] distance of the soma from the neighbors
    """
    soma_neighbors = xu.downstream_nodes(G,soma_node_name_global)
    s_center = G.nodes[soma_node_name_global]["mesh_center"]
    endpoints_dist = np.linalg.norm([G.nodes[k]["endpoint_upstream"] - s_center for k in soma_neighbors],axis = 1)
    stat_dist = getattr(np,stat)(endpoints_dist)
    if verbose:
        print(f"soma_neighbors = {soma_neighbors}")
        print(f"endpoints_dist = {endpoints_dist}")
        print(f"{stat} dist = {stat_dist}")
    return stat_dist
    
    
# --------------- For exporting graph as differnt file types -------------

compartment_index_swc_map = dict(
        dendrite = 0,
        soma = 1,
        axon = 2,
        basal = 3,
        apical = 4,
        apical_tuft = 4,
        oblique = 4,
        apical_shaft = 4,
        custom = 5,
        undefined = 6,
        glia = 7
    )

def export_swc_dicts(
    G,
    use_skeletal_coordinates = True,
    soma_node_name = soma_node_name_global,
    default_skeleton_pt = "mesh_center",
    default_compartment = "basal",#"undefined",
    center_on_soma= True,
    coordinate_divisor = 1000,
    width_divisor = 1000,
    verbose = False,
    return_df = False,
    ):
    """
    Purpose: To convert a Graph Neuron Object
    into a SWC file dict objects by creating dictionaries of all of the points

    Pseudocode: 
    In a depth first search manner of searching
    For each node: 
        1) get the skeleton points
        2) Get the compartment
        3) Get the parent idx

    """

    index = 0

    if soma_node_name in G.nodes():
        center_point = G.nodes[soma_node_name]["mesh_center"]
        source_node = soma_node_name
        node_order = nx.dfs_preorder_nodes(G,source = soma_node_name)
        
    else:
        center_point = np.mean([G.nodes[k]["mesh_center"]
                               for k in G.nodes()],axis=0)
        source_node = xu.most_upstream_node(G)
        node_order = nx.dfs_preorder_nodes(
            G,source = source_node)

    
    node_to_index_map = dict()
    swc_dicts = []
    for n in node_order:
        if verbose:
            print(f"\n---Working on {n}")
        node_to_index_map[n] = []

        if use_skeletal_coordinates and n != soma_node_name:
            skeleton_pts = G.nodes[n]["skeleton_data"][1:]
        else:
            skeleton_pts = np.array([G.nodes[n][default_skeleton_pt]])

        curr_compartment  = G.nodes[n]["compartment"]
        if n == source_node:
            if n != soma_node_name: 
                curr_compartment = "soma"
            parent_idx = -1
        else:
            parent_name = xu.parent_node(G,n)
            parent_idx = node_to_index_map[parent_name][-1]
            
        if n == soma_node_name:
            width_points = [nxu.soma_radius(G,verbose = False)]
        else:
            if use_skeletal_coordinates:
                width_points = [k["width"] for k in G.nodes[n]["width_data"]]
            else:
                width_points = [G.nodes[n]["width_new"]["no_spine_median_mesh_center"]]

        
        if curr_compartment is None:
            curr_compartment = default_compartment
            
#         if curr_compartment == "dendrite":
#             curr_compartment = default_compartment

        if center_on_soma:
            skeleton_pts = skeleton_pts - center_point

        if verbose:
            print(f"skeleton_pts = {skeleton_pts}")
            print(f"parent_idx= {parent_idx}")
            print(f"curr_compartment= {curr_compartment}")

        for i,coord in enumerate(skeleton_pts):
            index += 1

            if i == 0:
                curr_parent_idx = parent_idx
            else:
                curr_parent_idx = index - 1

            curr_dict = {
                "n":index,
                "type":compartment_index_swc_map[curr_compartment],
                "x": coord[0]/coordinate_divisor,
                "y":coord[1]/coordinate_divisor,
                "z":coord[2]/coordinate_divisor,
                "radius": width_points[i]/width_divisor,
                "parent":curr_parent_idx,
            }
            node_to_index_map[n].append(index)

            swc_dicts.append(curr_dict) 
    
    if return_df:
        pd.DataFrame.from_records(swc_dicts)
    return swc_dicts

def export_swc_df(G,**kwargs):
    return pd.DataFrame.from_records(export_swc_dicts(G,return_df=True,**kwargs))


def export_swc_file(
    G,
    filename=None,
    filename_append = None,
    directory="./",
    filepath = None,
    verbose = False,
    header =True,
    **kwargs):
    """
    Purpose: Create a SWC file from 
    graph object
    """
    
    swc_dicts= nxu.export_swc_dicts(G,verbose = False)
    if filename is None:
        filename = f"seg_{G.graph['segment_id']}_split_{G.graph['split_index']}_nucleus_{G.graph['nucleus_id']}"

    if filename_append is not None:
        filename = f"{filename}_{filename_append}"
        
    if filename[-4:] != ".swc":
        filename += ".swc"
        
    return fileu.file_from_dicts(
        swc_dicts,
        header = header,
        filename = filename,
        directory = directory,
        filepath = filepath,
        seperation_character=" ",
        verbose = verbose
    )
    



# -------------- computing morphology metrics using morphopy -----

def swc_df(G,flip_z = True,**kwargs):
    swc = nxu.export_swc_df(G)
    swc_df = mpu.swc_rotated_resampled(swc,resample=False,flip_z = flip_z,**kwargs)
    return swc_df

def morphometrics(
    G=None,
    swc = None,
    apply_basal_dendrite_swap = True,
    **kwargs
    ):
    
    if swc is None:
        swc = nxu.swc_df(G)
        
    if apply_basal_dendrite_swap:
        # --- fixed so dendrite compartment will be changed to basal and recognized as dendrite ---
        from datasci_tools import pandas_utils as pu
        def basal_rename(row):
            if row["type"] == compartment_index_swc_map["dendrite"]:
                return compartment_index_swc_map["basal"]
            else:
                return row["type"]

        swc["type"] = pu.new_column_from_row_function(swc,basal_rename).astype('int')
        #return swc
    
    #swc_df = mpu.swc_rotated_resampled(swc,resample=False,flip_z = True)
        
    return mpu.morphometrics(swc=swc,**kwargs)


# ------------ for querying neuron graphs ----
def axon_dendrite_nodes(
    G,
    compartment="axon",
    return_node_df=False,):
    
    node_df = xu.node_df(G)
    try:
        node_df = node_df.query(f"axon_compartment=='{compartment}'")
    except:
        pass 
    
    if return_node_df:
        return node_df
    else:
        return node_df[node_label].to_numpy()
    
def axon_nodes(
    G,
    return_node_df=False,
    **kwargs):
    
    return nxu.axon_dendrite_nodes(
    G,
    compartment="axon",
    return_node_df=return_node_df,**kwargs)

def dendrite_nodes(
    G,
    return_node_df=False):
    
    return nxu.axon_dendrite_nodes(
    G,
    compartment="dendrite",
    return_node_df=return_node_df,)

def axon_dendrite_subgraph(
    G,
    compartment,
    include_soma = True,
    verbose = False,
    remove_node_method = True,
    ):
    
    compartment_nodes =  list(axon_dendrite_nodes(
    G,
    compartment=compartment,
    return_node_df=False,))
    
    if verbose:
        print(f"compartment_nodes = {compartment_nodes}")
        
    if include_soma:
        compartment_nodes.append(nxu.soma_node_name_global)
        
    if remove_node_method:
        nodes_to_remove = np.setdiff1d(list(G.nodes()),compartment_nodes)
        return nxu.remove_node(
            G,
            nodes_to_remove,
        )
    else:
        return G.subgraph(compartment_nodes).copy()
        
def axon_subgraph(
    G,
    include_soma = True,
    verbose = False,
    ):
    return nxu.axon_dendrite_subgraph(
    G,
    compartment="axon",
    include_soma = include_soma,
    verbose = verbose,
    )

def dendrite_subgraph(
    G,
    include_soma = True,
    verbose = False,
    ):
    return nxu.axon_dendrite_subgraph(
    G,
    compartment="dendrite",
    include_soma = include_soma,
    verbose = verbose,
    )

def plot(
    G,
    verbose = False,
    xlim = None,
    ylim=None,
    zlim = None,
    **kwargs):
    
    
    #st = time.time()
    swc_df = nxu.swc_df(G)
    ntree_obj = mpu.ntree_obj_from_swc(swc=swc_df)
    mpu.plot_ntree(
        ntree_obj,
        xlim = xlim,
        ylim=ylim,
        zlim = zlim,
        **kwargs)

# --------- mapping errors to nodes ----------
def limb_from_node_name(name):
    return int(name[1:name.find("_")])

def branch_from_node_name(name):
    return int(name[name.find("_")+1:])

def limb_branch_nodes(G):
    return [k for k in G.nodes() if "S" not in k]

def limb_branch_subgraph(G):
    return G.subgraph(nxu.limb_branch_nodes(G)).copy()

def all_limb_idxs_in_G(G):
    return np.sort(np.unique([nxu.limb_from_node_name(k)
                     for k in limb_branch_nodes(G)]))

def all_limb_graphs(G,return_idxs = False):
    limb_idxs = nxu.all_limb_idxs_in_G(G)
    return_graphs = [limb_graph(G,k) for k in limb_idxs]
    if return_idxs:
        return return_graphs,limb_idxs
    else:
        return return_graphs
    
def all_limb_graphs_off_soma(G,verbose = False):
    components = xu.connected_components(
        xu.remove_nodes_from(G,[nxu.soma_node_name_global]
                            )
    )
    if verbose:
        print(f"components = {components}")
    
    return_graphs = [G.subgraph(subgraph_nodes).copy()
                     for subgraph_nodes in components]
    
    if verbose:
        print(f"# of limb subgraphs off soma = {len(return_graphs)}")
    
    return return_graphs
    
    

def limb_graph(
    G,
    limb_idx=None,
    most_upstream_node = None,
    branches_idx = None,
    verbose = False,
    ):
    """
    Purpose: To return a graph of a certain limb
    
    Ex: nxu.limb_graph(G_ax,limb_idx = 3,verbose = True)
    """
    
    if limb_idx is not None:
        if verbose:
            print(f"Using the limb_idx method")
        if type(limb_idx) == str:
            limb_idx = int(limb_idx[1:])

        if limb_idx == -1:
            if verbose:
                print(f"Returning whole graph")
            return G

        subgraph_nodes = [k for k in nxu.limb_branch_nodes(G)
                         if nxu.limb_from_node_name(k) == limb_idx]
        
        if branches_idx is not None:
            subgraph_nodes = [k for k in subgraph_nodes
                             if nxu.branch_from_node_name(k) in branches_idx]

            if verbose:
                print(f"subgraph_nodes after branch restriction: {subgraph_nodes}")
    elif most_upstream_node is not None:
        if verbose:
            print(f"Using the most upstream method")
        subgraph_nodes = xu.all_downstream_nodes(
            G,
            most_upstream_node,
            include_self=True)
    else:
        raise Exception("")
    if verbose:
        print(f"subgraph_nodes after restriction: {subgraph_nodes}")
        
    return G.subgraph(subgraph_nodes).copy()


def node_match_by_dict(
    dict1,
    dict2,
    attributes = (
        "endpoint_upstream",
        #"endpoint_downstream",
        "skeleton_vector_upstream",
    ),
    verbose = False,
    ):
    """
    Purpose: To compare whether two
    nodes are the same or not
    
    Ex: 
    node_match_by_dict(
    G_ax.nodes["L0_0"],
    G_ax.nodes["L0_1"],
    verbose = True
    )
    """
    
    match_flag = True
    
    for a in attributes:
        val1 = dict1[a]
        val2 = dict2[a]
        
        if "array" in str(type(val1)):
            comp_result = np.array_equal(val1,val2)
        elif type(val1) == list:
            comp_result = np.all([k1==k2 for k1,k2 in zip(val1,val2)])
        else:
            comp_result  = val1==val2
            
        if comp_result == False:
            match_flag = False
            if verbose:
                print(f"Did not have equal {a}: {val1},{val2}")
            
    
    return match_flag


def node_map(
    G_source,
    G_target,
    verbose = False,
    ):
    """
    Purpose: To find the matches of all the nodes
    of one graph to another:

    Pseudocode: 
    For each node in the source graph
        Iterate through the target graph nodes
            Compare the certain properties, and if all match then assign as the correct mapping
            
            
    nxu.node_map(
        G_source = G_ax,
        G_target = G_proof,
        verbose = True
    )
    """
    source_to_target_map = dict()

    for n in nxu.limb_branch_nodes(G_source):

        limb_idx = nxu.limb_from_node_name(n)
        matching_nodes = []
        for nt in nxu.limb_branch_nodes(G_target):

            limb_idx_t = nxu.limb_from_node_name(nt)
            if limb_idx != limb_idx_t:
                continue

            match_result = nxu.node_match_by_dict(
                G_source.nodes[n],
                G_target.nodes[nt],
            )

            if match_result:
                matching_nodes.append(nt)

        if verbose:
            print(f"Node {n} --> {matching_nodes}")

        if len(matching_nodes) > 1:
            raise Exception("More than 1 match")

        if len(matching_nodes) == 0:
            source_to_target_map[n] = None
            continue

        match = matching_nodes[0]

        if nxu.limb_from_node_name(n) != nxu.limb_from_node_name(match):
            raise Exception("Not on same limb")

        source_to_target_map[n] = match

    #check to see if unique
    final_matches = list([k for k in source_to_target_map.values()
                          if k is not None])
    if len(final_matches) != len(np.unique(final_matches)):
        raise Exception("Some repeats")
            
    return source_to_target_map
            
def nodes_without_match(
    G_source,
    G_target,
    verbose = False):
    
    curr_map = nxu.node_map(
        G_source,
        G_target)
    
    no_map = np.array([k for k,v in curr_map.items() if v is None])
    
    if verbose:
        print(f"Nodes with no match = {len(no_map)}")
        
    return no_map

def skeleton_from_node(
    G,
    n):
    """
    Ex: nxu.skeleton_from_node(G_obj,n = "L0_16")
    """
    skeleton_points = G.nodes[n]["skeleton_data"]
    sk_idx = np.vstack([
        np.arange(0,len(skeleton_points)-1),
        np.arange(1,len(skeleton_points))]).T
    return skeleton_points[sk_idx]
    
def skeleton_coordinates_from_G(
    G,
    include_upstream_node = False,
    verbose = False,
    return_node_names = False,
    ):
    """
    Purpose: To output the skeleton points of a graph
    and the node names those skeleton points came from
    
    Ex: 
    sk_pts,sk_names = nxu.skeleton_coordinates_from_G(
        G_ax,
    return_node_names=True,
    verbose = True)
    """

    total_skeleton_points = []
    total_node_names = []

    for n in nxu.limb_branch_nodes(G):
        skeleton_points = G.nodes[n]["skeleton_data"]
        if not include_upstream_node:
            upstream_coordinate = G.nodes[n]["endpoint_upstream"].reshape(-1,3)
            skeleton_points = nu.setdiff2d(skeleton_points,upstream_coordinate)

        node_names = np.array([n]*len(skeleton_points))

        if verbose:
            print(f"Node {n} has {len(skeleton_points)} skeleton points")

        total_skeleton_points.append(skeleton_points)
        total_node_names.append(node_names)

    if len(total_skeleton_points) > 0:
        total_skeleton_points = np.vstack(total_skeleton_points)
        total_node_names = np.concatenate(total_node_names)
    else:
        total_skeleton_points = np.array([])
        total_node_names = np.array([])

    if verbose:
        print(f"Total skeleton points = {len(total_skeleton_points)}")

    if return_node_names:
        return total_skeleton_points,total_node_names
    else:
        return total_skeleton_points
    
    

def split_location_node_map_df(
    G,
    split_locations,
    G_target = None,
    nodelist = None,
    distance_max = 5000,
    error_if_no_one_match = True,
    error_on_non_unique_node_names = True,
    verbose = False,
    verbose_loop = False,
    ):
    """
    Purpose: Want to map split locaitons to the nodes that
    were cut off due to proofreading with what rule was used to cut off

    Things can leverage:
        The limb name
        Nodes that shouldn't be mapped

    Pseudocode: 
    Iterate through all filters and limbs to work
    with the split locations and possible nodes to match with



    """

    split_info = []

    if G_target is not None and nodelist is None:
        nodelist = nxu.nodes_without_match(
            G_source = G,
            G_target=G_target,
            verbose = verbose )

    if nodelist is not None: 
        G_search = G.subgraph(nodelist).copy()
    else:
        G_search = G

    split_counter = 0
    for filter_name,filter_splits in split_locations.items():
        if type(filter_splits) != dict:
            filter_splits = {"L-1":filter_splits}

        for limb_name,split_loc in filter_splits.items():
            G_limb = nxu.limb_graph(G_search,limb_name)

#             sk_pts,sk_names = nxu.skeleton_coordinates_from_G(
#                 G_limb,
#                 return_node_names=True,
#                 verbose = False)

            if len(G_limb) == 0:
                raise Exception("Empty graph")

            for i,s in enumerate(split_loc):
                
                local_split = dict(limb_split_idx = i,
                                   limb_name = limb_name,
                                   coord = s,
                                  filter_name = filter_name)
                
                """
                4/13: Want to determine the node that has the lowest average distance
                
                Pseudocode:
                1) Iterate through all of the nodes
                2) Build a KDTree on their skeleton points
                3) Find the average distance
                4) Add to name and distance to list to find min later
                
                """
                sk_names = []
                dist_from_split = []
                for n in nxu.limb_branch_nodes(G_limb):
                    '''    
                    dist_from_split = np.linalg.norm(sk_pts - s,axis = 1)
                    '''
                    n_kd = KDTree(G_limb.nodes[n]["skeleton_data"])
                    dist,_ = n_kd.query(s.reshape(-1,3))
                    sk_names.append(n)
                    dist_from_split.append(np.mean(dist))
                    
                sk_names = np.array(sk_names)
                dist_from_split = np.array(dist_from_split)

                min_dist = np.min(dist_from_split)
                min_arg = np.where(dist_from_split == min_dist)[0]
                min_nodes = sk_names[min_arg]

                if len(min_nodes) != 1 and error_if_no_one_match:
                    raise Exception(f"Split {i} ({s}) had min_dist ({min_dist}) had more than one match ({min_nodes})")

                if error_if_no_one_match:
                    min_nodes = min_nodes[0]

                if min_dist > distance_max:
                    raise Exception(f"Split {i} ({s}) had min_dist ({min_dist}) that was greater than distance_max ({distance_max})")

                local_split["node"] = min_nodes
                local_split["min_distance"] =  min_dist

                split_info.append(local_split)
                split_counter += 1

                if verbose_loop:
                    print(f"local_split = {local_split}")

    split_df = pd.DataFrame.from_records(split_info)
    
    
    if error_on_non_unique_node_names and len(split_df) > 0:
        if len(split_df["node"].unique()) != len(split_df):
            raise Exception("Not unique nodes")
    
    if verbose:
        print(f"Total split mappings: {len(split_df)}")
    
    return split_df

def nodes_with_auto_proof_filter(
    G,
    return_filter_names=False,
    verbose = False,
    ):
    return xu.nodes_with_non_none_attributes(
        G,
        attribute_name=nxu.auto_proof_filter_name_default,
        return_attribute_value=return_filter_names,
        verbose=verbose)

def set_auto_proof_filter_attribute(
    G,
    split_df = None,
    split_locations = None,
    inplace = True,
    filter_attribute_name = None,
    default_value = None,
    verbose = False,
    error_on_non_unique_node_names=True,
    filter_axon_on_dendrite_splits = True,
    **kwargs
    ):
    """
    Purpose: To take the split locations df and
    to label the nodes with the right error
    """
    
    if split_df is None:
        split_df = nxu.split_location_node_map_df(
            G = G,
            split_locations=split_locations,
            verbose = verbose,
            error_on_non_unique_node_names=error_on_non_unique_node_names,
            **kwargs
            )

    if filter_attribute_name is None:
        filter_attribute_name = nxu.auto_proof_filter_name_default

    if not inplace:
        G = copy.deepcopy(G)

    xu.set_node_attribute(
        G,
        attribute_name = filter_attribute_name,
        attribute_value = default_value
    )

    if len(split_df) > 0:
        filter_names = split_df["filter_name"]
        nodes_to_label = split_df["node"]

        for n,f in zip(nodes_to_label,filter_names):
            if error_on_non_unique_node_names:
                G.nodes[n][filter_attribute_name] = f
            else:
                if G.nodes[n][filter_attribute_name] is None:
                    G.nodes[n][filter_attribute_name] = []
                G.nodes[n][filter_attribute_name].append(f)

    if verbose:
        print(f"nodes with filter attributes")
        nxu.nodes_with_auto_proof_filter(
            G,
            verbose = True)
        
    if filter_axon_on_dendrite_splits:
        if verbose:
            print(f"filtering the axon on dendrite splits")
        G = nxu.filter_axon_on_dendrite_splits_to_most_upstream(G)

    return G

def filter_axon_on_dendrite_splits_to_most_upstream(G):
    return G

def segment_id_from_G(G,return_split_index = True):
    seg_id = G.graph["segment_id"]
    if return_split_index:
         return seg_id,G.graph["split_index"]
    else:
        return seg_id


# ---------- For filtering parts of the graph ---------
def soma_connected_nodes(
    G,
    ):
    try:
        curr_node = nxu.soma_node_name_global
        return list(G[curr_node].keys())
    except:
        curr_node = nxu.most_upstream_node(G)
        return list(G[curr_node].keys())


def most_upstream_nodes(
    G,
    nodes=None,
    verbose = False,
    return_downstream_count = True,
    ):
    """
    Purpose: To get a count of the number of
    downstream nodes
    
    Ex: 
    nxu.most_upstream_nodes(
        G,
        nodes = ["L1_10","L1_2","L1_8","L0_20"],
        verbose = True
    )
    """
    if nodes is None:
        nodes = list(G.nodes())
        
    nodes = np.array(nodes)

    down_count = np.array([xu.n_all_downstream_nodes(G,k) for k in nodes])
    down_count_idx_sorted = np.flip(np.argsort(down_count))
    nodes_sorted = nodes[down_count_idx_sorted]
    down_count_sorted = down_count[down_count_idx_sorted]

    if verbose:
        print(f"nodes_sorted = {nodes_sorted}")
        print(f"down_count_sorted = {down_count_sorted}")
    
    if return_downstream_count:
        return nodes_sorted,down_count_sorted
    else:
        return nodes_sorted

def most_upstream_node(
    G,
    nodes = None,
    verbose = False,
    ):
    
    if len(G.nodes()) == 0:
        return None
    
    if len(G.nodes()) == 1:
        return list(G.nodes())[0]
    
    return nxu.most_upstream_nodes(
        G,
        nodes=nodes,
        verbose = verbose,
        return_downstream_count = False,
        )[0]

def most_upstream_coordinate(G):
    """
    Purpose: Get upstream skeleton point of the 
    most upstream node
    """
    node = nxu.most_upstream_node(G)
    return G.nodes[node]["endpoint_upstream"]
    
def limb_graphs_from_soma_connected_nodes(
    G,
    verbose = False,
    plot = False,
    ):
    """
    Purpose: To get all of the limb graphs
    defined by those boardering the soma

    Pseucode: 
    1) Get all of the soma connected nodes
    2) For each soma connected node: get the connected subgraph
    
    Ex: 
    limb_graphs = nxu.limb_graphs_from_soma_connected_nodes(
    G,
    verbose = True,
    plot = False,
    )
    """
    soma_conn_nodes = nxu.soma_connected_nodes(G)
    if verbose:
        print(f"soma_conn_nodes = {soma_conn_nodes}")

    total_limb_graphs = []
    for node in soma_conn_nodes:
        if verbose:
            print(f"-- Working on soma node {node}")
        limb_graph = nxu.limb_graph(G,most_upstream_node=node)

        if plot:
            most_up_node = xu.most_upstream_node(limb_graph)
            print(f"Plotting limb with most upstream node {most_up_node}")
            nxu.draw_tree(limb_graph)
            print(f"\n\n")

        total_limb_graphs.append(limb_graph)

    return total_limb_graphs


def distance_from_node_to_soma(
    G,
    node,
    include_self_distance = False,
    distance_attribute = "skeletal_length",
    destination_node = "S0",
    verbose = False,
    return_path= False,
    ):
    """
    Purpose: To find the distance of a node from soma
    (both with inclusion and without of its own skeletal length)

    Pseudocode: 
    1) 
    """


    total_lengths = []
    node_path = []
    
    if verbose:
        print(f"Finding distance from {node} to {destination_node}")

    if include_self_distance:
        total_lengths.append(G.nodes[node][distance_attribute])
        node_path.append(node)
        
    # here is where can check if the attribute already exists

    upstream_node = xu.upstream_node(G,node)

    while upstream_node != destination_node:
        if verbose:
            print(f"Working on upstream node {upstream_node}")

        node_path.append(upstream_node)
        total_lengths.append(G.nodes[upstream_node][distance_attribute])
        upstream_node = xu.upstream_node(G,upstream_node)

    path_length = np.sum(total_lengths)
    if verbose:
        print(f"path_length = {path_length}")
        print(f"Path to soma: {node_path}")
        print(f"Path {distance_attribute}: {total_lengths}")

    if return_path:
        return path_length,node_path
    else:
        return path_length
    
def distance_upstream_from_soma(
    G,
    node,
    verbose = False,
    from_attributes=False,
    **kwargs
    ):
    """
    nxu.distance_upstream_from_soma(
    G,
    "L0_19",
    verbose = True,   
    )
    """
    if from_attributes:
        return G.nodes[node]["soma_distance_skeletal"]
    
    return distance_from_node_to_soma(
    G,
    node,
    include_self_distance = False,
    verbose = verbose,
    return_path= False,
    **kwargs
    )

def distance_downstream_from_soma(
    G,
    node,
    verbose = False,
    from_attributes=False,
    **kwargs
    ):
    """
    Ex: 
    nxu.distance_downstream_from_soma(
        G,
        "L0_19",
        verbose = True,   
    )
    """
    if from_attributes:
        return G.nodes[node]["soma_distance_skeletal"] + G.nodes[node]["skeletal_length"]
    
    return distance_from_node_to_soma(
    G,
    node,
    include_self_distance = True,
    verbose = verbose,
    return_path= False,
    **kwargs
    )
    
def distance_from_soma_df(
    G,
    nodes = None,
    distance_type = "upstream",
    from_attributes=False):
    """
    Purpose: Find all the soma distances of 
    all the nodes
    """
    if nodes is None:
        nodes = nxu.limb_branch_nodes(G)

    dist_dict = [{"node":n,"soma_distance":getattr(nxu,f"distance_{distance_type}_from_soma")(G,n,from_attributes=from_attributes)}
                for n in nodes]

    dist_dict = pd.DataFrame.from_records(dist_dict)

    return dist_dict

def distance_upstream_from_soma_df(
    G,
    nodes = None,
    from_attributes=False,
    ):
    
    return nxu.distance_from_soma_df(
    G,
    nodes = nodes,
    distance_type = "upstream",
    from_attributes=from_attributes,)

def distance_downstream_from_soma_df(
    G,
    nodes = None,
    from_attributes=False):
    
    return nxu.distance_from_soma_df(
    G,
    nodes = nodes,
    distance_type = "downstream",
    from_attributes=from_attributes,)

def nodes_distance_query_from_soma(
    G,
    distance_threshold,
    within_distance = True,
    distance_type = "upstream",
    nodes=None,
    return_subgraph=False,
    return_soma_with_sugraph = True,
    verbose = False,
    maintain_skeleton_connectivity = True,
    from_attributes = False,
    **kwargs
    ):
    """
    Purpose: Find all the nodes within
    a certain distance or farther away than
    a certain distance from soma
    """
    
    # calculates the distance if not already available
    if from_attributes:
        G = nxf.add_any_missing_node_features(G)
    dist_df = getattr(nxu,f"distance_{distance_type}_from_soma_df")(G,nodes=nodes,
                                                                        from_attributes=from_attributes)
    
    
    
    
    if within_distance:
        query = f"soma_distance <= {distance_threshold}"
        query_descriptor = "closer"
    else:
        query = f"soma_distance > {distance_threshold}"
        query_descriptor = "farther"

    filt_df = dist_df.query(query)
    
    filt_nodes = filt_df["node"].to_list()
    if verbose:
        print(f"Nodes {distance_type} dist {query_descriptor} than {distance_threshold} from soma ({len(filt_nodes)}):\n{np.array(filt_nodes)} ")
        
        
    if return_subgraph:
        
        if return_soma_with_sugraph:
            filt_nodes += [nxu.soma_node_name_global]
            
        # Old way:
        #G_sub = G.subgraph(filt_nodes).copy()
        
        node_to_delete = np.setdiff1d(list(G.nodes()),filt_nodes)
        if len(node_to_delete) > 0:
            G_sub = nxu.remove_node(
            G,
            node_to_delete,
            verbose= False,
            inplace = False,
            maintain_skeleton_connectivity=maintain_skeleton_connectivity,
            **kwargs)
        else:
            G_sub = G
       
        
        return G_sub
    else:
        return filt_nodes

def nodes_within_distance_from_soma(
    G,
    distance_threshold,
    distance_type = "upstream",
    nodes=None,
    return_subgraph=False,
    verbose = False,
    from_attributes=False,
    **kwargs
    ):
    
    """
    Ex: 
    distance_threshold = 1000
    curr_nodes = nxu.nodes_within_distance_from_soma(G,distance_threshold = distance_threshold,verbose = True)
    """
    
    return_G =  nxu.nodes_distance_query_from_soma(
    G,
    distance_threshold,
    within_distance = True,
    distance_type = distance_type,
    nodes=nodes,
    return_subgraph=return_subgraph,
    verbose = verbose,
    from_attributes=from_attributes,
    **kwargs
    )
    
    
    return return_G
    
def nodes_farther_than_distance_from_soma(
    G,
    distance_threshold,
    distance_type = "upstream",
    nodes=None,
    return_subgraph=False,
    verbose = False,
    from_attributes = False,
    **kwargs
    ):
    
    return nxu.nodes_distance_query_from_soma(
    G,
    distance_threshold,
    within_distance = False,
    distance_type = distance_type,
    nodes=nodes,
    return_subgraph=return_subgraph,
    verbose = verbose,
    from_attributes=from_attributes,
    **kwargs
    )

def nodes_within_distance_upstream_from_soma(
    G,
    distance_threshold,
    nodes=None,
    return_subgraph=False,
    verbose = False,
    ):
    
    return nxu.nodes_within_distance_from_soma(
    G,
    distance_threshold,
    distance_type = "upstream",
    nodes=nodes,
    return_subgraph=return_subgraph,
    verbose = verbose,
    )


def soma_filter_by_complete_graph(
    G,
    inplace = False,
    verbose = False,
    connect_previous_touching_soma_nodes = False,
    plot = False):
    """
    Problem: Want to resolve the soma so it does
    not affect...

    Solution 1: Delete the soma network and then 
    connect all the 

    Pseudocode:
    1) Connect all the soma connecting nodes
    2) Get the subgraph of only the limb branches
    
    Ex: 
    G_no_soma = nxu.soma_filter_by_complete_graph(G_filt,plot=True)
    nxu.draw_tree(G_no_soma)
    """
    if not inplace:
        G = copy.deepcopy(G)
    
    if connect_previous_touching_soma_nodes:
        soma_conn_nodes = nxu.soma_connected_nodes(G)
        if verbose:
            print(f"soma_conn_nodes ({len(soma_conn_nodes)})= {soma_conn_nodes}")

        all_conn = np.concatenate([[(k,v) for v in soma_conn_nodes if v != k] for k in soma_conn_nodes])

        if verbose:
            print(f"New connections ({len(all_conn)}) = {all_conn}")

        G.add_edges_from(all_conn)
        
    return_G = nxu.limb_branch_subgraph(G)
    
    if plot:
        nx.draw(nx.Graph(return_G),with_labels = True)
        
    return return_G


def nodes_with_auto_proof_filter_type(
    G,
    filter_type,
    filter_feature = "auto_proof_filter",
    verbose = False,
    ):
    """
    Purpose: To get all nodes with a certain filter marking 
    
    Ex: 
    nxu.nodes_with_auto_proof_filter_type(G,filter_type="axon_on_dendrite",verbose = True)
    """
    filter_df = xu.node_df(G).query(f"{filter_feature} == {filter_feature}")
    nodes_with_filter_type = filter_df[filter_df[filter_feature].str.contains(filter_type)][xu.upstream_name].to_numpy()
    if verbose:
        print(f"nodes with {filter_type} = {nodes_with_filter_type}")
        
    return nodes_with_filter_type

def clear_nodes_auto_proof_filter_feature(
    G,
    nodes,
    filter_feature = "auto_proof_filter",
    verbose=False,
    ):
    for n in nodes:
        curr_filter_name = G.nodes[n][filter_feature]
        if verbose:
            print(f"Clearning {curr_filter_name} for node {n} ")
        G.nodes[n][filter_feature] = None
    return G

def filter_axon_on_dendrite_splits_to_most_upstream(
    G,
    verbose=False,
    inplace = False,
    filter_out_only_if_parent_in_split = False,
    **kwargs):
    """
    Purpose: To reduce the axon on dendrite merges to only those
    that are the most upstream of the group
    
    Pseudocode:
    1) Get all of the nodes that are in axon on dendrite mergers
    2) For each node:
    a. Get either just the parent or all of the upstream node
    b. add node to list ot be cleared if has upstream axon on dendrite
    
    Ex: 
    G_filt = filter_axon_on_dendrite_splits_to_most_upstream(G,verbose = True)
    nxu.nodes_with_auto_proof_filter(G_filt,"axon_on_dendrite")
    
    """
    axon_on_dendrite_nodes = nxu.nodes_with_auto_proof_filter_type(G,"axon_on_dendrite")
    
    if not inplace:
        G = copy.deepcopy(G)
    
    if verbose:
        print(f"axon_on_dendrite_nodes ({len(axon_on_dendrite_nodes)}) = {axon_on_dendrite_nodes}")
        
    nodes_to_clear = []
    for n in axon_on_dendrite_nodes:
        if filter_out_only_if_parent_in_split:
            up_nodes = [xu.upstream_node(G,n)]
        else:
            up_nodes = xu.all_upstream_nodes(G,n)
            
        ax_on_dendr_upstream = np.intersect1d(axon_on_dendrite_nodes,up_nodes)
        if len(ax_on_dendr_upstream) > 0:
            if verbose:
                print(f"Removing node {n} because had upstream axon on dendrite: {ax_on_dendr_upstream}")
            nodes_to_clear.append(n)
            
    G = nxu.clear_nodes_auto_proof_filter_feature(G,nodes_to_clear,verbose = verbose)
        
    
    if verbose:
        axon_on_dendrite_nodes = nxu.nodes_with_auto_proof_filter_type(G,"axon_on_dendrite")
        print(f"axon_on_dendrite_nodes AFTER FILTERING ({len(axon_on_dendrite_nodes)}) = {axon_on_dendrite_nodes}")
        
    return G
    
    
def fix_flipped_skeletons(
    G,
    verbose = False,):
    """
    Purpose: To fix the skeleton data
    in Graph objects if they are
    not aligned correctly
    
    Ex: 
    segment_id,split_index = 864691135162621741,0

    G_obj = hdju.graph_obj_from_auto_proof_stage(
        segment_id=segment_id,
        split_index=split_index,
    )
    
    G_obj = fix_flipped_skeletons(G_obj,verbose = True)
    """
    
    node_flipped = []
    for n in G.nodes():
        if "s" in n.lower():
            continue
        curr_dict = G.nodes[n]
        endpt_up = curr_dict["endpoint_upstream"]
        sk_up = curr_dict["skeleton_data"][0]
        if not np.array_equal(endpt_up,sk_up):
            G.nodes[n]["skeleton_data"] = np.flip(curr_dict["skeleton_data"],axis=0)
            node_flipped.append(n)
            
    if verbose:
        print(f"Nodes with skeleton flipped: {node_flipped}")
        
    return G

def soma_only_graph(
    G,
    soma_node_name=soma_node_name_global):
    """
    Purpose: To check if only a soma node is in the nodes
    """
    node_names = [n for n in G.nodes()]
    
    if len(node_names) == 1:
        if node_names[0] == soma_node_name:
            return True
    else:
        return False
    
# --------- For outputing graph attributes --------
def compartment_from_node(
    G,
    n=None,
    replace_underscore = True):
    """
    Purpose: To get the compartment from
    a node in a graph
    """
    if "graph" in str(type(G)).lower():
        G = G.nodes[n]
     
    comp = G["compartment"]
    if replace_underscore:
        comp = comp.replace("_","")
    return comp

def width_from_node(
    G,
    n,
    verbose = False):
    """
    Purpose: To get the width of a certain node
    
    Ex: 
    from neuron_morphology_tools from neurd import neuron_nx_utils as nxu
    nxu.width_from_node(
     G = G_obj,
     n = "L0_5",
        verbose = True

    )
    """
    curr_key = G.nodes[n]
    width = None
    if len(curr_key["width_new"]) > 0:
        width = curr_key["width_new"].get("no_spine_median_mesh_center",None)
    if width is None:
        if verbose:
            print(f"Getting width from default width (not new width)")
        width = key.get("width",None)

    return width


def attribute_graph_from_graph_obj(
    G,
    attribute,
    ids_name = None,
    verbose = False,
    return_attribute_nodes = False,
    return_attribute_nodes_to_branch_dict = True,
    return_upstream_dist = False,
    exclude_presyn = False,
    ):
    """
    Purpose: To convert a graph object into 
    a graph where an attribute of the graph 
    object are the nodes along with the branch points.
    The edges between nodes will be the upstream distance.

    Application: Will help find the distances between the attributes
    """

    attribute_name = f"{attribute}_data"

    if ids_name is None:
        if attribute == "synapse":
            ids_name = f"syn_id"
        else:
            ids_name = f"{attribute}_id"

    graph_edges = []
    graph_weights = []
    ids_counter = 0
    attribute_ids = []
    attribute_ids_branch_dict = dict()
    upstream_dist = {v:None for v in G.nodes()}
    nodes = list(G.nodes())

    for n in nodes:
        key = G.nodes[n]
        curr_dict = dict()

        upstream_node = xu.upstream_node(G,n)
        if upstream_node is None:
            upstream_node = f"L-1"

        comp = compartment_from_node(key)
        attribute_data = key[attribute_name]
        
        if attribute == "synapse" and exclude_presyn:
            #print(f"Excluding presyns")
            attribute_data = [k for k in attribute_data if k["syn_type"] == "postsyn"]

        #if len(attribute_data) > 0:
        upstream_data = np.array([k["upstream_dist"] for k in attribute_data])

        # generates the ids for the current branch
        ids_data = np.array([k[ids_name] for k in attribute_data])
        if None in ids_data:
            ids_data = np.arange(ids_counter,ids_counter + len(attribute_data)).astype('int')
            ids_counter += len(attribute_data)
            
        
        attribute_ids.append(ids_data)
        attribute_ids_branch_dict.update({str(k):n for k in ids_data})
        upstream_dist.update({str(k):upstream_data[j] for j,k in enumerate(ids_data)})

        order_idx = np.argsort(upstream_data)
        upstream_data_sort = upstream_data[order_idx]
        ids_data_sort = ids_data[order_idx]

        upstream_data_sort = np.hstack([[0],upstream_data_sort,[key["skeletal_length"]]])
        ids_data_sort = np.hstack([[upstream_node],ids_data_sort,[n]])

        edge_weights = upstream_data_sort[1:] - upstream_data_sort[:-1]
        edge_weights[edge_weights <= 0] = 0
        edges = np.vstack([ids_data_sort[:-1],ids_data_sort[1:]]).T
        #weighted_edges = np.vstack([edges,edge_weights]).T

        if verbose:
            print(f" --> node {n}: # of edges = {len(edges)}")
            #print(f"weighted_edges = {weighted_edges}")

        graph_edges.append(edges)
        graph_weights.append(edge_weights)


    output_G = nx.Graph()

    if len(graph_edges) > 0:
        graph_edges = np.vstack(graph_edges)
        graph_weights = np.hstack(graph_weights)
        output_G = xu.edges_and_weights_to_graph(graph_edges,
                              weights_list=graph_weights)
        
    if len(attribute_ids) > 0:
        attribute_ids = np.hstack(attribute_ids)
        
    attribute_ids = np.array(attribute_ids).astype('int').astype("str")
        
    if verbose:
        print(f"Total Graph stats:")
        xu.print_node_edges_counts(output_G)

    if ((not return_attribute_nodes) and (not return_attribute_nodes_to_branch_dict)
        and (not return_upstream_dist)):
        return output_G
    return_value =[output_G]
    
    if return_attribute_nodes:
        return_value.append(attribute_ids)
    if return_attribute_nodes_to_branch_dict:
        return_value.append(attribute_ids_branch_dict)
        
    if return_upstream_dist:
        return_value.append(upstream_dist)
    
    return return_value
    
    
def plot_inter_attribute_intervals(
    inter_attribute_info,
    title = None,
    bins=50,
    attribute = "attribute",
    verbose = False):
    """
    Purpose: To plot the histograms of the
    closest attribute arrays
    """
    if type(inter_attribute_info) != dict:
        inter_attribute_info = {1:inter_attribute_info}
        
    if type(inter_attribute_info[1]) == dict:
        if verbose:
            print(f"Combining the branch specific dicts")
        inter_attribute_info = {k:np.concatenate(list(v.values())) for k,v in inter_attribute_info.items()}
        
    fig,ax = plt.subplots(1,1,)
    for i in range(len(inter_attribute_info)):
        ax.hist(
            np.array(inter_attribute_info[i+1])/1000,
            label=f"{i+1} Hop",
            alpha = 0.5,
            bins = bins)

        if title is None:
            title = f"Closest {attribute}"
        ax.set_title(title)
        ax.set_xlabel(f"Distance (um)")
        ax.set_ylabel(f"Count")
        ax.legend()
        
    return ax

def inter_attribute_intervals_from_G(
    G,
    attribute,
    n_closest_neighbors = 1,
    default_value = -1,
    debug_time = False,
    plot = False,
    verbose = False,
    separate_branches = True,
    return_upstream_dist = True,
    exclude_presyn = True,
    ):
    """
    Purpose: To find the k inter-attribute
    distances for the attributes on the graph

    1) Turn the graph into an attribute graph
    2) For every attribute id:
        Remove the attribute id from the total list
        a. For 1 to k (the number of attributes away):
           if list empty then add distance of -1 and continue
           Calculate the closest neighbors from remaining nodes and get distance

           Save the distance
           Remove that closest neighbor from the list

    3) Return the lists of closest distances
    """

    (G_disc,
     att_nodes,
     att_to_branch_dict,
     att_to_upstream_dist)= nxu.attribute_graph_from_graph_obj(
        G,
        attribute = attribute,
        verbose = verbose,
        return_attribute_nodes = True,
        return_attribute_nodes_to_branch_dict = True,
        return_upstream_dist = True,
        exclude_presyn = exclude_presyn,
    )

    upstream_dists = {v:[] for v in G.nodes()}
    if not separate_branches:
        closest_neighbors = {k:[] for k in range(1,n_closest_neighbors + 1)}
        
    else:
        closest_neighbors = {k:{v:[] for v in G.nodes()} for k in range(1,n_closest_neighbors + 1)}
        

    for n in tqdm(att_nodes):
        local_nodes = att_nodes.copy()
        local_nodes = local_nodes[local_nodes != n]
        previous_closest = None
        
        upstream_dists[att_to_branch_dict[n]].append(att_to_upstream_dist[n])
        
        for i in closest_neighbors:
            if previous_closest is not None:
                local_nodes = local_nodes[local_nodes != previous_closest]
            if len(local_nodes) == 0:
                default_value
                short_path_dist = default_value
                n2 = None

            else:
                if debug_time:
                    st = time.time()
                short_path_dist,n1,n2 = xu.shortest_path_between_two_sets_of_nodes(
                    G_disc,
                    node_list_1 = [n],
                    node_list_2 = local_nodes,
                    return_node_pairs=True,
                    return_path_distance=True
                )

                if debug_time:
                    print(f"Time for calculating shortest path = {time.time() - st}")
                    st = time.time()


            if verbose:
                print(f"{i}th Closest neighbor of {n} was {n2} with path distance {short_path_dist}")
            previous_closest = n2
            
            if not separate_branches:
                closest_neighbors[i].append(short_path_dist)
            else:
                closest_neighbors[i][att_to_branch_dict[n]].append(short_path_dist)

    if not separate_branches:
        closest_neighbors = {k:np.array(v) for k,v in closest_neighbors.items()}
    else:
        for k in closest_neighbors:
            for v in closest_neighbors[k]:
                closest_neighbors[k][v] = np.array(closest_neighbors[k][v])
        

    if plot: 
        nxu.plot_inter_attribute_intervals(
            closest_neighbors,
            attribute=attribute)
        plt.show()
        
    if return_upstream_dist:
        return closest_neighbors,upstream_dists
    else:
        return closest_neighbors


def n_data_attribues(G,attribute,n=None,exclude_presyn = True):
    """
    Purpose: To get the number of data attributes
    belonging to a branch
    """
    if n is None:
        n = [k for k in list(G.nodes()) if "S" not in k]
    else:
        n = [n]
        
    curr_data = [G.nodes[n1][f"{attribute}_data"] for n1 in n]
    if exclude_presyn and attribute == "synapse":
        curr_data = [[k for k in v
                     if k["syn_type"] != "presyn"] for v in curr_data]
        
    return np.sum([len(k) for k in curr_data])




def inter_attribute_G_preprocessing(
    G,
    dendrite_only = True,
    remove_starter_branches = True,
    ):
    
    if dendrite_only:
        G = nxu.dendrite_subgraph(G)

    if remove_starter_branches:
        G = nxu.remove_small_starter_branches(
                G,
                verbose = False,
                maintain_skeleton_connectivity = True)
    return G
    
    

def inter_attribute_intervals_dict_from_neuron_G(
    G,
    attribute=None,
    dendrite_only = True,
    remove_starter_branches = True,
    #return_attribute_density = True,
    #return_compartment = True
    verbose = True,
    n_closest_neighbors = 3,
    branch_features_to_add = (
        "mesh_volume",
        'n_synapses_head',
        'n_synapses_neck',
        'n_synapses_no_head',
        'n_synapses_post',
        'n_synapses_pre',
        'n_synapses_shaft',
        'n_synapses_spine',
        "total_spine_volume",
        "soma_distance_euclidean",
        'parent_skeletal_angle',
        'siblings_skeletal_angle_max',
        'width_upstream',
         'width_downstream',
        ),
    
    
    # -------- arguments for random shuffling -------
    shuffle_upstream_dist = False,
    attribute_sk_nullification = None,
    exclude_presyn = True,
    discretization_length = 100,
    seed = None,
    
    ):
    """
    Purpose: Want to build an inter attribute 
    distance for all branches in a neuron
    """
    if attribute is None:
        attribute = ("spine","synapse")
        
    if type(attribute) == tuple:
        attribute = list(attribute)
    attribute= nu.convert_to_array_like(attribute)
    data_dicts = []


    if not nxu.soma_only_graph(G):
        
        G = nxu.inter_attribute_G_preprocessing(
            G,
            dendrite_only = dendrite_only,
            remove_starter_branches = remove_starter_branches,
        )
        
        limb_graphs,limb_idxs = nxu.all_limb_graphs(G,return_idxs=True)

        if verbose:
            print(f"# of limb graphs: {len(limb_graphs)}")

        for idx,G_limb in zip(limb_idxs,limb_graphs):
            
            if shuffle_upstream_dist: 
                if verbose:
                    print(f"Applying shuffle_upstream_dist")
                G_limb = nxu.shuffle_upstream_dist_on_data_attribute(
                    G_limb,
                    attribute = attribute,
                    attribute_sk_nullification = attribute_sk_nullification,
                    exclude_presyn = exclude_presyn,
                    discretization_length = discretization_length,
                    seed = seed,
                )

            if verbose:
                print(f"\n--------Working on Limb {idx}---------")

            limb_dicts = dict()
            for n in G_limb.nodes():
                curr_dt = dict(branch=n,
                                   compartment=nxu.compartment_from_node(G_limb,n),
                                     skeletal_distance_to_soma = nxu.distance_upstream_from_soma(G,node=n),
                                   skeletal_length = G_limb.nodes[n]["skeletal_length"],
                                    width = nxu.width_from_node(G_limb,n))
                if branch_features_to_add is not None:
                    for k in branch_features_to_add:
                        curr_dt[k] = G_limb.nodes[n][k]
                        
                limb_dicts[n] = curr_dt
                              
            
            
            for att in attribute:
                for n in limb_dicts:
                    limb_dicts[n][f"n_{att}"] = nxu.n_data_attribues(G_limb,attribute=att,n=n)
                (inter_dict_att,
                 upstream_dist)= nxu.inter_attribute_intervals_from_G(
                    G_limb,
                    attribute=att,
                    n_closest_neighbors=n_closest_neighbors,
                    separate_branches = True,
                    plot = False,
                     return_upstream_dist = True
                )

                # Add the attribute data to the dicts
                for i,data in inter_dict_att.items():
                    for node_name,node_array in data.items():
                        limb_dicts[node_name][f"{att}_intervals_{i}"] = node_array
                        
                for n in limb_dicts.keys():
                    limb_dicts[n][f"{att}_upstream_dist"] = np.array(upstream_dist[n])

            data_dicts += list(limb_dicts.values())


    return data_dicts



# =============== For the random shuffling (6320)==================
def shuffle_upstream_dist_on_data_attribute(
    G,
    attribute = None,
    attribute_sk_nullification = None,
    exclude_presyn = True,
    discretization_length = 100,
    seed = None,
    plot_G = False,
    verbose = False,
    ):
    """
    Purpose: to randomly shuffle 
    the upstream dists of data attributes
    in a neuron object (for null testing)
    
    Ex: 
    from neuron_morphology_tools from neurd import neuron_nx_utils as nxu

    segment_id = 864691134885060346
    split_index = 0

    plot = False
    plot_proofread_neuron = False


    G_obj = hdju.graph_obj_from_auto_proof_stage(
            segment_id=segment_id,
            split_index=split_index,
        )

    if plot:
        nxu.plot(G_obj)

    if plot_proofread_neuron:
        hdju.plot_proofread_neuron(
            segment_id,split_index,
        )

    G = nxu.inter_attribute_G_preprocessing(
        G,
        dendrite_only = dendrite_only,
        remove_starter_branches = dendrite_only,
    )

    limb_graphs,limb_idxs = nxu.all_limb_graphs(G,return_idxs=True)

    nxu.shuffle_upstream_dist_on_data_attribute(
        limb_graphs[0],
        verbose = True
    )
    """
    if attribute_sk_nullification is None:
        attribute_sk_nullification = dict(spines=1000)
        
    if attribute is None:
        attribute = ["spine",'synapse']
        
    attribute = nu.convert_to_array_like(attribute)


    if seed is not None:
        np.random.seed(seed)

    G = G.copy()

    if plot_G:
        nxu.plot(G)
        
    for n in G.nodes():
        if verbose:
            print(f"-- Working on node {n}----")
        branch_dict = G.nodes[n]

        #1) Get the skeleton length
        skeletal_length = branch_dict["skeletal_length"]
        if verbose:
            print(f"skeletal_length = {skeletal_length}")


        #2) Create an array like a skeleton
        sk_array = nu.arange_with_leftover(skeletal_length,step = discretization_length)

#         if verbose:
#             print(f"sk_array = {sk_array}")
            


        for f in attribute:
            #a) gets the nullification distance
            f_null = attribute_sk_nullification.get(f,0)
            #b) Get the number of that feature to randomly sample
            n_name = f"n_{f}s"
            if f == "synapse" and exclude_presyn:
                n_name = f"n_{f}s_post"
            n_feat = branch_dict[n_name]

            if verbose:
                print(f"{n_name} = {n_feat}")


            sk_array_curr = sk_array[(sk_array >= f_null)
                                    & (sk_array <= (sk_array[-1] - f_null))]

        #     if verbose:
        #         print(f"sk_array_curr = {sk_array_curr}")

            # randomly sample from the distance array
            sample = nu.randomly_sample_array(sk_array_curr,n_samples = n_feat,replace = True)
            if verbose:
                print(f"samples = {sample}")

            # need to alter the upstream distances
            counter = 0
            for curr_dict in G.nodes[n][f"{f}_data"]:
                if f == "synapse" and exclude_presyn:
                    if curr_dict["syn_type"] == "presyn":
                        continue
                        print(f"Skipping presyn")
                curr_dict["upstream_dist"] = sample[counter]
                counter += 1


    return G


def plot_inter_attribute_intervals_from_dicts(
    dicts,
    attribute = "spine",
    bins = 100,
    title = None,
    um = True,
    attributes_to_plot = None,
    verbose = True
    ):
    
    """
    Purpose: To plot a certain attributes from a list of 
    datajoint dicts storing the distributions

    Pseudocode: 
    """

    attributes_to_plot = [k for k in dicts[0] if f"{attribute}_interval" in k]


    fig,ax = plt.subplots(1,1,)

    for att in attributes_to_plot:
        all_data = np.hstack([k[att] for k in dicts])
        if verbose:
            print(f"{att} mean = {np.mean(all_data)}")
        if um:
            all_data  = all_data/1000
        ax.hist(all_data,
            label=att,
            alpha = 0.5,
            bins = bins)

        if title is None:
            title = f"Closest {attribute}"

        ax.set_title(title)
        ax.set_xlabel(f"Distance (um)")
        ax.set_ylabel(f"Count")
        ax.legend()
        
def filter_graph(
    G,
    remove_starter_branches = True,
    distance_threshold = None,
    distance_threshold_min=None,
    features_to_output = None,
    filter_away_soma = True,
    output_graph_type = "Graph",
    verbose = False,
    ):
    """
    Purpose: To filter the graph object
    before the GNN processes
    
    Pseudocode: 
    1) Reduces to only dendrite subgraph
    2) Removes any small starter nodes
    3) Restricts to a certain distance
    4) Filter to certain features
    5) Filter into soma 
    6) Turn into non-directed graph
    """
    debug = False
    verbose_soma_conn = False
    
    if features_to_output is None:
        features_to_output = nxf.features_to_output_for_gnn
    
    if debug:
        print('L0_123' in G)
    
    #1) Reduces to only dendrite subgraph
    G = nxu.dendrite_subgraph(G)
    
    if debug:
        print('L0_123' in G)
            
    
    if verbose_soma_conn:
        print(f"Soma connected nodes = {nxu.soma_connected_nodes(G)}")
    
    #2) Removes any small starter nodes
    if remove_starter_branches:
        G_filt = nxu.remove_small_starter_branches(
            G,
            verbose = verbose,
            maintain_skeleton_connectivity = True)
    else:
        G_filt = G
        
    if debug:
        print('L0_123' in G_filt)
        
    if verbose_soma_conn:
        print(f"After remove starter branches: Soma connected nodes = {nxu.soma_connected_nodes(G_filt)}")
        
    #3) Restricts to a certain distance
    if distance_threshold is not None:
        G_dist_filt = nxu.nodes_within_distance_upstream_from_soma(
            G_filt,
            verbose = verbose,
            distance_threshold = distance_threshold,
            return_subgraph = True,
        )
    else:
        G_dist_filt = G_filt
        
    if verbose_soma_conn:
        print(f"After dist threshold: Soma connected nodes = {nxu.soma_connected_nodes(G_dist_filt)}")
        
    if distance_threshold_min is not None:
        G_dist_filt = nxu.nodes_farther_than_distance_from_soma(
            G_dist_filt,
            verbose = verbose,
            distance_threshold = distance_threshold_min,
            return_subgraph = True,
            distance_type = "downstream",
        )
        
    if verbose_soma_conn:
        print(f"After dist threshold min: Soma connected nodes = {nxu.soma_connected_nodes(G_dist_filt)}")
        

    #4) Filter to certain features
    if ((len(nxu.limb_branch_subgraph(G_dist_filt).nodes()) > 0)
        and (features_to_output is not None) and (len(features_to_output) > 0)):
        
        G_with_feats = nxf.filter_G_features(
                    G_dist_filt,
                    features=features_to_output,
                    inplace = False,
                    verbose = verbose,
                )
    else:
        G_with_feats = G_dist_filt
    
    if verbose_soma_conn:
        print(f"After filter_G_features: Soma connected nodes = {nxu.soma_connected_nodes(G_dist_filt)}")
    
    
    if filter_away_soma:
        #5) Filter into soma 
        G_no_soma = nxu.soma_filter_by_complete_graph(G_with_feats,plot=False)
    else:
        G_no_soma = G_with_feats
        
    #6) Turn into non-directed graph
    if output_graph_type is not None:
        G_no_soma = getattr(nx,output_graph_type)(G_no_soma)
    
    return G_no_soma


# --------- skeleton data ------------
def skeleton_soma_to_limb_start(G):
    if soma_node_name_global in G.nodes():
        nodes = np.vstack([G.nodes[soma_node_name_global]['endpoint_upstream']] + [G.nodes[k]["endpoint_upstream"] for k
                   in nxu.soma_connected_nodes(G)])
        edges = np.array([[0,k] for k in range(1,len(nodes))])
    else:
        nodes = np.array([G.nodes[soma_node_name_global]['endpoint_upstream']]).reshape(-1,3)
        edges = np.array([]).reshape(-1,2)
        
    return nodes,edges

    

def skeleton(
    G,
    include_soma = True,
    plot = False,
    verbose = False,
    mesh = None,
    return_verts_edges = True,
    ):
    
    skeleton_nodes = np.array([])
    skeleton_edges = np.array([])

    for node in nxu.limb_branch_nodes(G):
        
        curr_nodes = G.nodes[node]["skeleton_data"]
        edges = np.vstack([np.arange(len(curr_nodes))[:-1],
                          np.arange(len(curr_nodes))[1:],]).T
        edges += len(skeleton_nodes)
        if len(skeleton_nodes) > 0:
            skeleton_nodes = np.vstack([skeleton_nodes,curr_nodes])
            skeleton_edges = np.vstack([skeleton_edges,edges])
        else:
            skeleton_nodes = curr_nodes
            skeleton_edges = edges

    # add on the soma edges if have them
    if include_soma and soma_node_name_global in G.nodes():
        soma_nodes,soma_edges = nxu.skeleton_soma_to_limb_start(G)
        if len(soma_edges) > 0:
            soma_edges += len(skeleton_nodes)
        skeleton_nodes = np.vstack([skeleton_nodes,soma_nodes])
        skeleton_edges = np.vstack([skeleton_edges,soma_edges])

    if verbose:
        print(f"# of nodes = {len(skeleton_nodes)}")
        print(f"# of edges = {len(skeleton_edges)}")
        
    if plot:
        new_figure = True
        if mesh is not None:
            ipvu.plot_mesh(
                mesh,
                new_figure=True,
                show_at_end=False,
                flip_y=True,
                alpha = 0.1
            )
            new_figure = False
        ipvu.plot_obj(
            array = skeleton_nodes,
            lines=skeleton_edges,
            flip_y = True,
            new_figure = False,

        )
        
    
        
    if return_verts_edges:
        return skeleton_nodes,skeleton_edges
    else:
        return skeleton_nodes[skeleton_edges]
    
def plot_skeleton(G,mesh=None,**kwargs):
    nxu.skeleton(
        G,
        plot = True,
        mesh = mesh,
        **kwargs
    )

def skeleton_nodes(
    G,
    verbose = False,
    include_soma = False,
    **kwargs):
    
    return nxu.skeleton(
        G,
        verbose = verbose,
        include_soma=include_soma,
        **kwargs)[0]
    
    
def soma_center(G,use_most_upstream_as_backup = True):
    if soma_node_name_global in G.nodes():
        return G.nodes[soma_node_name_global]["endpoint_upstream"]
    else:
        try:
            return G.nodes[nxu.most_upstream_node(G)]["endpoint_upstream"]
        except:
            return None
        
        
def starting_coordinates_all_limbs(
    G,
    verbose=False):
    """
    Purpose: To get all of the limb starting coordinates

    Pseudocode: 
    2) Get all those bordering the soma
    3) Assemble the starting coordinates
    """
    soma_conn_nodes = nxu.soma_connected_nodes(G)
    if len(soma_conn_nodes) > 0:
        starting_coordinates = np.vstack([G.nodes[k]["endpoint_upstream"] 
                                for k in soma_conn_nodes])
    else:
        starting_coordinates = np.array([]).reshape(-1,3)
        
    if verbose:
        print(f"starting_coordinates = {starting_coordinates}")
        
    return starting_coordinates

def skeleton_width_data_from_node(
    G,
    n,
    skeleton_midpoints = False,
    width_to_repeat = "last",
    
    ):
    """
    Purpose: To get the skeleton data and 
    the width associated with each skeleton point
    """
    node_data = G.nodes[n]

    skeleton_points = node_data["skeleton_data"]
    width_array = np.array([k["width"] for k in node_data["width_data"]])
    if skeleton_midpoints:
        skeleton_points = (skeleton_points[1:]+skeleton_points[:-1])/2
    else:
        if width_to_repeat == "last":
            repeat_index = -1
        elif width_to_repeat == "first":
            repeat_index = 0

        width_array = np.hstack([width_array,[width_array[repeat_index]]])

    assert len(skeleton_points) == len(width_array)
    return skeleton_points,width_array

def skeleton_width_compartment_arrays_from_G(
    G,
    compartments = None,
    replace_underscore_in_compartments = False,
    plot = False,
    mesh = None,
    **kwargs
    ):
    """
    Purpose: To extract the skeleton,width,compartment
    arrays from a neuron object
    
    segment_id = 864691136422863407
    split_index = 0
    G = hdju.graph_obj_from_proof_stage(segment_id,split_index)
    mesh = hdju.fetch_proofread_mesh(segment_id)

    nxu.skeleton_width_compartment_arrays_from_G(
        G,
        plot = True,
        mesh = mesh)
    """

    skeleton_array = []
    width_array = []
    compartment_array = []

    for n in nxu.limb_branch_nodes(G):
        skel_data,width_data = nxu.skeleton_width_data_from_node(
            G,
            n = n,
            **kwargs)
        
        if np.any(width_data > 1000000):
            raise Exception(f"{n}")
            
        comp = nxu.compartment_from_node(G,n,replace_underscore=replace_underscore_in_compartments)
        if compartments is not None:
            if comp not in compartments:
                continue
        comp_data = np.repeat([comp],len(skel_data))

        skeleton_array.append(skel_data)
        width_array.append(width_data)
        compartment_array.append(comp_data)

    skeleton_array = np.vstack(skeleton_array)
    width_array = np.hstack(width_array)
    compartment_array = np.hstack(compartment_array)
    
    if plot:
        new_figure = True
        if mesh is not None:
            new_mesh = ipvu.plot_mesh(
                mesh,
                alpha=0.2,
                flip_y = True,
                show_at_end=False,
                new_figure = True,
            )
            new_figure = False
            
        scat_mesh = meshu.scatter_mesh_with_radius(skeleton_array,width_array)
        new_mesh = ipvu.plot_mesh(
            scat_mesh,
            alpha=1,
            color = "red",
            flip_y = True,
            new_figure = new_figure,
            show_at_end=True,
        )        
        
    return skeleton_array,width_array,compartment_array


def nodes_between_soma_and_nodes(
    G,
    nodes,
    verbose = False,
    ):
    """
    Purpose: to find the nodes in between a set of nodes
    and the soma node
    
    Ex: 
    nxu.nodes_between_soma_and_nodes(
        G_presyn,
        nodes=["L0_5","L0_7"],
        verbose = True
    )
    """
    if "str" in str(type(nodes)):
        nodes = getattr(nxu,f"{nodes}_nodes")(G)
    nodes = nu.convert_to_array_like(nodes)
    path,_,end_node = xu.shortest_path_between_two_sets_of_nodes(
        G,
        [nxu.soma_node_name_global],
        nodes,

    )

    in_between_nodes = path[1:-1]
    if verbose:
        print(f"in_between_nodes = {in_between_nodes}, closest node to soma = {end_node}")

    return in_between_nodes

def compartment_nodes(
    G,
    compartment,
    include_path_to_soma=False,
    verbose = False):
    """
    Ex: nxu.compartment_nodes(G_presyn,"apical_shaft")
    """
    compartment = nu.convert_to_array_like(compartment)
    nodes = []
    for k in nxu.limb_branch_nodes(G):
        comp_reg = nxu.compartment_from_node(G,k,replace_underscore=False)
        comp = nxu.compartment_from_node(G,k)
        if len(np.intersect1d(compartment,[comp_reg,comp])) > 0:
            nodes.append(k)
    if verbose:
        print(f"{compartment} nodes = {nodes}")
        
    if include_path_to_soma:
        new_nodes = list(nxu.nodes_between_soma_and_nodes(
            G,
            nodes=nodes,
        ))

        if verbose:
            print(f"Non axon nodes added on path to soma = {new_nodes}")
            
        nodes += new_nodes
        
    return nodes

def compartment_subgraph(
    G,
    compartment,
    include_path_to_soma=False,
    verbose = False,
    plot = False,):
    """
    Purpose: To get the axon skeleton
    (and optionally the skeleton in between axon and soma)


    """
    nodes = nxu.compartment_nodes(
        G,compartment,include_path_to_soma=include_path_to_soma)
    
    if verbose:
        print(f"{compartment} nodes = {nodes}")


    G_sub = G.subgraph(nodes).copy()
    return G_sub

def compartment_skeleton(
    G,
    compartment,
    include_path_to_soma=False,
    include_soma = False,
    verbose = False,
    plot = False,
    **kwargs):
    
    sub_G = nxu.compartment_subgraph(
        G,
        compartment,
        include_path_to_soma=include_path_to_soma,
        verbose = verbose,
        plot = False,
    )
    return nxu.skeleton(
        sub_G,
        include_soma = include_soma,
        plot = plot,
        **kwargs
    )

def axon_skeleton(
    G,
    include_path_to_soma=False,
    verbose = False,
    plot = False,
    **kwargs
    ):
    
    return nxu.compartment_skeleton(
    G,
    compartment="axon",
    include_path_to_soma=include_path_to_soma,
    include_soma = False,
    verbose = verbose,
    plot = plot,
    **kwargs)


def coordinate_estimation_from_upstream_dist_from_node(
    G,
    n,
    attribute= "spine_data",
    ):
    """
    Purpose: To estimate the coordinates of the a data
    attribute with an upstream distance

    Pseudocode: 
    1) Get the skeleton and calculate the distance between each
    2) Get the cumulative distance
    3) Find the two skeleton points it's in between
    4) Do a weighted average of the skeleton points after subtracting the cumulative distance
    
    Ex: 
    nxu.coordinate_estimation_from_upstream_dist_from_node(
        G_postsyn,
        "L1_6"
    ).shape


    """
    if "_data" not in attribute:
        attribute = f"{attribute}_data"

    node_data = G.nodes[n]
    sk_verts = node_data["skeleton_data"]
    spine_upstream = np.array([k["upstream_dist"] for k in node_data["spine_data"]])

    if len(spine_upstream) > 0:
        dists = np.linalg.norm(sk_verts[1:] - sk_verts[:-1],axis=1)
        bins = np.hstack([[0],np.cumsum(dists)])

        highest_bound = np.digitize(spine_upstream,bins)
        highest_bound[highest_bound >= len(bins)] = len(bins) - 1
        highest_bound[highest_bound <= 0] = 1

        # do the weighted average of the to get the actual distance
        dist_from_up = bins[highest_bound] - spine_upstream
        dist_from_down = spine_upstream - bins[highest_bound - 1]
        denom = dist_from_up+dist_from_down
        
        #makes sure no zero denominator values
        dist_from_up[denom == 0] = 1
        denom[denom == 0] = 1

        up_weight = dist_from_up/(dist_from_up+dist_from_down)
        down_weight = 1- up_weight

        spine_shaft_coords = sk_verts[highest_bound]*up_weight.reshape(-1,1) + sk_verts[highest_bound-1]*down_weight.reshape(-1,1)
    else:
        spine_shaft_coords = np.array([]).reshape(-1,3)

    return spine_shaft_coords

    
# purpose: Get the spine data in the area
def spine_shaft_coordinates(
    G,
    verbose = False,
    plot = False,
    mesh = None):
    """
    Purpose: Get all of the spine coordinates (located on the shaft)
    """

    spine_coordinates = []
    for n in nxu.limb_branch_nodes(G):
        spine_coordinates.append(
            nxu.coordinate_estimation_from_upstream_dist_from_node(G,n)
        )

    spine_coordinates = np.vstack(spine_coordinates)
    
    if verbose:
        print(f"# of spine coordinates = {len(spine_coordinates)}")
        
    if plot:
        new_figure = True
        if mesh is not None:
            ipvu.plot_mesh(
                mesh,
                new_figure=True,
                show_at_end=False,
                flip_y=True,
                alpha = 0.25,
            )
            new_figure = False
        ipvu.plot_obj(
            array = spine_coordinates,
            flip_y = True,
            new_figure = new_figure,
        )
        
    return spine_coordinates

def most_upstream_node_on_axon_limb(
    G,
    return_endpoint_upstream=False,
    verbose = False,
    ):

    node = nxu.most_upstream_node(
        G,
        nxu.compartment_nodes(
        G,
        compartment="axon",
        include_path_to_soma=True
        )
    )
    
    if verbose:
        print(f"Most upstream node on axon branch = {node}")
    if return_endpoint_upstream:
        return G.nodes[node]["endpoint_upstream"]
    else:
        return node
    
def skeleton_graph(
    G,
    graph_type="Graph"):
    from mesh_tools import skeleton_utils as sk
    verts,edges = nxu.skeleton(G)
    return sk.graph_from_non_unique_vertices_edges(verts,edges)

def fix_attribute(G,
        attribute="spine",
        verbose = False):
    
    attribute = nu.convert_to_array_like(attribute)
        
    for n in G.nodes():
        if "s" in n.lower():
            continue
        for a in attribute:
            f_name = f"n_{a}s"
            G.nodes[n][f_name] = len(G.nodes[n].get(f"{a}_data",[]))

    return G

def fix_flipped_skeleton(
    G,
    verbose = False,):
    
    """
    Purpose: To fix the skeleton data
    in Graph objects if they are
    not aligned correctly

    Ex: 
    segment_id,split_index = 864691135162621741,0

    G_obj = hdju.graph_obj_from_auto_proof_stage(
        segment_id=segment_id,
        split_index=split_index,
    )

    G_obj = fix_flipped_skeletons(G_obj,verbose = True)
    """

    node_flipped = []
    for n in G.nodes():
        if "s" in n.lower():
            continue
        curr_dict = G.nodes[n]
        if len(curr_dict) == 0:
            continue
        endpt_up = curr_dict["endpoint_upstream"]
        sk_up = curr_dict["skeleton_data"][0]
        if not np.array_equal(endpt_up,sk_up):
            G.nodes[n]["skeleton_data"] = np.flip(curr_dict["skeleton_data"],axis=0)
            node_flipped.append(n)

    if verbose:
        print(f"Nodes with skeleton flipped: {node_flipped}")

    return G


def upstream_limb_branch(G,n):
    up_node = xu.upstream_node(G,n)
    if "S" in up_node:
        return None
    else:
        return up_node
    
def downstream_limb_branch(G,n):
    d_node = xu.downstream_nodes(G,n)
    if len(d_node) == 0:
        return None
    else:
        return d_node
    
def fix_width_inf_nan(
    G,
    default_value = 300,
    verbose = False,):


    """
    Purpose: to replace all inf width values in node
    with either the upstream, downstream width, or default value

    Pseudocode: 
    1a) Try and get an upstream width
    1b) Try and get a downstream width
    1c) Use the default width
    2) Go and replace all width values and in the width array
    with the default value

    Width values to replace: 
    - width (scalar)
    - width_new (a dictionary)
    - width_upstream (scalar)
    - width_downstream (scalar)
    - width_data: list of dict with width as key

    """
    nodes_fixed = []
    for n in nxu.limb_branch_nodes(G):
        try:
            curr_wid = G.nodes[n]["width"]
        except:
            continue
        if nu.is_nan_or_inf(curr_wid):
            nodes_fixed.append(n)
            #1a) Try and get an upstream width
            up_node = nxu.upstream_limb_branch(G,n)
            new_width = default_value
            if up_node is not None:
                try:
                    new_width = G.nodes[up_node]["width_new"]["no_spine_median_mesh_center"]
                except:
                    pass
            else:
                d_nodes = nxu.downstream_limb_branch(G,n)
                if d_nodes is not None:
                    try:
                        new_width = np.mean([G.nodes[k]["width_new"]["no_spine_median_mesh_center"] for k in d_nodes])
                    except:
                        pass

            # do the replacement of all the width values
            G.nodes[n]["width"] =new_width
            G.nodes[n]["width_new"] = {k:new_width for k in G.nodes[n]["width_new"]}
            G.nodes[n]["width_upstream"] =new_width
            G.nodes[n]["width_downstream"] =new_width
            for k in G.nodes[n]["width_data"]:
                k["width"] = new_width

    if verbose:
        print(f"nodes_fixed = {nodes_fixed}")
    return G  


def soma_vector_from_node(
    G,
    n,
    soma_coordinate = None,
    normalize = True,
    verbose = False,  
    ):
    """
    Purpose: To get the angle between a node
    and the soma a node
    
    Ex:
    nxu.soma_vector_from_node(
        G = G,
        soma_coordinate = None,
        normalize = True,
        n = "L0_0",
    )
    """
    if soma_coordinate is None:
        soma_coordinate = nxu.soma_center(G)

    soma_vector = G.nodes[n]["endpoint_upstream"] - soma_coordinate
    if normalize:
        soma_vector = soma_vector / np.linalg.norm(soma_vector)

    if verbose:
        print(f"soma_vector = {soma_vector}")
        
    return soma_vector

def skeleton_vector_upstream_and_width_from_node(
    G,
    n,
    upstream_dist_max = None,#np.inf
    verbose = False,
    return_width = True,
    ):
    """
    Purpose: To get the skeleton vector of a node pointing in downstream
    (and may be restricted by certain downstream distance)

    - can return the associated width

    Pseudocode: 
    1) Find the n_idx to restrict to for the downstream dist
    
    Ex: 
    nxu.skeleton_vector_upstream_and_width_from_node(
        G = G_limbs[0],
        n = "L0_22",
        upstream_dist_max = 3000,#np.inf
        verbose = True,
        return_width = True,

    )

    """
    if upstream_dist_max is None:
        upstream_dist_max = default_upstream_dist_max
    
    node_data = G.nodes[n]

    idx = [k for k in node_data["width_data"]
              if k["upstream_dist"] <= upstream_dist_max]
    max_idx = len(idx)

    if verbose:
        print(f"max_idx = {max_idx} (out of {len(node_data['width_data'])} nodes)")

    skeleton_nodes = node_data["skeleton_data"][:max_idx+1]
    skeleton_vector = skeleton_nodes[-1,:] - skeleton_nodes[0,:]
    skeleton_vector = skeleton_vector/np.linalg.norm(skeleton_vector)

    if verbose:
        print(f"skeleton_vector = {skeleton_vector}")

    # find the width associated with the skeleton
    skeleton_lengths = np.linalg.norm(skeleton_nodes[1:]-skeleton_nodes[:-1],axis=1)
    widths = [k["width"] for k in node_data["width_data"][:max_idx]]

    weighted_width = nu.weighted_average(widths,skeleton_lengths)
    if verbose:
        print(f"widths = {widths}, skeletal_seg_lengths = {skeleton_lengths}")
        print(f"weighted_width = {weighted_width}")

    if return_width:
        return skeleton_vector,weighted_width
    else:
        return skeleton_vector
    
    
def most_upstream_node_vector_stats(
    G,
    soma_coordinate=None,
    most_upstream_node=None,
    upstream_dist_max=None,
    verbose = False
    ):

    """
    Purpose: to compute the soma and skeleton
    vector and other statitistics about the
    most upstream node on a graph
    """
    return_dict = dict()
    
    if upstream_dist_max is None:
        upstream_dist_max = default_upstream_dist_max
    
    subG = G
    if most_upstream_node is None:
        most_upstream_node = xu.most_upstream_node(subG)
    if verbose:
        print(f"    most_upstream_node = {most_upstream_node}")
        

    # a) Determine the soma vector
    
    soma_coord_dict = dict()
    if soma_coordinate is not None:
        soma_vector = nxu.soma_vector_from_node(
            G = G,
            soma_coordinate = soma_coordinate,
            normalize = True,
            n = most_upstream_node,
        )

        if verbose:
            print(f"soma_vector = {soma_vector}")
            
        soma_coord_dict = dict(
        soma_vector_x_nm = soma_vector[0],
        soma_vector_y_nm = soma_vector[1],
        soma_vector_z_nm = soma_vector[2],
        )
        


    # b) Determine the skeleton vector
    skeleton_vector,width = nxu.skeleton_vector_upstream_and_width_from_node(
        G = G,
        n = most_upstream_node,
        upstream_dist_max = upstream_dist_max,#np.inf
        verbose = False,
        return_width = True,

    )
    
    if verbose:
        print(f"skeleton_vector = {skeleton_vector}")
        print(f"width = {width}")

    n = most_upstream_node
    return_dict = dict(
        node=most_upstream_node,
        endpoint_upstream_x_nm=G.nodes[n]["endpoint_upstream"][0],
        endpoint_upstream_y_nm=G.nodes[n]["endpoint_upstream"][1],
        endpoint_upstream_z_nm=G.nodes[n]["endpoint_upstream"][2],
        endpoint_downstream_x_nm=G.nodes[n]["endpoint_downstream"][0],
        endpoint_downstream_y_nm=G.nodes[n]["endpoint_downstream"][1],
        endpoint_downstream_z_nm=G.nodes[n]["endpoint_downstream"][2],
        skeleton_vector_x_nm = skeleton_vector[0],
        skeleton_vector_y_nm = skeleton_vector[1],
        skeleton_vector_z_nm = skeleton_vector[2],
        width = width,

    )
    
    return_dict.update(soma_coord_dict)
    
    return return_dict

# def skeletal_length_on_G(
#     G,
#     ):
#     return np.sum([G.nodes[k]["skeletal_length"] for k in G.nodes()])

def adjusted_scholl_coordinates(
    array,
    soma_coordinate,
    upstream_endpoint,
    verbose = False,
    ):
    
    adjusted_offset = upstream_endpoint - soma_coordinate
    if verbose:
        print(f"Adjusted offset = {adjusted_offset}")
    array = array - adjusted_offset
    return array

def scholl_coordinates(
    G,
    soma_coordinate,
    axes = None,
    interval = None,
    max_distance = None,
    verbose = False,
    return_dict = True,
    adjusted = True,
    plot = False,
    ):
    """
    Purpose: to calculate the coordinates
    of the skeleton that are interval distances
    away from the soma center
    
    Pseudocode: 
    1) Calculate the intervals will measure at based
    on max_distance or max of the G skeleton
    """
    if interval is None:
        interval = default_scholl_interval
    
    if axes is None:
        axes = np.array([0,2])
    
    if "np" not in str(type(G)):
        skeleton = nxu.skeleton(G,return_verts_edges=False)
        
    if adjusted:
        upstream_endpoint = nxu.most_upstream_coordinate(G)
        skeleton = adjusted_scholl_coordinates(
            skeleton,
            soma_coordinate = soma_coordinate,
            upstream_endpoint=upstream_endpoint,
            verbose = verbose)
#         upstream_endpoint = nxu.most_upstream_coordinate(G)
#         adjusted_offset = upstream_endpoint - soma_coordinate
#         if verbose:
#             print(f"Adjusted offset = {adjusted_offset}")
#         skeleton = skeleton - adjusted_offset
        
    # limiting the skeleton and center to certain axes
    skeleton_axes = skeleton[:,:,axes]
    soma_center_axes = soma_coordinate[axes]
    
        
    if max_distance is None:
        max_distance = np.max(
            np.linalg.norm(skeleton_axes.reshape(-1,len(axes))-soma_center_axes,axis=1)
        )
        
        if verbose:
            print(f"max_distance computed = {max_distance}")
            
    radius_intervals = np.arange(0,max_distance+0.0001,interval)[1:]
    
    if verbose:
        print(f"radius_intervals = {radius_intervals}")
        
    data_points = {
        k:nsku.cirle_intersections(
            skeleton,
            soma_coordinate,
            radius=k,
            axes=axes,
            verbose = verbose,
            plot=plot) for k in radius_intervals
    }
    
    if adjusted:
        data_points = dict([(k,v + adjusted_offset) if len(v) > 0 else (k,v)
            for k,v in data_points.items()])
    
    if return_dict:
        return data_points
    else:
        return list(data_points.values())
    
def vector_stats_from_G(
    G,
    soma_coordinate=None,
    upstream_dist_max = None,
    include_scholl_coordinates = False,
    include_scholl_coordinates_adjusted = False,
    include_bbox = True,
    include_internal_branching = True,
    include_n_scholl = True,
    plot_scholl = False,
    verbose = False,
    ):
    
    """
    Purpose: To compute the following statistics for 
    a subgraph of a neuron object

    a. starting width
    b. starting y above the soma
    c. soma angle
    d. skeletal angle
    e. skeletal length
    f. starting endpoint
    g. intersection points of skeleton 
    (for as far as can go or max distance)
    """
    
    if upstream_dist_max is None:
        upstream_dist_max = default_upstream_dist_max
    
    skeletal_length = nsku.skeletal_length(G)
    most_upstream_node = xu.most_upstream_node(G)

    if verbose:
        print(f"most_upstream_node = {most_upstream_node}")

    curr_dict = nxu.most_upstream_node_vector_stats(
        G=G,
        most_upstream_node=most_upstream_node,
        soma_coordinate=soma_coordinate,
        upstream_dist_max=upstream_dist_max,
        verbose = verbose,
        )

    curr_dict["skeletal_length"] = skeletal_length
    if soma_coordinate is not None:
        curr_dict["y_soma_relative"] =  curr_dict["endpoint_upstream_y_nm"] - soma_coordinate[1]
    curr_dict["n_nodes"] = len(G.nodes())
    
    
    leaf_nodes = xu.leaf_nodes(G)
    curr_dict["leaf_nodes"] = leaf_nodes
    curr_dict["n_leaf_nodes"] = len(leaf_nodes)
    
    if include_scholl_coordinates and soma_coordinate is not None:
        curr_dict["scholl_coords"] = nxu.scholl_coordinates(
            G,
            soma_coordinate=soma_coordinate,
            verbose = plot_scholl,
            plot = plot_scholl,
            adjusted = False,
        )
        
        if include_n_scholl:
            curr_dict.update(
                n_scholl_dict_from_scholl_dict(
                    curr_dict["scholl_coords"],
                    prefix = ""
                )
            )
        
    if include_scholl_coordinates_adjusted and soma_coordinate is not None:
        curr_dict["scholl_coords_adjusted"] = nxu.scholl_coordinates(
            G,
            soma_coordinate=soma_coordinate,
            verbose = plot_scholl,
            plot = plot_scholl,
            adjusted = True,
        )
        
        if include_n_scholl:
            curr_dict.update(
                n_scholl_dict_from_scholl_dict(
                    curr_dict["scholl_coords_adjusted"],
                    prefix = "_adjusted"
                )
            )
    
    if include_internal_branching:
        curr_dict.update(nxu.internal_branching_stats(G))
        
    if include_bbox:
        curr_dict.update(nxu.skeleton_bbox(G,return_dict = True))
    
    return curr_dict



        

def compartment_vector_width_stats_from_G(
    G,
    small_starter_branch_skeletal_length_min = 1000,
    upstream_dist_max = None,
    min_skeletal_length_limb = 10_000,#50_000,
    verbose = False,
    ):
    """
    Purpose: To determine the max vector
    and weighted vector of all the compartments of a neuron

    1) Download the neuron graph
    2) Filter the neuron graph for small starter branches
    3) For each limb and then for each compartment:
        a. Find if nodes exist with that label
        b. Find the most upstream node
        c. Determine the width of that node (using the width data and upstream search distance)
        d. Save off the soma starting angle of that branch and the current skeleton angle and width

    4) For each compartment:
        a. Find the soma starting vector and skeleton vector of the weighted options (weight by width_max and width)

    5) Save off curr data to datajoint for each compartment:
    - angles (max and not max)
    - n_limbs in compartment

    """
    if upstream_dist_max is None:
        upstream_dist_max = default_upstream_dist_max
    
    
    G_filt = nxu.remove_small_starter_branches(
                G,
                verbose = verbose,
                skeletal_length_min=small_starter_branch_skeletal_length_min,
                maintain_skeleton_connectivity = True)
    soma_coordinate = nxu.soma_center(G_filt)

    compartments = ["axon","dendrite","apicalshaft","basal"]
    comp_dict = {k:[] for k in compartments}



    G_limbs = nxu.all_limb_graphs_off_soma(G_filt)
    for j,limb in enumerate(G_limbs):
        if verbose:
            print(f"---Working on limb {j}---")


        for comp in compartments:
            subG = nxu.compartment_subgraph(limb,comp)

            # check if any nodes with that label
            if len(subG) == 0 and comp == 'apicalshaft':
                subG = nxu.compartment_subgraph(limb,"apical")


            if len(subG) == 0:
                if verbose:
                    print(f"No {comp} nodes in subgraph")
                continue
            else:
                if verbose:
                    print(f"# of {comp} nodes = {len(subG)}")

            most_upstream_node = xu.most_upstream_node(subG)
            G_downstream = limb.subgraph(xu.all_downstream_nodes(
                 limb,
                most_upstream_node,
                include_self=True
            )).copy()

            limb_skeletal_length = nsku.skeletal_length(G_downstream)

            if verbose:
                print(f"limb_skeletal_length = {limb_skeletal_length}")

            if limb_skeletal_length < min_skeletal_length_limb:
                if verbose:
                    print(f"   *** skipping limb {j} because skeletal_length ({limb_skeletal_length})"
                         f" was less than minimum ({min_skeletal_length_limb})")
                continue

            curr_dict = nxu.most_upstream_node_vector_stats(
                G=subG,
                most_upstream_node=most_upstream_node,
                soma_coordinate=soma_coordinate,
                upstream_dist_max=upstream_dist_max,
                verbose = verbose,
                )
            
            curr_dict["skeletal_length"] = limb_skeletal_length

            comp_dict[comp].append(curr_dict)

    """
    Purpose: For each compartment:
        a. Find the soma starting vector and skeleton vector of the weighted options (weight by width_max and width)

    Pseudocode:
    For each compartment:
    a. Find the max width
    b. keep that dict for the max width one (add max to the labels)
    c. For vectors (soma and skeleton), do a weighted average with all widths and add to dict
    d. add compartment prefixs to dictionary and add to list

    """


    total_comp_dict = dict()
    for comp in compartments:
        if verbose:
            print(f"-- Working on consolidating {comp} --")
        curr_dicts = comp_dict[comp]
        if len(curr_dicts) == 0:
            continue
        widths = np.array([k["width"] for k in curr_dicts])
        if verbose:
            print(f"widths = {widths}")

        width_max_idx = np.argmax(widths)
        max_dict = curr_dicts[width_max_idx]

        #do weighted average of all the vectors
        soma_vectors = np.vstack([np.array([k[f"soma_vector_{v}_nm"] for v in ["x","y","z"]]) for k in curr_dicts])
        skeletal_vectors = np.vstack([np.array([k[f"skeleton_vector_{v}_nm"] for v in ["x","y","z"]]) for k in curr_dicts])

        soma_vector_weighted=nu.weighted_average_along_axis(
            soma_vectors,weights=widths,axis=0
        )

        skeleton_vector_weighted=nu.weighted_average_along_axis(
            skeletal_vectors,weights=widths,axis=0
        )

        final_dict = dict(
            max_dict.copy(),
            n_limbs = len(curr_dicts),
            soma_vector_weighted_x_nm = soma_vector_weighted[0],
            soma_vector_weighted_y_nm = soma_vector_weighted[1],
            soma_vector_weighted_z_nm = soma_vector_weighted[2],

            skeleton_vector_weighted_x_nm = skeleton_vector_weighted[0],
            skeleton_vector_weighted_y_nm = skeleton_vector_weighted[1],
            skeleton_vector_weighted_z_nm = skeleton_vector_weighted[2],
        )

        if comp == "apicalshaft":
            comp = "apical"
        #append the compartment name
        final_dict = {f"{comp}_{k}":v for k,v in final_dict.items()}
        total_comp_dict.update(final_dict)
        
    return total_comp_dict

def G_no_soma(G,soma_node_name = None):
    if soma_node_name is None:
        soma_node_name = nxu.soma_node_name_global
    return xu.remove_nodes_from(G,[soma_node_name])

def all_node_graphs(G,verbose = False):
    G = nxu.G_no_soma(G)             
    if verbose:
        print(f"components = {components}")
    
    return_graphs = [G.subgraph([n]).copy()
                     for n in G.nodes()]
    
    if verbose:
        print(f"# of node subgraphs = {len(return_graphs)}")
    
    return return_graphs

def skeleton_downstream_of_node(
    G,
    node,
    include_self=True,
    return_verts_edges = True,):
    return nxu.skeleton(
        xu.subgraph_downstream_of_node(G,node,include_self=include_self),
        return_verts_edges=return_verts_edges,
    )

def all_compartment_subgraphs(
    G,
    compartments = None,
    verbose = False,
    return_empty_graphs = False,
    ):
    
    if compartments is None:
        #compartments = getattr(nxu,f"compartments_{cell_type}")
        compartments = getattr(nxu,f"compartments")
        
    if verbose:
        print(f"Compartments Trying = {compartments}")
        
    comp_dict = {k:nxu.compartment_subgraph(G,k) for k in compartments}
    
    if not return_empty_graphs:
        comp_dict = {k:v for k,v in comp_dict.items() if len(v.nodes()) > 0}
        
    if verbose:
        print(f"Compartments Returning = {list(comp_dict.keys())}")
        
    return comp_dict
        
def all_compartment_skeletons(
    G,
    compartments = None,
    verbose = False,
    plot = False,
    mesh = None,
    return_empty_skeletons = False,
    ):
    
    """
    Purpose: To get all the compartment skeletons
    from a graph
    """
    
    comp_subs = nxu.all_compartment_subgraphs(
        G,
        compartments = compartments,
        verbose =verbose,
        return_empty_graphs = return_empty_skeletons,
        )
    comp_dict = {k:nxu.skeleton(sub_G,include_soma = False) for k,sub_G in comp_subs.items()}
    
    
    if plot:
        new_figure = True
        if mesh is not None:
            ipvu.plot_mesh(
                mesh,
                new_figure=True,
                show_at_end=False,
                flip_y=True,
                alpha = 0.1
            )
            new_figure = False
            
        comp_dict_to_plot = {k:v for k,v in comp_dict.items() if len(v[0]) > 0}
        for j,(k,(skeleton_nodes,skeleton_edges)) in enumerate(comp_dict_to_plot.items()):
            color = nxu.compartment_colors[k]
            
            if j == len(comp_dict_to_plot) - 1:
                show_at_end=True
            else:
                show_at_end = False
            
            ipvu.plot_skeleton(
               skeleton_nodes,
                skeleton_edges,
                new_figure = new_figure,
                show_at_end = show_at_end,
                color = color
            )
            
            new_figure = False
            
    return comp_dict

def plot_all_skeleton_compartments(G,mesh=None,**kwargs):
    nxu.all_compartment_skeletons(G,plot=True,mesh=mesh,**kwargs)
    
    

def skeleton_edge_df_with_edge_graph_with_compartments(
    G,
    verbose = True,
    plot = False,
    return_edge_G = True,
    remove_starter_branches = False,
    ):

    """
    Purpose: To get all the edge data for all compartments
    and to have a graph representation of those edges so can 
    walk along:

    Pseudocode: 
    1) Reduce the vertices down to unique vertices and edges (and pass back the index)
    2) Turn the vertices into a graph (and each node should have a compartment)
    3) Trun graph into an edge graph
    --> use the new node names to index into compartment and get the mode
    --> use node names to index into coordinates and compute: skeletal length and midpoint

    store all in dataframe for the edges

    """
    if remove_starter_branches:
        G = nxu.remove_small_starter_branches(G)

    comp_skeletons = nxu.all_compartment_skeletons(G,plot=False,)

    nodes_list = []
    edges_list = []
    comp_list = []

    counter = 0
    for comp,(nodes,edges) in comp_skeletons.items():
        if verbose:
            print(f"Working on {comp}")
        if len(nodes) == 0:
            continue

        nodes_list.append(nodes)
        edges_list.append(edges + counter)
        comp_list.append(np.repeat(comp,len(nodes)))

        counter += len(nodes)

    nodes_list = np.vstack(nodes_list)
    edges_list = np.vstack(edges_list)
    comp_list = np.hstack(comp_list)

    nodes_unique,edges_unique,node_orig_idx = xu.unique_vertices_edges_from_vertices_edges(
        nodes_list,
        edges_list,
        return_vertex_index = True
    )

    comps_per_node = comp_list[node_orig_idx]
    coord_per_nodes = nodes_list[node_orig_idx]

    G_node = xu.graph_from_unique_vertices_edges(
        nodes_unique,
        edges_unique
    )

    G_edge = xu.edge_graph(G_node)
    edge_nodes = np.array(list(G_edge.nodes())).astype('int')


    edge_dict = dict()
    edge_dict["edge_coord"] = coord_per_nodes[edge_nodes]
    edge_dict["compartment"] = nu.mode(comps_per_node[edge_nodes],axis=1)
    edge_dict["skeletal_length"] = np.linalg.norm(
        coord_per_nodes[edge_nodes][:,1]-
        coord_per_nodes[edge_nodes][:,0],axis=1)
    edge_dict["midpoint"] = np.mean(coord_per_nodes[edge_nodes],axis=1)
    edge_df = pd.DataFrame.from_dict({k:list(v) for k,v in edge_dict.items()})
    
    G_edge = xu.xu.relabel_node_names(
        G_edge,{tuple(k):i for i,k in enumerate(list(edge_nodes))}
    )

    # plot to make sure coordinates came out well

    #plot to make sure assembled correctly
    if plot:
        ipvu.plot_scatter(
            np.vstack(edge_df["midpoint"].to_numpy()),
            color=[nxu.compartment_colors[k] for k in edge_df["compartment"].to_list()],
        )
        
    if return_edge_G:
        return edge_df,G_edge
    else:
        return edge_df
    
def all_compartment_conn_comp_subgraphs(
    G,
    compartments = None,
    verbose = False,
    return_compartments = True,
    ):
    """
    Purpose: Divide a neuron graph
    into the connected components of each
    compartment type (so 2 oblique branches
    should be 2 different subgraphs)
    """
    #1) Divide neuron into different compartment
    comp_subgraphs = nxu.all_compartment_subgraphs(
        G,
        verbose = verbose,
        compartments=compartments)

    #2) Divide all subgraphs into connected components
    all_subgraphs = []
    all_subgraphs_comps = []
    
    for comp,sub_G in comp_subgraphs.items():
        comp_Gs = xu.connected_components_subgraphs(sub_G)
        all_subgraphs += comp_Gs
        all_subgraphs_comps += [comp]*len(comp_Gs)

    if verbose:
        print(f"Total number of subgraphs = {len(all_subgraphs)}")
        if len(all_subgraphs_comps) > 0:
            lab,count = np.unique(all_subgraphs_comps,return_counts=True)
            for l,c in zip(lab,count):
                print(f"   {l}:{c}")

    if return_compartments:
        return all_subgraphs,all_subgraphs_comps
    else:
        return all_subgraphs
    
    
def all_compartment_conn_comp_subgraph_vector_stats(
    G,
    small_starter_branch_skeletal_length_min = 1000,
    filter_starter_branches = True,
    upstream_dist_max = None,
    verbose = True,
    debug_idx = None,
    mesh = None,
    return_subgraphs = False,
    include_scholl_coordinates = True,
    include_scholl_coordinates_adjusted = True,
    return_df = False,
    ):
    """
    Purpose: To divide a neuron into individual connected components of the
    interested compartments. and then to calculate the following stats about them

    1) soma vector
    2) skeleton vector
    3) width
    4) y above the soma
    5) skeletal length
    6) The intersection points at different 
    lengths from the soma (on the xz plane)
    - predefine a stepsize and go as far as need be


    Pseudocode: 
    1) Divide neuron into different compartment
    subgraphs
    2) Divide all subgraphs into connected components
    (put into one list)
    3) For each graph in the list compute:
    a. starting width
    b. starting y above the soma
    c. soma angle
    d. skeletal angle
    e. skeletal width
    f. intersection points of skeleton 
    g. starting endpoint
    (for as far as can go or max distance)
    """

    if upstream_dist_max is None:
        upstream_dist_max = default_upstream_dist_max

    if debug_idx is None:
        debug_idx = []


    #0) Filter the starter branches
    if filter_starter_branches:
        G_filt = nxu.remove_small_starter_branches(
            G,
            verbose = verbose,
            skeletal_length_min=small_starter_branch_skeletal_length_min,
            maintain_skeleton_connectivity = True
        )

    #1) Getting the soma coordinate (for later computation)
    soma_coordinate = nxu.soma_center(G_filt)

    comp_subgraphs,comps = nxu.all_compartment_conn_comp_subgraphs(
        G_filt,
        verbose = verbose
    )

    all_dicts = []
    for idx in range(len(comp_subgraphs)):
        subG,comp = comp_subgraphs[idx],comps[idx]

        if verbose:
            print(f"--- Working on subgraph {idx}: comp = {comp}, n_nodes = {len(subG.nodes())} ")

        curr_dict = nxu.vector_stats_from_G(
            subG,
            soma_coordinate=soma_coordinate,
            include_scholl_coordinates = include_scholl_coordinates,
            include_scholl_coordinates_adjusted = include_scholl_coordinates_adjusted,
            verbose = verbose
        )

        curr_dict["compartment"] = comp
        curr_dict["subgraph_idx"] = idx
        
        if return_subgraphs:
            curr_dict["G"] = subG


        if idx in debug_idx:
            nviz.plot_objects(
                mesh,
                scatters = [
                   G.nodes[curr_dict["node"]]["skeleton_data"],
                    np.vstack(curr_dict["scholl_coords"].values())
                ],
            )

        all_dicts.append(curr_dict)
        
    if return_df:
        return pd.DataFrame.from_records(all_dicts)
    else:
        return all_dicts

def skeleton_bbox(
    G,
    return_dict = True,
    verbose = False,
    suffix = "_nm",
    ):
    """
    Purpose: To find the skeleton's bounding box

    1) Generate the skeleton
    2) Send the vertices to the bounding box
    3) Return dict
    """
    verts,_ = nxu.skeleton(G,return_verts_edges=True)
    bbox = nu.bouning_box_corners(verts,return_dict = return_dict)
    
    if return_dict:
        bbox = {f"{k}{suffix}":v for k,v in bbox.items()}
    if verbose:
        print(f"bbox = {bbox}")
        
    return bbox

def internal_branching_coordinates(
    G,
    verbose = False,):
    """
    Purpose: Calculate the internal branch
    coordinates

    Application: Can calculate the closest and farthest away 
    internal branch points

    Pseudocode: 
    2) Find all nodes not the leaf nodes and record their endpoints
    """
    non_leaf = xu.non_leaf_nodes(G)
    internal_coordinates = np.array([G.nodes[k]["endpoint_downstream"] for k in non_leaf])
    if len(internal_coordinates) > 0:
        internal_coordinates = np.vstack(internal_coordinates)

    if verbose:
        print(f"internal_coordinates = {internal_coordinates}")
    return internal_coordinates

def n_internal_branching_coordinates(G,**kwargs):
    return len(nxu.internal_branching_coordinates(G,**kwargs))

def internal_branching_coordinates_max_min(
    G,
    default_value = None,
    verbose = False,
    plot = False,
    return_dict = False,
    return_distance = False,
    suffix = "_nm"):
    """
    Purpose: Find the farthest and closest
    internal branching point

    Pseudocode: 
    1) Get the most upstream endpoint
    2) Get all the internal branch points
    3) Calculate the distance of all the internal branch points
    4) Pick the largest and smallest distance
    
    Ex: 
    nxu.internal_branching_coordinates_max_min(
        subG,
        plot = True,
        return_dict = True,
        verbose = True,
    )
    """

    coord = nxu.most_upstream_coordinate(G)
    branch_coords = nxu.internal_branching_coordinates(G)
    if len(branch_coords) > 0:
        dists = np.linalg.norm(branch_coords- coord,axis=1)
        max_idx = np.argmax(dists)
        min_idx = np.argmin(dists)
        
        max_dist = dists[max_idx]
        min_dist = dists[min_idx]
        
        if verbose:
            print(f"Max branch coord dist = {max_dist}")
            print(f"Min branch coord dist = {min_dist}")

        max_branch_coord = branch_coords[max_idx]
        min_branch_coord = branch_coords[min_idx]

    else:
        max_branch_coord=max_branch_coord=max_dist=min_dist= default_value
        
        
    if verbose:
        print(f"max_branch_coord = {max_branch_coord}")
        print(f"min_branch_coord = {min_branch_coord}")
        
    if plot:
        ipvu.plot_skeleton(
            *nxu.skeleton(G,return_verts_edges=True),
            show_at_end = False
        )
        max_branch_color = "red"
        min_branch_color = "blue"
        start_coordinate = "black"
        print(f"max_branch_color = {max_branch_color}, min_branch_color = {min_branch_color}, start_coordinate = {start_coordinate}")
        ipvu.plot_multi_scatters(
            scatters=[max_branch_coord,min_branch_coord,coord],
            color=[max_branch_color,min_branch_color,start_coordinate],
            new_figure=False,
        )

    if return_dict:
        if return_distance:
            return {
                f"internal_branch_coordinate_max_dist{suffix}":max_dist,
                f"internal_branch_coordinate_min_dist{suffix}":min_dist,
            }
        else:
            if max_branch_coord is None:
                max_branch_coord = min_branch_coord = [None,None,None]
            return {
                f"internal_branch_coordinate_max_x{suffix}":max_branch_coord[0],
                f"internal_branch_coordinate_max_y{suffix}":max_branch_coord[1],
                f"internal_branch_coordinate_max_z{suffix}":max_branch_coord[2],

                f"internal_branch_coordinate_min_x{suffix}":min_branch_coord[0],
                f"internal_branch_coordinate_min_y{suffix}":min_branch_coord[1],
                f"internal_branch_coordinate_min_z{suffix}":min_branch_coord[2],
            }
       
    if return_distance:
        return max_dist,min_dist
    else:
        return max_branch_coord,min_branch_coord

def internal_branching_stats(
    G,
    include_max_min_coordinates = True,
    include_max_min_dist = True,
    **kwargs):
    curr_dict = dict(
        n_internal_branching = nxu.n_internal_branching_coordinates(G)
    )

    if include_max_min_coordinates:
        max_min_branch_dict = nxu.internal_branching_coordinates_max_min(
            G,
            return_distance = False,
            return_dict = True,
            **kwargs
        )
        
        curr_dict.update(max_min_branch_dict)
        
    if include_max_min_dist:
        max_min_branch_dict = nxu.internal_branching_coordinates_max_min(
            G,
            return_distance = True,
            return_dict = True,
            **kwargs
        )
        
        curr_dict.update(max_min_branch_dict)
    
    return curr_dict

default_small_endnodes_skeltal_length_min = 10_000
def small_endnode_branches(
    G,
    dendrite_only = True,
    skeletal_length_min = None,
    exclude_soma_connected_nodes = True,
    verbose = False,
    ):
    """
    Purpose: To detect any end nodes
    that are below a certain skeletal length
    
    Pseudocode: 
    1) Find all of the leaf nodes
    2) Find the skeletal length of all the leaf nodes
    3) Identify any that are subthreshold
    """
    if dendrite_only:
        G = nxu.dendrite_subgraph(G)
    
    if skeletal_length_min is None:
        skeletal_length_min = default_small_endnodes_skeltal_length_min
    
    if nxu.soma_only_graph(G):
        return []
    
    leaf_nodes = xu.leaf_nodes(G)
    
    if exclude_soma_connected_nodes:
        leaf_nodes = np.setdiff1d(leaf_nodes,nxu.soma_connected_nodes(G))
    
    if verbose:
        print(f"Leaf nodes (after soma connected excluded): {leaf_nodes}")
        
    leaf_nodes_sub_thresh = [k for k in leaf_nodes if G.nodes[k]["skeletal_length"] < skeletal_length_min]
    
    if verbose:
        print(f"Leaf nodes (after skeletal length min {skeletal_length_min}) = {leaf_nodes_sub_thresh}")
        
    return leaf_nodes_sub_thresh

def remove_small_endnode_branches(
    G,
    skeletal_length_min = None,
    inplace = False,
    loop_until_fail = True,
    verbose=False,
    **kwargs
    ):
    """
    Purpose: To iteratively remove small endnode branches
    from a graph

    Application: When split into separate path graphs,
    don't want more paths created just because have small
    endnodes on them
    """

    if not inplace:
        G = copy.deepcopy(G)

    while True:
        sm_end_branches = nxu.small_endnode_branches(
            G,
            #skeletal_length_min = 20_000,
            verbose = False,
            **kwargs
        )

        if verbose:
            print(f"Soma End Node Branches = {sm_end_branches}")

        G.remove_nodes_from(sm_end_branches)

        if not loop_until_fail:
            break
        elif len(sm_end_branches) == 0:
            break
        else:
            continue

    return G

def n_scholl_dict_from_scholl_dict(
    scholl_dict,
    return_named_dict = True,
    prefix = ""
    ):
    
    n_scholl_dict = {k:len(v) for k,v in scholl_dict.items()}
    if return_named_dict:
        n_scholl_dict = {f"n_scholl{prefix}_{int(k)}":v for k,v in n_scholl_dict.items()}
    
    return n_scholl_dict

def subgraph_df_and_endnodes_df(
    G,
    remove_small_endnodes = True,
    include_scholl_coordinates = True,
    include_scholl_coordinates_adjusted = True,
    include_n_scholl = True,
    verbose = False,
    endnode_name = "leaf_node",
    ignore_axon = True,
    ):
    """
    Purpose: To compute both the subgraph vector dataframe and the 
    offshoot vector dataframes

    Pseudocode: 
    1) Calculate the subgraph df. WILL RETURN
    2) For each compartment subgraph (each entry in the subgraph df):
        For All end-nodes
        a. Run the stats
        b. Add a leaf node column
        c. Add on the endpoint coordinate
        d. Attributes to add on: 
        - subgraph_idx
        - endnode_idx
    3) Compile into endnode dataframe
    """

    if remove_small_endnodes:
        G = nxu.remove_small_endnode_branches(G,verbose= False)

    soma_coordinate = nxu.soma_center(G)

    #1) Calculate the subgraph df. WILL RETURN
    subgraph_df= nxu.all_compartment_conn_comp_subgraph_vector_stats(
        G,
        verbose = False,
        return_subgraphs=True,
        return_df = True
    )

    endnode_dicts = []
    for subgraph_dict in pu.df_to_dicts(subgraph_df):
        subgraph_idx = subgraph_dict["subgraph_idx"]
        leaf_nodes = subgraph_dict[f"{endnode_name}s"]
        
        if (subgraph_dict["compartment"] == "axon") and ignore_axon == True:
            continue
            
        subG = subgraph_dict["G"]
        if verbose:
            print(f"    -- Working on subgraph {subgraph_idx} --")

        for j,endnode in enumerate(leaf_nodes):
            if verbose:
                print(f"       -- working on endnode {leaf_nodes}")

            endG = xu.shortest_path_graph_from_most_upstream(
                subG,
                endnode,
            )

            #a. Run the stats
            curr_dict = nxu.vector_stats_from_G(
                endG,
                soma_coordinate=soma_coordinate,
                include_scholl_coordinates = include_scholl_coordinates,
                include_scholl_coordinates_adjusted = include_scholl_coordinates_adjusted,
                include_n_scholl = include_n_scholl,
                verbose = verbose,
            )

            #b. Add a leaf node column
            curr_dict[f"{endnode_name}"] = endnode

            #c. Add on the endpoint coordinate
            endnode_coordinate = endG.nodes[endnode]["endpoint_downstream"]
            curr_dict[f"{endnode_name}_coordinate_x_nm"] = endnode_coordinate[0]
            curr_dict[f"{endnode_name}_coordinate_y_nm"] = endnode_coordinate[1]
            curr_dict[f"{endnode_name}_coordinate_z_nm"] = endnode_coordinate[2]
            curr_dict[f"{endnode_name}_dist"] = np.linalg.norm(
                soma_coordinate - endnode_coordinate)

            #d. Attributes to add on: subgraph_idx, endnode_idx
            curr_dict["subgraph_idx"] = subgraph_idx
            curr_dict[f"{endnode_name}_idx"] = j

            endnode_dicts.append(curr_dict)

    endnode_df = pd.DataFrame.from_records(endnode_dicts)

    if verbose:
        print(f"# of subgraph entries = {len(subgraph_df)}")
        print(f"# of endnode entries = {len(endnode_df)}")

    return subgraph_df,endnode_df

def shortest_path_from_most_upstream(
    G,
    node,
    nodes_to_exclude = ("S0",),
    **kwargs
    ):
    """
    Ex: 
    nxu.shortest_path_from_most_upstream(
        G,
        node="L0_1",
    )
    """
    
    return xu.shortest_path_from_most_upstream(
    G,
    node=node,
    nodes_to_exclude = nodes_to_exclude,
    **kwargs
    )

def node_endpoint(G,node):
    node= nu.array_like(node)
    return np.vstack([
        G.nodes[n]["endpoint_downstream"]
        for n in node])


def coordinates_from_coordinate_type(
    G,
    node,
    coordinate_type,
    ):
    node_dict = G.nodes[node]
    
    if coordinate_type == "skeleton":
        coordinate_type = "skeleton_data"
        
    if coordinate_type == "endpoints":
        coordinates = np.vstack([node_dict["endpoint_upstream"],
                                 node_dict["endpoint_downstream"],]).reshape(-1,3)
    else:
        coordinates = np.array(node_dict[coordinate_type]).reshape(-1,3)
        
    return coordinates
def coordinate_array_with_node_map_from_G(
    G,
    coordinate_type = "skeleton",
    nodes = None,
    suppress_errors = False,
    verbose = True,
    ):
    """
    Purpose: to get two parallel arrays
    of coordinates pulled from a node somhow
    (skeletons points, midpoints, endpoints...)
    and the node name

    Pseudocode: 
    1) For each node in limb branch:
    a. extract the coordinates
    b. Create an array of the same lenght with name of node

    Possible coordinate types: 
    1) skeleton_data/skeleton
    2) endpoints, endpoint_upstream, endpoint_downstream
    3) mesh_center
    """
    if nodes is None:
        nodes =  nxu.limb_branch_nodes(G)

    coordinates_array = []
    node_array = []
    for node in nodes:
        
        try:
            coordinates = coordinates_from_coordinate_type(
                G,
                node,
                coordinate_type=coordinate_type,
            )
            
        except Exception as e:
            if suppress_errors:
                continue
            else:
                raise Exception(e)

        coordinates_array.append(coordinates)
        node_array += [node]*len(coordinates)

    if len(coordinates_array) > 0:
        coordinates_array = np.vstack(coordinates_array)

    node_array = np.array(node_array)

    if verbose:
        print(f"# of coordinates = {len(coordinates_array)}")
        
    return coordinates_array,node_array

def closest_node_to_coordinates(
    G,
    coordinates,
    node_coordinate_type = "skeleton",
    closest_idx_algorithm = "linalg",
    verbose = False,
    suppress_errors = False,
    ):
    """
    Purpose: map a coordinate(s) to the closest
    branch (using branches' endpoints, 
    upstream endpoints or all skeleton data, midpoint, etc..)

    Pseudocode: 
    1) Get a mapping of all of the coordinates to 
    the node name
    2) Do a kdtree mapping of coordinates to 
    list of coordinates (or iteratively do the linalg.norm)
    to find the closest idx
    3) Map closest idx to the label
    """

    graph_coords,graph_labels = nxu.coordinate_array_with_node_map_from_G(
        G,
        suppress_errors=suppress_errors,
        coordinate_type = node_coordinate_type,
        verbose = False,
    )

    idx = nu.closest_idx_for_each_coordinate(
        array=coordinates,
        array_for_idx=graph_coords,
        closest_idx_algorithm = closest_idx_algorithm,
        verbose = verbose,
    )

    closest_branch = graph_labels[idx]
    if verbose:
        print(f"closest_branch = {closest_branch}")
        
    return closest_branch
'''
def cluster_and_downstream_filter_coordinates_old(
    G,
    coordinates,
    radius = 5000,
    mapping_node_coordinate_type = "skeleton",
    filter_away_downstream_nodes = True,
    return_coordinates = True,
    output_coordinates_type = "endpoint_upstream",
    verbose = False,
    
    ):
    """
    Purpose: Given a set of edit coordinates want to filter them
    to non-redundant coordinates with the option of filtering
    away any coordinates that are downstream of others 
    (can return nodes or coordinates)

    Pseudocode: 
    1) Filter the coordinates with a radius thresholding
    -- how to keep only those most upstream
    2) Map the filtered coordinates to the closest branches
    3) (Optional) Filter away any downstream branches
    4) Return the branches are convert them to coordinates
    """


    #1) Filter the coordinates with a radius thresholding
    mean_coordinates = nu.mean_coordinates_from_radius_threshold_clustering(
            coordinates,
            radius = radius,
    )
    
    if verbose:
        print(f"# of filtered coordinates = {len(mean_coordinates)} (vs {(len(coordinates))} original)")
    
    #2) Map the filtered coordinates to the closest branches
    nodes = nxu.closest_node_to_coordinates(
        G,
        mean_coordinates,
        node_coordinate_type = mapping_node_coordinate_type,
        suppress_errors=suppress_errors
    )

    if verbose:
        print(f"Matching nodes = {nodes}")

    #3) (Optional) Filter away any downstream branches
    if filter_away_downstream_nodes:
        nodes = xu.filter_away_downstream_nodes(G,nodes)
        if verbose:
            print(f"Nodes after filtering away downstream = {nodes}")

    #4) Return the branches are convert them to coordinates
    if return_coordinates:
        return np.vstack([
            nxu.coordinates_from_coordinate_type(G,n,coordinate_type=output_coordinates_type)
            for n in nodes
        ])
    else:
        return nodes'''
    
def cluster_and_downstream_filter_coordinates(
    G,
    coordinates,
    radius = 5000,
    mapping_node_coordinate_type = "skeleton",
    filter_away_downstream_nodes = True,
    #return_coordinates = True,
    #output_coordinates_type = "endpoint_upstream",
    verbose = False,
    return_downstream_skeletal_length = False,
    suppress_errors=True,
    ):
    """
    Purpose: Given a set of edit coordinates want to filter them
    to non-redundant coordinates with the option of filtering
    away any coordinates that are downstream of others 
    (can return nodes or coordinates)

    Pseudocode: 
    1) Filter the coordinates with a radius thresholding
    -- how to keep only those most upstream
    2) Map the filtered coordinates to the closest branches
    3) (Optional) Filter away any downstream branches
    4) Return the branches are convert them to coordinates
    """


    #2) Map the filtered coordinates to the closest branches
    nodes = nxu.closest_node_to_coordinates(
        G,
        coordinates,
        node_coordinate_type = mapping_node_coordinate_type,
        suppress_errors=suppress_errors,
    )

    if verbose:
        print(f"Matching nodes = {nodes}")

    #3) (Optional) Filter away any downstream branches
    if filter_away_downstream_nodes:
        nodes = xu.filter_away_downstream_nodes(G,nodes)
        if verbose:
            print(f"Nodes after filtering away downstream = {nodes}")
            
    if return_downstream_skeletal_length:
        downstream_skeletal_lengths = np.array(
            [nxst.skeletal_length_downstream(G,n,include_self = True)
             for n in nodes]
        )
    
    # --- want to gather length information before filtering ----
    coordinates_upstream = np.vstack([
            nxu.coordinates_from_coordinate_type(G,n,coordinate_type="endpoint_upstream")
            for n in nodes
    ])
        
    #1) Filter the coordinates with a radius thresholding
    mean_coordinates,conn_comp = nu.mean_coordinates_from_radius_threshold_clustering(
            coordinates_upstream,
            radius = radius,
            return_clustering_idx = True,
    )
    
    if return_downstream_skeletal_length:
        downstream_skeletal_lengths = [np.sum(downstream_skeletal_lengths[k])
                                      for k in conn_comp]
    
    if verbose:
        print(f"# of filtered coordinates = {len(mean_coordinates)} (vs {(len(coordinates_upstream))} original)")
        
    if return_downstream_skeletal_length:
        return mean_coordinates,downstream_skeletal_lengths
    else:
        return mean_coordinates
    
def clean_G(
    G,
    verbose =False,
    debug_time = False,
    **kwargs):
    
    if debug_time:
        print(f"Total time for pulling down neuron graph obj: {time.time() - st}")
        st = time.time()
    return_G =  nxu.fix_flipped_skeleton(G,verbose=verbose)
    if debug_time:
        print(f"Total time for flipping sk: {time.time() - st}")
        st = time.time()
    return_G = nxu.fix_attribute(return_G,verbose = verbose)
    if debug_time:
        print(f"Total time for fixing attr: {time.time() - st}")
        st = time.time()
    return_G = nxu.fix_width_inf_nan(return_G, verbose = verbose)
    if debug_time:
        print(f"Total time for fixing width inf: {time.time() - st}")
        st = time.time()

    if remove_small_starter_branches:
        return_G = nxu.remove_small_starter_branches(return_G)

    if remove_small_endnode_branches:
        return_G = nxu.remove_small_endnode_branches(return_G)
    
    return return_G

def node_compartment_color_dict(
    G,
    compartment_color_dict=None):
    
    
    comp_colors = nxu.compartment_colors.copy()
    
    if compartment_color_dict is not None:
        comp_colors.update(compartment_color_dict)
    
    color_dict = dict([
        (n,comp_colors.get(G.nodes[n]["compartment"],comp_colors[None],))
         for n in G.nodes]
    )
    return color_dict

def print_compartment_colors():
    print(f"Compartment Colors")
    print(f"------------------")
    for k,v in nxu.compartment_colors.items():
        print(f"   {k} : {v}")

def soma_start_angle_max(G):
    """
    Purpose
    -------
    calculates the maximum value of the soma starting angle
    over all the nodes in the graph (application: use on limb graphs
    to compute their soma starting angle)
    """
    attr = "soma_start_angle"
    return np.max([
        G.nodes[n].get(attr,0)
        for n in G.nodes
    ])
    
def skeletal_length(G):
    """
    Purpose
    -------
    Compute the summed total skeletal length over all the nodes of a graph
    """
    attr = "skeletal_length"
    return np.sum([
        G.nodes[n].get(attr,0)
        for n in G.nodes
    ])
    
    
#     #4) Return the branches are convert them to coordinates
#     if return_coordinates:
#         return 
#     else:
#         return nodes

#--- from neuron_morphology_tools ---
from . import morphopy_utils as mpu
from . import neuron_nx_feature_processing as nxf  
from . import neuron_nx_stats as nxst
from . import neuron_skeleton_utils as nsku

swc_df_from_file = mpu.swc_df_from_file

#--- from datasci_tools ---
from datasci_tools import file_utils as fileu
from datasci_tools import ipyvolume_utils as ipvu
from datasci_tools import mesh_utils as meshu
from datasci_tools import networkx_utils as xu
from datasci_tools import numpy_utils as nu
from datasci_tools import pandas_utils as pu
from datasci_tools import system_utils as su
from datasci_tools.tqdm_utils import tqdm

from . import neuron_nx_utils as nxu