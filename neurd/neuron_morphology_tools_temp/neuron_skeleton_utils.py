
from datasci_tools import numpy_dep as np

def skeletal_length(skeleton):
    if len(skeleton) == 0:
        return 0
    
    if "np" not in str(type(skeleton)):
        skeleton = nxu.skeleton(skeleton,return_verts_edges=False)
    total_distance = np.sum(np.linalg.norm(skeleton[:,1,:] - skeleton[:,0,:],axis=1))
    return float(total_distance) 

def cirle_intersections(
    skeleton,
    center,
    radius,
    axes = None,
    verbose = False,
    plot = False,
    ):

    """
    Purpose: To find every coordinate where a skeleton 
    would intersect a circle of a certain radius
    
    Ex:
    import ipyvolume as ipv
    from neuron_morphology_tools from neurd import neuron_skeleton_utils as nsku

    segment_id = 864691134884743930
    split_index = 0

    G = hdju.graph_obj_from_auto_proof_stage(
        segment_id,
        split_index
    )

    soma_center = nxu.soma_center(G)

    skeleton = nxu.skeleton(
        G,
        include_soma=False,
        return_verts_edges=False,
        plot=True,

    )
    """
    if verbose:
        print(f"\n--- working on interseciton of circle with radius = {radius} ")
    
    orig_dim = skeleton.shape[-1]
    if axes is None:
        axes = np.arange(0,orig_dim).astype("int")

    skeleton_axes = skeleton[:,:,axes]
    soma_center_axes = center[axes]
    
    
    # finding the distances of all skeleton points away from the center
    skeleton_dists = np.linalg.norm(
        skeleton_axes-soma_center_axes,axis=2)
    
    # -- finding the edges that have one side one the inside of circle and one side outside
    sort_col_idx = np.argsort(skeleton_dists,axis=1)
    sort_row_idx = nu.repeat_vector_hstack(np.arange(len(skeleton_dists)))
    sort_dists_from_radius = skeleton_dists[sort_row_idx,sort_col_idx] - radius
    cross_map = (
        (sort_dists_from_radius[:,0] <= 0) & 
        (sort_dists_from_radius[:,1] > 0)
    )
    
    n_edges_cross = np.sum(cross_map) 
    if n_edges_cross == 0:
        if verbose:
            print(f"No intersecting edges")
        return np.array([]).reshape(-1,orig_dim)
    
    if verbose:
        print(f"# of intersecting edges = {n_edges_cross}")
        scatters_axes = skeleton_axes[cross_map].reshape(-1,len(axes))
        print(f"Distances of scatters = {np.linalg.norm(scatters_axes - soma_center_axes,axis=1)}")
        
        
    # --- finding the intersection points that are along the crossing edges
    sorted_dists_from_radius_abs = np.abs(sort_dists_from_radius[cross_map])
    sorted_weights = 1- sorted_dists_from_radius_abs/np.sum(sorted_dists_from_radius_abs,axis=1,keepdims=True)

    intersection_points = np.sum(
        skeleton[sort_row_idx,sort_col_idx][cross_map]*np.expand_dims(sorted_weights,axis=-1),
        axis=1)
    
    if verbose:
        print(f"# of intersecting edges = {n_edges_cross}")
        scatters_axes = intersection_points[:,axes]
        print(f"Distances of scatters = {np.linalg.norm(scatters_axes - soma_center_axes,axis=1)}")

    if plot:
        from neurd import neuron_visualizations as nviz
        print(f"Plotting the crossing edges (green) with intersections (red)")
        scatters = skeleton[cross_map].reshape(-1,orig_dim)
        nviz.plot_objects(
            main_skeleton=skeleton,
            scatters_colors=["red","green","black"],
            scatters=[intersection_points,scatters,center],
            scatter_size=1,
            axis_box_off = False,
        )
        
    
    return intersection_points

    
#--- from neuron_morphology_tools ---
from . import neuron_nx_utils as nxu

#--- from datasci_tools ---
from datasci_tools import ipyvolume_utils as ipvu
from datasci_tools import numpy_utils as nu

from . import neuron_skeleton_utils as nsku
