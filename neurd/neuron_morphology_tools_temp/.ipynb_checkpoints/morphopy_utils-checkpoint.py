'''

Purpose: Utils functions for using the 
morphopy module to compute morphology statistics
of a neuron

cell types were saved in https://raw.githubusercontent.com/berenslab/mini-atlas/master/data/m1_patchseq_meta_data.csv


'''
from itertools import combinations,permutations
from morphopy.neurontree import NeuronTree as nt
from morphopy.neurontree.utils import angle_between
from morphopy.neurontree.utils import get_standardized_swc
from scipy.stats import wasserstein_distance
import copy
import matplotlib.pyplot as plt
import networkx as nx
from datasci_tools import numpy_dep as np
import pandas as pd
import seaborn as sns



exc_features_berenslab = ['"apical" branch points' ,'"apical" height',
 '"apical" log1p number of outer bifurcations',
 '"apical" mean bifurcation distance' ,'"apical" robust height',
 '"apical" robust width' ,'"apical" std bifurcation distance',
 '"apical" total length', '"apical" width',
 'dendrite bifurcation standard deviation', 'dendrite branch points',
 'dendrite first bifurcation moment' ,'dendrite height',
 'dendrite log max tortuosity' ,'dendrite log min tortuosity',
 'dendrite max Euclidean dist' ,'dendrite max branch order',
 'dendrite max path distance to soma', 'dendrite max segment length',
 'dendrite mean neurite radius', 'dendrite median path angle',
 'dendrite min branch angle' ,'dendrite robust height',
 'dendrite robust width', 'dendrite tips' ,'dendrite total length',
 'dendrite width', 'dendrite x-bias', 'dendrite z-bias', 'normalized depth',
 'soma radius', 'stems' ,'stems exiting down' ,'stems exiting to the sides',
 'stems exiting up']  

inh_features_berenslab = [
 'EMD axon dendrite' ,'Log1p fraction of axon above dendrite',
 'Log1p fraction of axon below dendrite',
 'Log1p fraction of dendrite above axon',
 'Log1p fraction of dendrite below axon', 'axon above soma',
 'axon bifurcation standard deviation' ,'axon branch points',
 'axon first bifurcation moment' ,'axon height' ,'axon log min tortuosity',
 'axon max Euclidean dist', 'axon max branch order',
 'axon max path distance to soma' ,'axon max segment length',
 'axon mean neurite radius' ,'axon min branch angle', 'axon robust height',
 'axon robust width' ,'axon tips', 'axon total length' ,'axon width',
 'axon x-bias' ,'axon z-bias' ,'dendrite above soma',
 'dendrite bifurcation standard deviation', 'dendrite branch points',
 'dendrite first bifurcation moment', 'dendrite height',
 'dendrite log max tortuosity', 'dendrite log median tortuosity',
 'dendrite log min tortuosity', 'dendrite max Euclidean dist',
 'dendrite max branch angle', 'dendrite max branch order',
 'dendrite max path distance to soma' ,'dendrite max segment length',
 'dendrite mean neurite radius', 'dendrite min branch angle',
 'dendrite robust height' ,'dendrite robust width', 'dendrite tips',
 'dendrite total length', 'dendrite width' ,'dendrite x-bias',
 'dendrite z-bias' ,'mean initial segment radius', 'normalized depth',
 'soma radius', 'stems', 'stems exiting down' ,'stems exiting to the sides',
 'stems exiting up'] 
 
def swc_df_from_file(filepath):
    """
    Purpose: To read in a swc file as a dataframe
    """
    swc = pd.read_csv(filepath, delim_whitespace=True, comment='#',
       names=['n', 'type', 'x', 'y', 'z', 'radius', 'parent'], index_col=False, header=None)
    
    # check that header not already there
    top_dict = swc.iloc[0,:].to_dict()
    if np.any([isinstance(k,str) for k in top_dict.values()]):
        swc = swc.iloc[1:,:]
    
    #swc["n"] = swc["n"].astype('int')
    swc = swc.apply(pd.to_numeric)
    return swc



def ntree_obj_from_swc(
    swc=None,
    filepath=None,
    plot = False):
    if swc is None:
        swc = mpu.swc_df_from_file(filepath)
    N = nt.NeuronTree(swc=swc)
    if plot:
        plot_ntree(N)
    return N

def swc_rotated_resampled(
    swc,
    swc_file = None,
    resample = False,
    flip_z = True,
    verbose = False,
    return_ntree = False
    ):



    """
    Purpose: To rotate and maybe downsample
    so the apical is pointing up
    """

    if swc is None:
        swc = mpu.swc_df_from_file(swc_file)

    # switch y and z since y corresponds to cortical depth
    swc = swc.rename(columns={'y': 'z', 'z': 'y'})
    # soma center for standardization
    rotated_swc = get_standardized_swc(swc, pca_rot=False)

    if flip_z: 
        rotated_swc["z"] = -rotated_swc["z"]

    N = nt.NeuronTree(swc=rotated_swc)
    # Resample neuron at distance 1 micron
    if resample:
        N = N.resample_tree(dist=1)
    # Smooth neurites in y direction
    smooth = False
    if smooth:
        N = N.smooth_neurites(dim=1, window_size=21)

    if return_ntree:
        return N
    else:
        return N.to_swc()
    
    
def plot_ntree(
    N,
    figsize = (10,5),
    xlim = None,
    ylim=None,
    zlim = None):
    
    """
    Ex: 
    mpu.plot_ntree(ntree_obj,ylim = [-300,100])
    """
    
    fig = plt.figure(figsize=figsize)
    ax1 = plt.subplot(121)
    ax2= plt.subplot(122)
    
    if xlim is not None:
        ax1.set_xlim(xlim)
        
    if ylim is not None:
        ax2.set_xlim(ylim)
        
    if zlim is not None:
        ax1.set_ylim(zlim)
        ax2.set_ylim(zlim)
        
    N.draw_2D(fig, ax=ax1, projection='xz')    
    N.draw_2D(fig, ax=ax2, projection='yz')
    
    
# ---------- Computing the morphology df -----------

def morphometrics(
    N=None,
    swc = None,
    depth=100,
    thickness=1000,
    cell_id = None,
    stats_to_remove=(
    'dendrite z-profile',
    'axon z-profile',
    'axon soma-centered z-profile',
    'dendrite soma-centered z-profile'
        )
    ):
    

    def get_perc_above_below_overlap(profile_a, profile_b):
    
        profile_a = np.hstack((np.array([0]),profile_a,np.array([0])))
        profile_b = np.hstack((np.array([0]),profile_b,np.array([0])))
        a = np.where(profile_a > 0)[0]
        b = np.where(profile_b > 0)[0]

        perc_a_above_b = np.sum(profile_a[:b[0]])/np.sum(profile_a)
        perc_a_below_b = np.sum(profile_a[b[-1]+1:])/np.sum(profile_a)
        perc_a_overlap_b  = 1 - perc_a_above_b - perc_a_below_b

        return (perc_a_above_b,perc_a_below_b,perc_a_overlap_b)

    def get_longest_neurite(R):

        r = R.get_root()

        stem_ids = [s[1] for s in R.edges() if s[0] == r]

        neurite_lengths = dict(zip(stem_ids,[0]*len(stem_ids)))
        neurite_paths = dict()
        G = R.get_graph()

        # get the path length and the path of each neurite extending from the soma
        for t in R.get_tips():
            path_length = nx.dijkstra_path_length(G,r,t,weight='path_length')
            path = nx.dijkstra_path(G,r,t)

            stem_ix = path[1]
            neurite_lengths[stem_ix] += path_length 
            if stem_ix in neurite_paths.keys():
                neurite_paths[stem_ix] += path
            else:
                neurite_paths[stem_ix] = path

        keys = list(neurite_lengths.keys())
        values = list(neurite_lengths.values())
        argix = np.argmax(values)

        #get subgraph with all nodes
        subgraph = nx.subgraph(G,set(neurite_paths[keys[argix]]))

        return nt.NeuronTree(graph=subgraph)
    
    if N is None:
        #print(f"Creating n object")
        N = mpu.ntree_obj_from_swc(swc)
    
    z = dict()
    #depth = item['Soma depth (µm)']
    #thickness = item['Cortical thickness (µm)']
    z['normalized depth'] = depth/thickness
   
    for part in ['axon', 'dendrite']:

        if part == 'axon':
            T = N.get_axonal_tree()
        elif part == 'dendrite':
            T = N.get_dendritic_tree() 


        if len(T.nodes()) > 5:

            z[part+ ' branch points'] = T.get_branchpoints().size
            extend = T.get_extent()

            z[part + ' width'] = extend[0]
            z[part + ' depth'] = extend[1]
            z[part + ' height'] = extend[2]

            robust_extend = T.get_extent(robust=True)
            z[part + ' robust width'] = robust_extend[0]
            z[part + ' robust depth'] = robust_extend[1]
            z[part + ' robust height'] = robust_extend[2]

            pos = np.array(list(T.get_node_attributes('pos').values()))
            bias = np.max(pos,axis=0) + np.min(pos, axis=0)

            z[part + ' x-bias'] = np.abs(bias[0])
            z[part + ' z-bias'] = bias[2]

            z[part + ' tips'] = T.get_tips().size

            z[part + ' total length'] = np.sum(list(T.get_edge_attributes('euclidean_dist').values()))

            z[part + ' max path distance to soma'] = np.max(list(T.get_path_length().values()))
            z[part + ' max branch order'] = np.max(list(T.get_branch_order().values()))

            path_angles = []
            for p1 in T.get_path_angles().items():
                if p1:
                    path_angles += [p1[1]]

            z[part + ' max path angle'] = np.percentile(path_angles,99.5)
            z[part + ' median path angle'] = np.median(path_angles)

            R = T.get_topological_minor()
            
            # maximal segment path length
            z[part + ' max segment length'] = np.max(list(R.get_segment_length().values()))

            tortuosity = [e[2]['path_length'] / e[2]['euclidean_dist'] for e in R.edges(data=True)]

            z[part + ' log max tortuosity'] = np.log(np.percentile(tortuosity,99.5))
            z[part + ' log min tortuosity'] = np.log(np.min(tortuosity))
            z[part + ' log median tortuosity'] = np.log(np.median(tortuosity))

            branch_angles = list(R.get_branch_angles().values())
            if branch_angles:
                z[part + ' max branch angle'] = np.max(branch_angles)
                z[part + ' min branch angle'] = np.min(branch_angles)
                z[part + ' mean branch angle'] = np.mean(branch_angles)
            else:
                z[part + ' max branch angle'] = np.nan
                z[part + ' min branch angle'] = np.nan
                z[part + ' mean branch angle'] = np.nan


            # z-profiles
            resampled_nodes = T.resample_nodes(d=1)
            z_profile, _ = np.histogram(-1*((resampled_nodes[:,2] - depth)/thickness), 
                                            bins=20, range=[0,1], density=True)
            soma_centered_z_profile, _ = np.histogram(((resampled_nodes[:,2])/thickness),
                                                      bins=81, range=[-1,1], density=True)
            
            z[part + ' z-profile'] = z_profile
            z[part + ' soma-centered z-profile'] = [soma_centered_z_profile]

            z[part + ' above soma'] = np.sum(resampled_nodes[:,2]>0)/resampled_nodes[:,2].shape[0]

            radii = R.get_node_attributes('radius')
            edges = R.edges()
            r = R.get_root()

            z['soma radius'] = radii[int(r)]

            if part == 'axon':

                # get thickness of initial segments
                node_ids = [e[1] for e in edges if (e[0] == r)]
                initial_segments_radius = [radii[int(n)] for n in node_ids]
                z['mean initial segment radius'] = np.mean(initial_segments_radius)

            # get mean neurite thickness
            radii.pop(r)  # remove the soma as it is usually the thickest
            z[part + ' mean neurite radius'] = np.mean(list(radii.values()))

        
            ec = []
            G = R.get_graph()
            for p in np.concatenate((R.get_tips(),R.get_branchpoints())):
                ec.append(np.sqrt(np.sum(G.nodes[p]['pos']**2)))

            z[part + ' max Euclidean dist'] = np.max(ec)


            # get bifurcation moments
            branch_point_positions = [R.get_graph().nodes[k]['pos'] for k in R.get_branchpoints()]
            if branch_point_positions:
                z[part + ' first bifurcation moment'] = np.mean(branch_point_positions, axis=0)[2]
                z[part + ' bifurcation standard deviation'] = np.std(branch_point_positions, axis=0)[2]


            if part == 'dendrite':

                stems = [R.get_graph().nodes[k[1]]['pos'] for k in R.edges(R.get_root())]
                # only calculated in xz plane in degree
                stem_exit_angles = np.array([angle_between([0,-1],s[[0,2]])/np.pi*180 for s in stems])

                # stems
                z['stems'] = len(stems)

                # stem exit histogram
                z['stems exiting up'] = np.sum(stem_exit_angles < 45 )/len(stems)
                z['stems exiting down'] = np.sum(stem_exit_angles > 135 )/len(stems)
                z['stems exiting to the sides']= np.sum((stem_exit_angles>=45) & (stem_exit_angles <= 135))/len(stems)

                # now get morphometrics for longest dendrite
                L = get_longest_neurite(R)
                
                # get branch point positions for apical
                bpp_L = [L.get_graph().nodes[k]['pos'] for k in L.get_branchpoints()]
                
                path_length = nx.single_source_dijkstra_path_length(L.get_graph(), source=L.get_root(),weight='path_length')
                # get furthes node and the line to it from soma. Furthest in terms of path length.

                G = L.get_graph()
                tips = L.get_tips()
                pl = [path_length[t] for t in tips]
                

                farthest_tip = tips[np.argmax(pl)]
                farthest_tip_pos = G.nodes[farthest_tip]['pos']
                max_pl = np.max(pl)

                proj_bp = [np.dot(bpp,farthest_tip_pos)/np.linalg.norm(farthest_tip_pos)/max_pl
                           for bpp in bpp_L]

                if proj_bp:
                    # mean bifurcation distance
                    z['"apical" mean bifurcation distance'] = np.mean(proj_bp)

                    # std bifurcation distance
                    z['"apical" std bifurcation distance'] = np.std(proj_bp)

    
                # number of outer bifurcations
                ec = []
                for t in tips:
                    ec.append(np.sqrt(np.sum(G.nodes[t]['pos']**2)))

                max_ec = np.max(ec)

                outer_bifurcations = 0
                for bp in L.get_branchpoints():
                    if np.sqrt(np.sum(G.nodes[bp]['pos']**2)) > max_ec/2:
                        outer_bifurcations += 1
                        
                z['"apical" log1p number of outer bifurcations'] = np.log(outer_bifurcations + 1) 
                
                z['"apical" height'] = L.get_extent()[2]
                z['"apical" width'] = L.get_extent()[0]
                
                z['"apical" robust height'] = L.get_extent(robust=True)[2]
                z['"apical" robust width'] = L.get_extent(robust=True)[0]
                
                z['"apical" total length'] = np.sum(list(L.get_edge_attributes('path_length').values()))
                z['"apical" branch points'] = len(L.get_branchpoints())
        else:
            print("No %s recovered"%part)

        # get the overlap and earth mover's distance here
        # overlap
        for key1, key2 in permutations(['axon z-profile', 'dendrite z-profile'],2):
            try:
                profile_a = z[key1]
                profile_b = z[key2]

                above, below, overlap = get_perc_above_below_overlap(profile_a,profile_b)
                z['Log1p fraction of %s above %s'%(key1.split(" ")[0], key2.split(" ")[0])]= np.log(1+above)
                z['Log1p fraction of %s below %s'%(key1.split(" ")[0], key2.split(" ")[0])]= np.log(1+below)
               
            except KeyError:
                continue


        # earth mover's distance
        for key1, key2 in combinations(['axon z-profile', 'dendrite z-profile'],2):
            try:
                profile_a = z[key1]
                profile_b = z[key2]

                z['EMD %s %s'%(key1.split(" ")[0], key2.split(" ")[0])] = wasserstein_distance(profile_a,profile_b)
            except KeyError:
                continue


    # make the arrays a list
    for key in ['axon z-profile', 'dendrite z-profile']:
        try:
            z[key] = [z[key]]
        except KeyError:
            continue
            
    if cell_id is not None:
        z['cell id'] = file_name
        
    morphometry_data = pd.DataFrame(z)
    
    if stats_to_remove is not None:
        morphometry_data = pu.delete_columns(
            morphometry_data,
            list(stats_to_remove))
    
    return morphometry_data


#--- from datasci_tools ---
from datasci_tools import pandas_utils as pu

from . import morphopy_utils as mpu