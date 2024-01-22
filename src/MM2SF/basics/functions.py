# Import the required libraries and modules
import numpy as np
import scipy
from scipy.integrate import quad
from scipy.stats import norm
from itertools import combinations
import sys as sys
from scipy.spatial.distance import cdist
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy import stats
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
import matplotlib.patches as mpatches
import os
from scipy.integrate import simps

def read_xyz_trajectory_wstep(filename, sampling_step=1):
    """
    Read a trajectory from an XYZ file and return a list of geometries with a certain sampling frequency.
    Parameters:
        filename (str): The name of the XYZ file.
    Returns:
        geometries (list): A list of geometries. Each geometry is represented as a dictionary
                           with 'symbols' and 'positions' keys, where 'symbols' is a list of
                           atomic symbols and 'positions' is a numpy array of atomic positions
                           with shape (n_atoms, 3).
    """
    geometries = []
    test=0
    with open(filename, 'r') as f:
        counter = 0
        while True:
            try:
                # Read the number of atoms
                n_atoms = int(f.readline())
                # Skip the comment line
                comment = f.readline().strip()
                # Read the atomic symbols and positions
                symbols = []
                positions = []
                for i in range(n_atoms):
                    line = f.readline().split()
                    symbol = line[0]
                    x, y, z = map(float, line[1:])
                    symbols.append(symbol)
                    positions.append([x, y, z])
                positions = np.array(positions)
                geometry = {'symbols': symbols, 'positions': positions}
                # Append the geometry to the list based on the sampling step
                if counter % sampling_step == 0:
                    geometries.append(geometry)
                    test +=1
                counter += 1
            except ValueError:
                # End of file reached
                break
    return geometries

def compute_distance_distribution(geometries):
    """
    Compute the distribution of distances for each element in the set with respect to all the remaining elements.
    Parameters:
        geometries (list): A list of geometries. Each geometry is represented as a dictionary
                           with 'symbols' and 'positions' keys, where 'symbols' is a list of
                           atomic symbols and 'positions' is a numpy array of atomic positions
                           with shape (n_atoms, 3).
    Returns:
        dist_dists (dict): A dictionary containing the distance distributions for each element.
                           The keys are the atomic symbols, and the values are numpy arrays
                           containing the distance distributions.
    """
    # Create a list of all unique atomic symbols
    symbols = np.unique([geom['symbols'] for geom in geometries])
    # Compute the pairwise distances between atoms in each geometry
    dists = []
    for geom in geometries:
        positions = geom['positions']
        dists.append(cdist(positions, positions))
    dists = np.array(dists)
    # Initialize the distance distribution dictionary
    distance_distribution = {}
    # Loop over each element in the set of symbols
    for j in np.arange(0,len(symbols)):
        for k in np.arange(j,len(symbols)):
            symbol1=str(symbols[j])
            symbol2=str(symbols[k])
            symbol_dists = []
            # Loop over each geometry and compute the distances between atoms of the current symbols
            for i, geom in enumerate(geometries):
                positions = geom['positions']
                symbols_array = np.array(geom['symbols'])
                mask1 = (symbols_array == symbol1)
                mask2 = (symbols_array == symbol2)
                positions1 = positions[mask1]
                positions2 = positions[mask2]
                dists12 = dists[i][mask1][:, mask2].flatten()
                symbol_dists.append(dists12)
            distance_distribution.setdefault(symbol1, {})[symbol2] = np.concatenate(symbol_dists)
            distance_distribution.setdefault(symbol2, {})[symbol1] = np.concatenate(symbol_dists)
    # Create a sorted set for the symbols
    symbols=list(symbols)
    symbols=set(symbols)
    symbols=sorted(symbols)
    return symbols,distance_distribution

def extract_distribution(values,bins):
    """
    Parses the radial distribution of values into specified bins (2D data).

    Parameters:
    - values (array-like): The input values for which the distribution is to be calculated.
    - bins (int): it defines the number of equal-width bins in the range.
    
    Returns:
    - bin_centers (ndarray): The central values of each bin.
    - counts (ndarray): The counts or frequencies of values in each bin.
    """
    counts, bin_edges = np.histogram(values, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, counts

def smooth_data(xy,bw):
    """
    Smooths input data using kernel density estimation (KDE) and resamples from the estimated distribution.

    Parameters:
    - xy (array-like): Input data points as a 2D array-like structure.
    - bw (float, optional): The bandwidth for the kernel density estimation.

    Returns:
    - xy_kde (ndarray): Smoothed data points obtained from kernel density estimation.
    - samples (ndarray): Resampled data points from the estimated distribution.

    """
    x=xy[:,0]
    y=xy[:,1]
    # Recreate the data starting from the histogram
    array=[]
    for i in range(x.shape[0]):
        for j in range(int(y[i])):
            array.append(x[i])
    array=np.array(array)
    kde = stats.gaussian_kde(array,bw_method=bw)
    x_kde=x.copy()
    y_kde = kde(x_kde)
    y_kde -=y_kde.min()
    y_kde *=(y.max()-y.min())
    x_kde= np.reshape(x_kde, (-1, 1))
    y_kde= np.reshape(y_kde, (-1, 1))
    xy_kde=np.hstack((x_kde, y_kde))
    # Generate a data pool with 100000 samples
    samples = kde.resample(100000)
    return xy_kde,samples

def normalize(xy):
    """
    Normalizes the y-values of an array of 2D-data.

    Parameters:
    - xy (array-like): Input data points as a 2D array-like structure.

    Returns:
    - xy_norm (ndarray): Normalized data points, where the y-values are scaled between 0 and 1.

    """
    x=xy[:,0]
    y=xy[:,1]
    min_dat=min(y)
    max_dat=max(y)
    norm_y=[]
    for val in y:
        norm_y.append((val-min_dat)/(max_dat-min_dat))
    norm_y=np.array(norm_y)
    norm_y= np.reshape(norm_y, (-1, 1))
    norm_x= np.reshape(x, (-1, 1))
    xy_norm=np.hstack((norm_x, norm_y))
    return xy_norm

def norm_3ddata(data):
    """
    Normalizes 3D data corresponding to the angular distribution.

    Parameters:
    - data (ndarray): Input 3D data array.

    Returns:
    - data_norm (ndarray): Normalized 3D data array after subtracting the mean and dividing by the standard deviation.
    - data_mean (ndarray): Mean values of each dimension in the original data.
    - data_std (ndarray): Standard deviation values of each dimension in the original data.
    """

    data_mean=np.mean(data,axis=0)
    data_std=np.std(data,axis=0)
    data_norm = (data - data_mean) / data_std

    return data_norm,data_mean,data_std

def fcutoff(r,rcut,cut_type='hard'):
    """
    Cutoff function kernel.

    Parameters:
    - r (float): Distance from a reference point.
    - rcut (float): Cutoff radius for the function.
    - cut_type (str, optional): Type of approximation used for truncating the neighborhood.
                                'soft' applies a smooth cutoff function, 'hard' simply
                                cuts off values above the cutoff radius.
    Returns:
    - rc (float): Cutoff function value based on the specified type and distance.
    """

    if cut_type == "soft":
       if (r <= rcut):
          rc=0.5*(np.cos(np.pi*r/rcut)+1)
       else:
          rc=0.0
    elif cut_type == "hard":
       if (r <= rcut):
          rc=1.0
       else:
          rc=0.0
    return rc

def apply_cut(xy,rcut,cut_type='hard'):
    """
    Applies the cutoff function to a distribution.

    Parameters:
    - xy (array-like): Input data points as a 2D array-like structure.
    - rcut (float): Cutoff radius for the cutoff function.
    - cut_type (str, optional): Type of approximation used for truncating the neighborhood.
                                'soft' applies a smooth cutoff function, 'hard' simply
                                cuts off values above the cutoff radius.
    Returns:
    - data (ndarray): Data points after applying the cutoff function based on the specified type and radius.

    Note: The cutoff type ('cut_type') is optional and defaults to 'hard'.
    """

    x_cut=[]
    y_cut=[]
    for i in xy:
        r=i[0]
        val=i[1]
        x_cut.append(r)
        y_cut.append(val*fcutoff(r,rcut,cut_type=cut_type))
    x_cut=np.reshape(np.array(x_cut),(-1,1))
    y_cut=np.reshape(np.array(y_cut),(-1,1))
    data=np.hstack((x_cut, y_cut))
    return data

def xy2raw(xy):
    """
    Recreates raw data from an already computed histogram.

    Parameters:
    - xy (array-like): Input data points in histogram format as a 2D array-like structure,
                      where the first column represents x-values, and the second column represents
                      the corresponding histogram counts or frequencies.
    Returns:
    - array (ndarray): Recreated raw data based on the histogram information.
    """

    x=xy[:,0]
    y=xy[:,1]
    array=[]
    for i in range(x.shape[0]):
        for j in range(int(100*y[i])):
            array.append(x[i])
    array=np.array(array)
    return array 

def find_opt_components(X,nmax=10,max_iter=10000):
    """
    Finds the ideal number of components of a given distribution using the Bayesian Information Criterion (BIC).

    Parameters:
    - X (ndarray): Input data as a 1D array.
    - nmax (int, optional): Maximum number of components to consider. Defaults to 10.
    - max_iter (int, optional): Maximum number of iterations for the Gaussian Mixture Model (GMM) fitting. Defaults to 10000.

    Returns:
    - best_n_components (int): The optimal number of components determined by the BIC criteria.

    """

    X=X.reshape(-1,1) 
    n_components_range = range(1, nmax)
    bic_scores = []
    for n_components in n_components_range:
        mix = GaussianMixture(n_components=n_components, random_state=1, max_iter=max_iter).fit(X)
        bic_scores.append(mix.bic(X))
    # Choose the number of components with the lowest BIC
    best_n_components = n_components_range[np.argmin(bic_scores)]

    return best_n_components

def find_gmm_params(dist,n_comp,maxfev,nbins):
    """
    Finds the parameters of a Gaussian Mixture Model (GMM) with a given number of components.

    Parameters:
    - dist (ndarray): Input distribution data as a 1D array.
    - n_comp (int): Number of components in the GMM.
    - maxfev (int): Maximum number of iterations for the GMM fitting.
    - nbins (int): Number of bins for creating a grid for visualization.

    Returns:
    - mix (GaussianMixture): Fitted Gaussian Mixture Model.
    - mu (ndarray): Means of the components in the GMM.
    - sigma2 (ndarray): Covariance matrix of the components in the GMM.
    - pi (ndarray): Weights of the components in the GMM.
    - sum_gau (ndarray): Sum of all Gaussian components over a grid.
    - grid (ndarray): Grid points for visualization.
    """

    y=dist
    y=y.reshape(-1, 1)
    mix = GaussianMixture(n_components=n_comp, random_state=1, max_iter=maxfev).fit(y)
    pi, mu, sigma2 = mix.weights_.flatten(), mix.means_.flatten(), mix.covariances_.flatten()
    grid = np.arange(np.min(dist), np.max(dist), 1/nbins)
    # Calculate the sum of all Gaussians
    sum_gau = np.zeros_like(grid)
    for i in range(n_comp):
        sum_gau += pi[i] * norm.pdf(grid, loc=mu[i], scale=np.sqrt(sigma2[i]))
    return mix,mu,sigma2,pi,sum_gau,grid

def plot_angular_trios(data,elem,pair,rcut,nbins,cmap_corr='jet',cmap_polar='bwr',ps=0.5,file=None):

    """
    Generates a comprehensive plot to analyze the distribution of angles and distances formed by trios of atoms.

    Parameters:
    - data (list): List of tuples containing angle, distance I-J, and distance I-K data.
    - elem (str): Symbol of the central element of the trio (e.g., 'C').
    - pair (tuple): Pair of atoms forming the trio (e.g., ('O', 'H') to form C-O-H).
    - rcut (float): Cutoff radius.
    - nbins (int): Number of bins to use in histograms.
    - cmap_corr (str, optional): Color map for the distance correlation plot. Defaults to 'jet'.
    - cmap_polar (str, optional): Color map for the polar plot. Defaults to 'bwr'.
    - ps (float, optional): Size of the marker. Defaults to 0.5.
    - file (str or None, optional): If provided, saves the plot to the specified file; if None, displays the plot interactively.

    Returns:
    - None

    The function creates a multi-panel plot consisting of:
    1) A polar plot with angular distributions and distances I-J.
    2) A histogram of distances I-J.
    3) A histogram of distances I-K.
    4) A distance correlation plot (I-J vs. I-K) colored based on the angle.

    Note: The function is designed for analyzing trio distributions in molecular systems.
    """

    # Set the main title 
    title=str(elem)+str(pair)+" angular distribution"

    # Retrieve the data
    angles = []
    dist_ij = []
    dist_ik = []
    dist_ik_color = []
    angle, d_ij, d_ik = zip(*data)
    angles.extend(angle)
    dist_ij.extend(d_ij)
    dist_ik.extend(d_ik)
    angles = np.array(angles)
    dist_ij = np.array(dist_ij)
    dist_ik = np.array(dist_ik)

    # Create boolean mask for filtering the data based on the cutoff
    mask = (dist_ij <= rcut) & (dist_ik <= rcut)
    angles_filtered = angles[mask]
    dist_ij_filtered = dist_ij[mask]
    dist_ik_filtered = dist_ik[mask]

    # Create a color variable for the polar plot
    dist_ratio_color = np.array(dist_ik_filtered/dist_ij_filtered)

    # Set the plots
    fig = plt.figure(constrained_layout=True,figsize=(12,12))
    fig.suptitle(title)
    gs = fig.add_gridspec(10, 10)
    fig_polar  = fig.add_subplot(gs[0:4, 0:4],projection='polar') # polar plot
    fig_main = fig.add_subplot(gs[4:, 4:])                        # distance correlation plot
    fig_ij   = fig.add_subplot(gs[:4, 4:],sharex=fig_main)        # IJ distance histogram
    fig_ik = fig.add_subplot(gs[4:,:4],sharey=fig_main)           # IK distance histogram

    # CORRELATION PLOT
    fig_main.set_xlim(0, rcut)
    fig_main.set_ylim(0, rcut)
    fig_main.set_xlabel('Distance I-J('+str(elem)+"-"+str(pair[0])+")")
    fig_main.set_ylabel('Distance I-K('+str(elem)+"-"+str(pair[1])+")")
    # create a 2D histogram of the data
    H, xedges, yedges = np.histogram2d(dist_ij_filtered, dist_ik_filtered, bins=(nbins, nbins))
    mesh = fig_main.pcolormesh(xedges, yedges, H.T, cmap='binary',alpha=0.5)
    # Add the scatter plot and color the points according their angle
    sc = fig_main.scatter(dist_ij_filtered, dist_ik_filtered, c=angles_filtered, cmap=cmap_corr, s=ps, alpha=1)
    cb = plt.colorbar(sc)
    cb.set_label(str(pair[0])+"-"+str(elem)+"-"+str(pair[1])+" Angle (º)")
    cb.mappable.set_clim(vmin=0,vmax=180)

    # IJ AND IK HISTOGRAMS
    fig_ij.hist(dist_ij_filtered, bins=nbins, color='grey', alpha=1, orientation='vertical')
    fig_ik.hist(dist_ik_filtered, bins=nbins, color='grey', alpha=1, orientation='horizontal')
    fig_ik.invert_xaxis()
    fig_ik.set_xlabel('Frequency (au)')
    fig_ij.set_ylabel('Frequency (au)')

    # POLAR PLOT
    # Radial axis
    fig_polar.set_rlim([0, rcut])
    rticks =[]
    for i in np.arange(0,int(rcut)):
         rticks.append(i)
    fig_polar.set_rticks(rticks)
    fig_polar.set_rlabel_position(135)
    # Angular axis
    fig_polar.set_thetamin(0)
    fig_polar.set_thetamax(180)
    fig_polar.set_xticks(np.linspace(0, np.pi, 9, endpoint=True))
    fig_polar.set_xticklabels(['0','22.5','45', '67.5','90','112.5', '135','157.5', '180'])
    # Plot the data
    scp = fig_polar.scatter(np.radians(angles_filtered), dist_ij_filtered, c=dist_ratio_color, cmap=cmap_polar, s=ps, alpha=1)
    cbp = plt.colorbar(scp, ax=fig_polar, location='right', pad=0.05)
    cbp.set_label('Distance '+str(elem)+"-"+str(pair[1])+"/Distance "+str(elem)+"-"+str(pair[0]))
    cbp.mappable.set_clim(vmin=0,vmax=rcut)
    # Add labels and title
    fig_polar.set_xlabel('Angle (º)')
    fig_polar.set_ylabel('Distance '+str(elem)+"-"+str(pair[0]), labelpad=25)

    # Show the plot
    if file is None: 
       plt.show()
    else:
       plt.savefig(file)
    plt.close()

    return None

def compute_angles_wijk(geometries):
    """
    Computes and gathers geometric features of atomic trios from XYZ trajectory geometries.

    Parameters:
    - geometries (list): List of dictionaries containing 'positions' and 'symbols' keys for each geometry.

    Returns:
    - symbols_set (list): List of unique element symbols present in the geometries.
    - geom_feat (dict): Dictionary containing geometric features for each atomic trio.
                   The structure is geom_feat[element][pair] = [(angle, dist_cen_nei1, dist_cen_nei2), ...]

    The function takes a list of geometries extracted from an XYZ trajectory file.
    For each unique atomic trio, it computes the angle and distances between atoms,
    centered around each atom in the trio. The results are stored in a dictionary.

    Note: The function uses handles all possible descriptions for each trio (it accounts
    for the 3 possible descriptions of a triangle, while considering the scenario where
    J and K have the same chemical identity.
    """
    geom_feat = {}
    count=0
    symbols_set=[]
    for geometry in geometries:
        positions = geometry['positions']
        symbols = geometry['symbols']
        natoms=len(symbols)
        print("   # running the parse for geometry ", count+1)
        for i in range(0,natoms-2):  
            for j in range(i+1,natoms-1): 
                    for k in range(j+1,natoms):
                            TPset=[]
                            TPset.append(i)
                            TPset.append(j)
                            TPset.append(k)
                            # Pivoting over all three atoms
                            for p1 in TPset:
                                atom_cen=positions[p1]
                                nset=TPset.copy()
                                nset.remove(p1)
                                p2=nset[0]
                                p3=nset[1]
                                atom_nei1=positions[p2]
                                atom_nei2=positions[p3]
                                # Compute neighboring distances
                                dist_cen_nei1=np.linalg.norm(atom_cen - atom_nei1)
                                dist_cen_nei2=np.linalg.norm(atom_cen - atom_nei2)
                                # Compute the angle scented at the reference atom
                                v1 = atom_nei1 - atom_cen
                                v2 = atom_nei2 - atom_cen
                                angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
                                angle = np.degrees(angle)
                                pair = symbols[p2] + symbols[p3] if symbols[p2] <= symbols[p3] else symbols[p3] + symbols[p2]
                                # Set the label for the neighboring atomic pair so that the id is always defined in the same way.
                                if symbols[p1] not in geom_feat:
                                    geom_feat[symbols[p1]] = {}
                                    symbols_set.append(symbols[p1])
                                if pair not in geom_feat[symbols[p1]]:
                                    geom_feat[symbols[p1]][pair] = []
                                # We store the geometric features in the dictionary, taking care to account for both 
                                # ij-ik and ik-ij orderings to align with the defined order of the atom pair "pair".
                                # This approach ensures that we capture all possible descriptions of atomic trios.
                                if symbols[p2] == pair[0]:
                                    geom_feat[symbols[p1]][pair].append((angle, dist_cen_nei1, dist_cen_nei2))
                                if symbols[p3] == pair[0]:
                                    geom_feat[symbols[p1]][pair].append((angle, dist_cen_nei2, dist_cen_nei1))
        count +=1
    # We create a set of element labels and sort it alphabetically to ensure that we are working with the same set
    # for both radial and angular computations, as they share the same input.type file.
    symbols_set=set(symbols_set)
    symbols_set=sorted(symbols_set)
    return symbols_set,geom_feat

def find_opt_components3D(X,nmax=10,max_iter=10000,cv_type="full",crit="bicmin",percbic=60,percdiff=30,file=None):
    """
    Finds the optimal number of components for a Gaussian Mixture Model (GMM) using BIC scores (for angular distributions).

    Parameters:
    - X (ndarray): Input data for the GMM as a 2D array.
    - nmax (int, optional): Maximum number of components to consider. Defaults to 10.
    - max_iter (int, optional): Maximum number of iterations for the GMM fitting. Defaults to 10000.
    - cv_type (str, optional): Type of covariance to use in the GMM ('full', 'tied', 'diag', 'spherical'). Defaults to 'full'.
    - crit (str, optional): Criterion used to select the best number of components ('bicmin' or 'bicconv'). Defaults to 'bicmin'.
    - percbic (int, optional): Percentile to screen in terms of the BIC score metrics. Defaults to 60.
    - percdiff (int, optional): Percentile to screen in terms of the finite differences in the BIC score metrics. Defaults to 30.
    - file (str or None, optional): If provided, saves the BIC score plot to the specified file; if None, displays the plot interactively.

    Returns:
    - best_n_components (int): The optimal number of components determined by the specified criterion.

    The function fits Gaussian Mixture Models (GMMs) with varying numbers of components and evaluates the models
    using the Bayesian Information Criterion (BIC). The optimal number of components is determined based on the chosen
    criterion ('bicmin' or 'bicconv').

    Note: The function uses the scikit-learn GaussianMixture class.
    """

    # Run the GMMs for an increasing number of components (clusters)
    n_components_range = range(1, nmax+1)
    bic_scores = []
    for n_components in n_components_range:
        print("  - Running the parse for {:4d} components.".format(n_components))
        mix = GaussianMixture(n_components=n_components, random_state=1, max_iter=max_iter,covariance_type=cv_type).fit(X)
        bic_scores.append(mix.bic(X))
    xdata = np.array(list(n_components_range))
    ydata = np.array(bic_scores)
    # Determine the optimum number of clusters
    if (crit == "bicmin"):
       # Choose the number of components with the lowest BIC
       best_n_components = n_components_range[np.argmin(bic_scores)] 
    elif (crit == "bicconv"): 
       # Determine convergence of BIC scores
       diff = np.diff(bic_scores)
       # Calculate the percentiles in terms of the bic scores and the finite difference
       percentiles = np.arange(0, 101)
       diff_distribution = [np.percentile(abs(diff), p) for p in percentiles]
       bic_distribution = [np.percentile(bic_scores, p) for p in percentiles]
       # Determine convergence threshold as a percentage of the maximum change in smoothed BIC scores
       bic_thres = bic_distribution[percbic]
       diff_thres = diff_distribution[percdiff]
       best_n_components = n_components_range[np.argmin(bic_scores)] 
       for p in range(1,nmax-1):
           low_dev=abs(bic_scores[p]-bic_scores[p-1]) 
           up_dev=abs(bic_scores[p+1]-bic_scores[p])
           score=bic_scores[p]
           if (low_dev <= diff_thres and up_dev <= diff_thres and score <= bic_thres) == True:
              best_n_components=p
              break
    else:
       raise AssertionError("Unrecognized keyword :", crit)
    plt.plot(xdata, ydata, label='BIC Scores')
    plt.plot(xdata[best_n_components-1], ydata[best_n_components-1], 'ro',label='Best score')
    plt.xlabel('Number of Components')
    plt.ylabel('BIC Score')
    plt.legend()
    if file is None:
       plt.show()
    else:
       plt.savefig(file)
    plt.close()
    
    return best_n_components


def gaussian_overlap(g1, g2):
    """
    Calculate the overlap between two Gaussian functions.

    Parameters:
    - g1 (tuple): Tuple containing the mean and standard deviation of the first Gaussian function.
    - g2 (tuple): Tuple containing the mean and standard deviation of the second Gaussian function.

    Returns:
    - overlap (float): The overlap between the two Gaussian functions.
    """

    mean1, std1 = g1
    mean2, std2 = g2
    overlap = norm.cdf(mean1, mean2, np.sqrt(std1**2 + std2**2))
    return overlap

def compute_auxiliar_gaussians(gmm, x_range, thres):
    """
    Fill gaps between Gaussians with auxiliary Gaussians to ensure full coverage of the given range.

    Parameters:
    - gmm (GaussianMixture): Gaussian Mixture Model (GMM) containing the original Gaussian components.
    - x_range (tuple): Range to be covered by Gaussian functions, specified as (start, end).
    - thres (float): Lower overlap threshold below which auxiliary Gaussians will be created.

    Returns:
    - new_gaussians (list): List of tuples representing the means and standard deviations of the newly created auxiliary Gaussians.

    The function takes a Gaussian Mixture Model (gmm) and the desired range (x_range) to be covered by Gaussian functions.
    It then identifies gaps between original Gaussians and fills these gaps with auxiliary Gaussians to ensure complete coverage
    of the specified range.
    """

    # Store the information about the current gaussians as an array
    means=gmm.means_.flatten()
    sdev=np.sqrt(gmm.covariances_.flatten())
    gaussians = []
    for mean, std in zip(means, sdev):
        gaussian = {'mean': mean, 'std': std}
        gaussians.append(gaussian)
    gaussians = sorted(gaussians, key=lambda x: x['mean'])
    gaussians = [(g['mean'], g['std']) for g in gaussians]
    sorted_gaussians = sorted(gaussians, key=lambda g: g[0])  # Sort by mean
    new_gaussians = []
    x_values = np.arange(x_range[0], x_range[1] + 1)
    for i in range(len(sorted_gaussians) - 1):
        current_gaussian = sorted_gaussians[i]
        next_gaussian = sorted_gaussians[i + 1]
        overlap = gaussian_overlap(current_gaussian, next_gaussian)
        # If the overlap is lower than the threshold new gaussians must be used
        if overlap < thres:
            gap_start = current_gaussian[0] + current_gaussian[1]
            gap_end = next_gaussian[0] - next_gaussian[1]
            gap = gap_end - gap_start
            # Add auxiliary Gaussian to fill the gap
            aux_mean = gap_start + gap / 2
            aux_std = gap / 4  # Adjust the standard deviation as needed
            new_gaussians.append((aux_mean, aux_std))
    # Combine the original Gaussians and auxiliary Gaussians
    all_gaussians = sorted_gaussians + new_gaussians
    all_gaussians.sort(key=lambda g: g[0])  # Sort by mean again
    # Combine all Gaussians
    all_gaussians += new_gaussians
    all_gaussians.sort(key=lambda g: g[0])  # Sort by mean one last time
    return new_gaussians

def plot_aux(rcut=None,gmm=None,aux_gaussians=None,file=None,title=None):
    """
    Plot Gaussian functions and auxiliary functions on the same graph.

    Parameters:
    - rcut (float or None): Cutoff radius for the plot. 
    - gmm (GaussianMixture or None): Gaussian Mixture Model (GMM) containing the original Gaussian components.
    - aux_gaussians (list or None): List of tuples representing the means and standard deviations of auxiliary Gaussians.
    - file (str or None): If provided, saves the plot to the specified file; if None, displays the plot interactively.
    - title (str or None): Title for the plot.

    Returns:
    - None

    The function plots Gaussian functions (in red) and auxiliary functions (in blue) on the same graph. The plot can include
    original Gaussians from a Gaussian Mixture Model (gmm) and additional auxiliary Gaussians.
    """

    # Example usage
    x_range = (0, rcut)  # Range from 0 to 10
    x_values = np.linspace(x_range[0], x_range[1], 1000) 
    means=gmm.means_.flatten()
    sdev=np.sqrt(gmm.covariances_.flatten())
    gaussians = []
    for mean, std in zip(means, sdev):
        gaussian = {'mean': mean, 'std': std}
        gaussians.append(gaussian)
    gaussians = sorted(gaussians, key=lambda x: x['mean'])
    gaussians = [(g['mean'], g['std']) for g in gaussians]
    # Compute the maximum and minimum y-values among all gaussians
    max_y = float('-inf')
    min_y = float('inf')
    auxiliary_patch = mpatches.Patch(color='blue', label='Auxiliary Functions')
    gaussian_patch = mpatches.Patch(color='red', label='Gaussians')
    # Iterate over aux_gaussians and gaussians to find the maximum and minimum y-values
    for gaussian in aux_gaussians + gaussians:
        y_values = norm.pdf(x_values, gaussian[0], gaussian[1])
        max_y = max(max_y, np.max(y_values))
        min_y = min(min_y, np.min(y_values))
    # Plot filled Gaussians in blue
    for gaussian in aux_gaussians:
        y_values = norm.pdf(x_values, gaussian[0], gaussian[1])
        normalized_y = (y_values - min_y) / (max_y - min_y)
        plt.plot(x_values, normalized_y, 'b')
    for gaussian in gaussians:
        y_values = norm.pdf(x_values, gaussian[0], gaussian[1])
        normalized_y = (y_values - min_y) / (max_y - min_y)
        plt.plot(x_values, normalized_y, 'r')
    # Show the plot
    plt.legend(handles=[auxiliary_patch, gaussian_patch])
    plt.xlabel('Distance (Angstrom)')
    plt.ylabel('Counts')
    if title is not None: plt.title(title)
    if file is None:
       plt.show()
    else:
       plt.savefig(file)
    plt.close()
    return None


def find_gmm_params3D(data,n_comp,maxfev,cv_type='full'):
    """
    Find the parameters of a Gaussian Mixture Model (GMM) with a given number of components.

    Parameters:
    - data (array-like): The input data for which the GMM parameters are estimated.
    - n_comp (int): The desired number of components in the GMM.
    - maxfev (int): Maximum number of iterations for the optimization algorithm.
    - cv_type (str): Type of covariance parameters to use in the GMM ('full', 'tied', 'diag', 'spherical').

    Returns:
    - gmm_mix (GaussianMixture): Fitted Gaussian Mixture Model.
    - means (array): Array containing the means of the Gaussian components.
    - covariances (array): Array containing the covariance matrices of the Gaussian components.
    - weights (array): Array containing the weights of the Gaussian components.
    """

    # Fit the GMM to the best number of components and get its main parameters
    gmm_mix = GaussianMixture(n_components=n_comp, covariance_type='full',max_iter=100000000,random_state=1)
    gmm_mix.fit(data)
    means=[]
    covariances=[]
    weights=[]
    for p in range(gmm_mix.n_components):
        means.append(gmm_mix.means_[p])
        covariances.append(gmm_mix.covariances_[p])
        weights.append(gmm_mix.weights_[p])
    # Transform the list into an array
    means=np.array(means)
    covariances=np.array(covariances)
    weights=np.array(weights)
    return gmm_mix,means,covariances,weights

def filter3Ddata(data,rcut,pair):
    """
    Filter 3D trio data based on a distance cutoff and pair type.

    Parameters:
    - data (array-like): Input 3D trio data containing angles, distances_ij, and distances_ik.
    - rcut (float): Distance cutoff to filter data based on proximity.
    - pair (tuple): Pair of elements forming the trio (e.g., ('C', 'O')).

    Returns:
    - data_filtered (array): Filtered 3D trio data containing angles, distances_ij, and distances_ik.

    This function takes 3D trio data as input and filters it based on the specified distance cutoff (rcut).
    If the pair of elements forming the trio is homogeneous, only elements from the upper triangular matrix of the system are returned.
    """

    # Retrieve the data
    angles = []
    dist_ij = []
    dist_ik = []
    angle, d_ij, d_ik = zip(*data)
    angles.extend(angle)
    dist_ij.extend(d_ij)
    dist_ik.extend(d_ik)
    angles = np.array(angles)
    dist_ij = np.array(dist_ij)
    dist_ik = np.array(dist_ik)
    # Create boolean mask for filtering the data based on the cutoff
    mask = (dist_ij <= rcut) & (dist_ik <= rcut)
    angles_filtered = angles[mask]
    dist_ij_filtered = dist_ij[mask]
    dist_ik_filtered = dist_ik[mask]
    # In case the pair is formed by the same elements, we need to account for symmetry
    if (pair[0] == pair[1]):
       mask_sym = (dist_ik_filtered >= dist_ij_filtered)
       angles_filtered  = angles_filtered[mask_sym]
       dist_ij_filtered = dist_ij_filtered[mask_sym]
       dist_ik_filtered = dist_ik_filtered[mask_sym]
    data = np.column_stack((angles_filtered,dist_ij_filtered,dist_ik_filtered))
    return data

def plot_3Dgmm_components(gmm,data_norm,data_mean,data_std,file=None):
    """
    Plot individual Gaussian components of a 3D Gaussian Mixture Model (GMM).

    Parameters:
    - gmm (GaussianMixture): Fitted Gaussian Mixture Model.
    - data_norm (array-like): Normalized data used for training the GMM.
    - data_mean (array-like): Mean values used for denormalizing the data.
    - data_std (array-like): Standard deviations used for denormalizing the data.
    - file (str, optional): Filepath to save the plot. If None, the plot is displayed.

    This function visualizes each of the individual Gaussian components in the 3D space. 
    The raw output is de-normalized so that the original distributions can be recovered. 
    Along with the GMM distributions, it also plots the actual data from which the GMM was obtained.
    """
    # Denormalize the data
    data_denorm = data_norm*data_std + data_mean
    # Create the figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Represent the data
    ax.scatter(data_denorm[:, 0], data_denorm[:, 1], data_denorm[:, 2], c='k',s=0.5,alpha=1)
    n_comp=gmm.n_components
    # Represent the data for each of the GMM components
    for p in range(n_comp):
        # For each component, a single-GMM component is recreated
        gmm_comp = GaussianMixture(n_components=1, covariance_type=gmm.covariance_type)
        gmm_comp.means_ = np.array([gmm.means_[p]])
        gmm_comp.covariances_ = np.array([gmm.covariances_[p]])
        gmm_comp.weights_ = np.array([1.0])
        # Create random samples from the GMM component
        samples = gmm_comp.sample(n_samples=10000)[0]
        # Denormalize the data
        samples= samples*data_std + data_mean
        # Plot the 3D plots
        ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2],s=1.0,alpha=0.1)
    # Show the plot
    if file is None:
       plt.show()
    else:
       plt.savefig(file)
    plt.close()
    return None

def plot_gmm_components(gmm,data,nbins,title=None,file=None):
    """
    Plot individual Gaussian components of a 1D Gaussian Mixture Model (GMM).

    Parameters:
    - gmm (GaussianMixture): Fitted Gaussian Mixture Model.
    - data (array-like): Raw data used for training the GMM.
    - nbins (int): Number of bins for the histogram plot of raw data.
    - title (str, optional): Title of the plot. If None, no title is set.
    - file (str, optional): Filepath to save the plot. If None, the plot is displayed.

    This function visualizes each of the individual Gaussian components in a 1D space. 
    Along with the GMM components, the raw data is also plotted as a histogram.
    """
    # Create the figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    counts, bins, _ =ax.hist(data, bins=nbins, color='black', alpha=1, density=True, histtype='step')
    ax.clear()
    normalized_counts = counts / max(counts)
    ax.plot(bins[:-1], normalized_counts, color='black', linestyle='-', linewidth=1,label="Raw data")

    if title is not None: ax.set_title(title)
    ax.set_xlabel("Distance (Å)")
    ax.set_ylabel("Frequency (au)")
    n_comp=gmm.n_components
    pi, mu, sigma = gmm.weights_.flatten(), gmm.means_.flatten(), np.sqrt(gmm.covariances_.flatten())
    grid = np.arange(np.min(data), np.max(data), 1/nbins)
    # Calculate the sum of all Gaussians
    sum_gau = np.zeros_like(grid)
    for i in range(n_comp):
        sum_gau += pi[i] * norm.pdf(grid, loc=mu[i], scale=sigma[i])
    # Represent the data corresponding to each of the components
    for i in range(n_comp):
        plt.plot(grid, (pi[i] * norm.pdf(grid, loc=mu[i], scale=sigma[i]))/max(sum_gau),color='b',lw=2)
        plt.plot(grid,sum_gau/max(sum_gau),color='r',linestyle='dashed',lw=2)
    plt.legend(loc="upper right")
    # Show the plot
    if file is None:
       plt.show()
    else:
       plt.savefig(file)
    plt.close()
    return None

def print_gmm_components(gmm,decomp=0):
    """
    Print basic components of a Gaussian Mixture Model (GMM) ordered by weight.

    Parameters:
    - gmm (GaussianMixture): Fitted Gaussian Mixture Model.
    - decomp (int, optional): Number of Gaussians in which each GMM component will be further decomposed.
    
    This function prints the weights, means, variances, and covariances of each GMM component, ordered by weight.
    
    If decomp is not 0, further decompose each GMM component as a sum of Gaussians, print the details of the sub-components,
    and return a list containing the means and variances of the sub-components.

    Returns:
    - List: A list containing means and variances of sub-components if decomp is not 0, otherwise, None.
    """

    # Retrieve the parameters of the GMM models
    g_means= gmm.means_ 
    n_comp = gmm.n_components
    g_covariances=gmm.covariances_ 
    g_weights=gmm.weights_ 
    g_sorted = sorted(zip(g_weights, g_means, g_covariances), key=lambda x: x[0], reverse=True)
    # Print the components
    gmm_sub=None
    if decomp != 0 : gmm_sub=[]
    for i, (weight, mean, covariance) in enumerate(g_sorted):
        # Print results for a GMM component
        print(f"    # GMM component {i}: w = {weight:.6f}")
        print("       -mean      : "+ ' '.join([f'{value:.6f}'.center(10+7) for value in mean]))
        print("       -variance  : "+ ' '.join([f'{value:.6f}'.center(10+7) for value in np.diag(covariance)]))
        print("       -covariance:")
        for row in covariance:
            row_str = ' '.join([f'{value:.6f}'.center(10+7) for value in row])
            print("                   ",row_str)
        # If it is requested, further decompose each GMM component as a sum of gaussians
        if decomp != 0:
           # For each component, create a single-gaussian GMM model
           g1_comp = GaussianMixture(n_components=1, random_state=1,covariance_type=gmm.covariance_type)
           g1_comp.means_ = np.array([mean])
           g1_comp.covariances_ = np.array([covariance])
           g1_comp.weights_ = np.array([1.0])
           # Create random samples from the GMM component
           g1_comp_samples = g1_comp.sample(n_samples=10000)[0]
           # Divide the gaussians into as many gaussians as indicated by decomp
           g1_comp_samples=g1_comp_samples.reshape(-1, 1)
           g1_decomp = GaussianMixture(n_components=decomp, random_state=1, max_iter=10000,covariance_type=gmm.covariance_type).fit(g1_comp_samples)
           dg_means= g1_decomp.means_
           dg_n_comp = g1_decomp.n_components
           dg_covariances=g1_decomp.covariances_
           dg_weights=g1_decomp.weights_
           dg_sorted = sorted(zip(dg_weights, dg_means, dg_covariances), key=lambda x: x[0], reverse=True)
           for j, (dgweight, dgmean, dgcovariance) in enumerate(dg_sorted):
               # Print results for a GMM component
               print(f"       sub-component {j}: w = {weight:.6f} mean = {dgmean[0]:.6f} var = {dgcovariance[0][0]:.6f}")
               gmm_sub.append([dgmean[0],dgcovariance[0][0]])
    return gmm_sub

def print_3Dgmm_components(gmm=None,data_std=None,data_mean=None):
    """
    Print basic components of a 3D Gaussian Mixture Model (GMM) ordered by weight.
    The results are denormalized with respect to the input data distribution.

    Parameters:
    - gmm (GaussianMixture): Fitted 3D Gaussian Mixture Model.
    - data_std (array-like, optional): Standard deviations used to normalize the input data.
    - data_mean (array-like, optional): Means used to normalize the input data.

    This function prints the weights, means, variances, and covariances of each 3D GMM component, ordered by weight.
    The means and covariances are denormalized using the provided data_std and data_mean.
    """

    # Retrieve the parameters of the GMM models
    g_means= gmm.means_ 
    if (data_std is None):
       data_std=np.ones(g_means.shape[1])
    if (data_mean is None):
       data_mean=np.zeros(g_means.shape[1])
    n_comp = gmm.n_components
    g_means_denorm = g_means * data_std + data_mean
    g_covariances=gmm.covariances_ 
    g_weights=gmm.weights_ 
    dim=data_mean.shape[0]
    covariances = []
    for l in range(n_comp):
        covariance_norm = g_covariances[l]
        covariance = np.zeros((dim, dim))
        for m in range(dim):
            for n in range(dim):
                covariance[m][n] = covariance_norm[m][n] * data_std[m] * data_std[n]
        covariances.append(covariance)
    g_covariances_denorm=np.array(covariances)
    g_sorted = sorted(zip(g_weights, g_means_denorm, g_covariances_denorm), key=lambda x: x[0], reverse=True)
    # Print the components
    for i, (weight, mean, covariance) in enumerate(g_sorted):
        print(f"    # GMM component {i}: w = {weight:.6f}")
        print("       -mean      : "+ ' '.join([f'{value:.6f}'.center(10+7) for value in mean]))
        print("       -variance  : "+ ' '.join([f'{value:.6f}'.center(10+7) for value in np.diag(covariance)]))
        print("       -covariance:")
        for row in covariance:
            row_str = ' '.join([f'{value:.6f}'.center(10+7) for value in row])
            print("                   ",row_str)
    return None

def create_input_type(symbols):
    """
    A function to create the input.type used to compute the ACSF features
    """

    with open("input.type", 'w') as f:
         f.write(str(len(symbols))+"\n")
         for elem in symbols:
             f.write(elem+"\n")
    return None

def save_radial_acsf_params(gmm=None,gmm_sub=None,aux=None,file=None):
    """
    Save the main parameters of the radial ACSFs functions into an output file.

    Parameters:
    - gmm (GaussianMixture): Original GMM model.
    - gmm_sub (list, optional): GMM model of sub-components.
    - aux (list, optional): Set of auxiliary functions.
    - file (str): Output file to save the data.

    This function saves the mean and inverse variance parameters of the radial ACSFs functions
    into an output file. It includes information from the main GMM components, sub-components (if available),
    and auxiliary functions. The data is sorted by mean value.

    Returns:
    - radial_acsf (list): List containing the mean and inverse variance parameters of the radial ACSFs functions.
    """
    
    radial_acsf=[]
    # Retrieve the data of the GMM components:
    n_gmm,gmm_mu, gmm_sigma2 = gmm.n_components,gmm.means_.flatten(), gmm.covariances_.flatten()
    gmm_eta = 1/(2*gmm_sigma2)
    gmm_mu=gmm_mu.tolist()
    gmm_eta=gmm_eta.tolist()
    gmm_mu,gmm_eta=zip(*sorted(zip(gmm_mu, gmm_eta), key=lambda x: x[0]))
    # Retrieve the data from the GMM sub-components:
    if (gmm_sub is not None):
       n_sub=int(len(gmm_sub)/(n_gmm)) # Number of sub-components
       gmm_sub_mu=[]
       gmm_sub_eta=[]
       for val in gmm_sub:
           gmm_sub_mu.append(val[0])
           gmm_sub_eta.append(1/(2*val[1]))
       gmm_sub_mu,gmm_sub_eta=zip(*sorted(zip(gmm_sub_mu, gmm_sub_eta), key=lambda x: x[0]))
    # Retrieve the data from the auxiliar functions:
    if (aux is not None):
       aux_mu=[]
       aux_eta=[]
       for val in aux:
           aux_mu.append(val[0])
           aux_eta.append(1/(2*(val[1]**2)))
    counter=0
    with open(file,'w') as f:
       for gau in range(len(gmm_mu)): # print the main gaussian
           f.write(f"{gmm_mu[gau]:.6f}  {gmm_eta[gau]:.6f}  # GMM\n")
           # Print the sub-components
           if gmm_sub is not None:
              for sub_gau in range(n_sub):
                  f.write(f"    {gmm_sub_mu[counter]:.6f}  {gmm_sub_eta[counter]:.6f}  # SGMM\n")
                  radial_acsf.append([gmm_sub_mu[counter],gmm_sub_eta[counter]])
                  counter+=1
           else:
               radial_acsf.append([gmm_mu[gau],gmm_eta[gau]])
           # Print the auxiliar functions:
       if aux is not None:
          for aux_gau in range(len(aux_mu)):
              f.write(f"{aux_mu[aux_gau]:.6f}  {aux_eta[aux_gau]:.6f}  # AGMM\n")
              radial_acsf.append([aux_mu[aux_gau],aux_eta[aux_gau]])
    return radial_acsf

def create_input_rad_header(rtype,rcut,max_rad):
    """
    Create the header of the input rad file.

    Parameters:
    - rtype (str): Type of radial function.
    - rcut (float): Cutoff radius.
    - max_rad (int): Maximum number of radial functions.

    This function creates the header of the input rad file with the specified radial function type,
    cutoff radius, and maximum number of radial functions. It reads the existing content of the file,
    then writes the new header information at the beginning, followed by the existing content.
    """

    with open('input.rad', 'r') as f:
        existing_content = f.read()
    # Open the file in write mode, which truncates the file
    with open('input.rad', 'w') as f:
        # Write the new content at the beginning of the file
        f.write(str(rtype)+"\n")
        f.write(str(rcut)+"\n")
        f.write(str(max_rad)+"\n")
        # Write the previously read content to the file
        f.write(existing_content)
    return None

def create_input_rad_header_new_format(rtype,rcut):
    """
    Create the header of the input rad file.

    Parameters:
    - rtype (str): Type of radial function.
    - rcut (float): Cutoff radius.

    This function creates the header of the input rad file with the specified radial function type,
    cutoff radius, and maximum number of radial functions. It reads the existing content of the file,
    then writes the new header information at the beginning, followed by the existing content.
    """

    with open('input.rad', 'r') as f:
        existing_content = f.read()
    # Open the file in write mode, which truncates the file
    with open('input.rad', 'w') as f:
        # Write the new content at the beginning of the file
        f.write(str(rtype)+"\n")
        f.write(str(rcut)+"\n")
        # Write the previously read content to the file
        f.write(existing_content)
    return None

def create_input_rad(elemi,elemj,symbols,rad_acsf):
    """
    Create the input_rad file for a pair of elements.

    Parameters:
    - elemi (str): Element 1 symbol.
    - elemj (str): Element 2 symbol.
    - symbols (list): List of chemical elements.
    - rad_acsf (list): List of parameters for radial Gaussians.

    This function creates the input_rad file for a pair of elements specified by elemi and elemj.
    It writes the element indices, the number of radial Gaussians, and the parameters for each Gaussian
    to the file. If the pair is heteroatomic, it also writes the reverse pair.
    """

    indexi = symbols.index(elemi)+1  
    indexj = symbols.index(elemj)+1  
    num=len(rad_acsf)
    with open("input.rad",'a+') as f:
         f.write(str(indexi) + " "+str(indexj)+" "+str(num)+"\n")
         for gaus in rad_acsf:
             f.write("{:.6f} {:.6f}\n".format(gaus[0], gaus[1]))
    # Account for heteroatomic pairs
    if (indexi != indexj):
        with open("input.rad",'a+') as f:
             f.write(str(indexj) + " "+str(indexi)+" "+str(num)+"\n")
             for gaus in rad_acsf:
                 f.write("{:.6f} {:.6f}\n".format(gaus[0], gaus[1]))
    return None

def create_input_rad_new_format(elemi,elemj,symbols,rad_acsf):
    """
    Create the input_rad file for a pair of elements.

    Parameters:
    - elemi (str): Element 1 symbol.
    - elemj (str): Element 2 symbol.
    - symbols (list): List of chemical elements.
    - rad_acsf (list): List of parameters for radial Gaussians.

    This function creates the input_rad file for a pair of elements specified by elemi and elemj.
    It writes the element indices, the number of radial Gaussians, and the parameters for each Gaussian
    to the file. If the pair is heteroatomic, it also writes the reverse pair.
    """

    indexi = symbols.index(elemi)
    indexj = symbols.index(elemj) 
    num=len(rad_acsf)
    with open("input.rad",'a+') as f:
         f.write(elemi + " "+elemj+" "+str(num)+"\n")
         for gaus in rad_acsf:
             f.write("{:.6f} {:.6f}\n".format(gaus[0], gaus[1]))
    # Account for heteroatomic pairs
    if (indexi != indexj):
        with open("input.rad",'a+') as f:
             f.write(elemj+ " "+elemi+" "+str(num)+"\n")
             for gaus in rad_acsf:
                 f.write("{:.6f} {:.6f}\n".format(gaus[0], gaus[1]))
    return None

def clean_files(tipo):
    """
    This function cleans the input ACSF files to avoid inconsistencies.
    """
    try:
       os.remove("input.type")
    except FileNotFoundError:
       pass
    else:
       print(" # INFO: an existing input.type file has been removed.")
    if tipo == "angular":
       try:
          os.remove("input.ang")
       except FileNotFoundError:
          pass
       else:
          print(" # INFO: an existing input.ang file has been removed.")
    elif tipo == "radial":
       try:
          os.remove("input.rad")
       except FileNotFoundError:
          pass
       else:
          print(" # INFO: an existing input.rad file has been removed.")
    return None

def even_gaussians(rc, n):
    """
    Create n evenly sampled Gaussian functions in the space from 0 to rc.

    Parameters:
    - rc (float): The maximum radius of the space (Angstroms).
    - n (int): The number of Gaussian functions to create.

    This function generates n Gaussian functions evenly spaced in the interval [0, rc].
    The standard deviations of the Gaussians are adjusted to achieve equal overlap.
    
    Returns:
    - gaussian_mean (list): List of means for each Gaussian.
    - gaussian_sdev (list): List of standard deviations for each Gaussian.
    """
    widths = np.full(n, rc / n)  # Set initial widths for all Gaussians
    overlaps = np.diff(np.linspace(0, rc, n + 1))  # Calculate overlap between Gaussians
    widths[:-1] *= np.sqrt(overlaps[1:] / overlaps[:-1])  # Adjust widths to achieve same overlap

    x = np.linspace(0, rc, 1000)  # Sample points for plotting
    gaussian_mean = []
    gaussian_sdev = []
    for i in range(n):
        gaussian_mean.append(i * rc / n + widths[i] / 2)  # Mean of the Gaussian
        gaussian_sdev.append(widths[i] / 2)  # Standard deviation of the Gaussian
    return gaussian_mean,gaussian_sdev

def displace_gaus(gmm,sigma_scale):
    """
    Displace the Gaussian components in a GMM to ensure unique coverage of lower bounds.

    Parameters:
    - gmm (GaussianMixture): The GMM model containing the Gaussian components.
    - sigma_scale (float): Scaling factor for displacing the Gaussian components.

    This function displaces the means of the Gaussian components in a GMM by adding a scaled
    value of the standard deviation to ensure that the lower bounds of the Gaussian distributions
    are uniquely covered by the new Gaussians.
    Returns:
    - gmm_disp (GaussianMixture): The GMM model with displaced Gaussian components.
    """

    n_components=gmm.n_components
    mean=gmm.means_.flatten()
    sdev=np.sqrt((gmm.covariances_.flatten()))
    disp_mean=[]
    disp_var=[]
    for k in range(n_components):
         disp_mean.append(mean[k] + sigma_scale*sdev[k])
         disp_var.append((sdev[k]*2)**2)
    gmm_disp = GaussianMixture(n_components=n_components, random_state=1,covariance_type=gmm.covariance_type)
    gmm_disp.means_ = np.array([disp_mean]).reshape(-1,1)
    gmm_disp.covariances_ = np.array([disp_var]).reshape(-1,1,1)
    gmm_disp.weights_ = gmm.weights_
    return gmm_disp

def binary_gaus(gmm, alpha=3,beta=0.5):
    """
    Create lower and upper displaced Gaussian functions for a GMM model.

    Parameters:
    - gmm (GaussianMixture): The GMM model containing the Gaussian components.
    - alpha (float): Scaling factor for the standard deviation. Determines the center of the binary Gaussians:
      u + alpha * sdev and u - alpha * sdev.
    - beta (float): Scaling factor in the offset of the tails of the binary Gaussians. Determines
      the effective range covered by each of the binary Gaussians:
        - u_low + alpha * sdev_low  =  u + beta * sdev
        - u_high - alpha * sdev_high =  u - beta * sdev
      Higher values of beta result in more overlap of the binary Gaussians around the cluster center.

    Returns:
    - gmm_bin (GaussianMixture): The GMM model with binary displaced Gaussian components.
    """

    n_components=gmm.n_components
    mean=gmm.means_.flatten()
    sdev=np.sqrt((gmm.covariances_.flatten()))
    weight=gmm.weights_.flatten()
    bin_mean=[]
    bin_var=[]
    bin_w=[]
    for k in range(n_components):
         # Lower gaussian
         bin_mean.append(mean[k] -alpha*sdev[k])
         bin_var.append((sdev[k]*(1+beta/alpha))**2)
         bin_w.append(weight[k]/2)
         # Upper gaussian
         bin_mean.append(mean[k] +alpha*sdev[k])
         bin_var.append((sdev[k]*(1+beta/alpha))**2)
         bin_w.append(weight[k]/2)
    gmm_bin = GaussianMixture(n_components=2*n_components, random_state=1,covariance_type=gmm.covariance_type)
    gmm_bin.means_ = np.array([bin_mean]).reshape(-1,1)
    gmm_bin.covariances_ = np.array([bin_var]).reshape(-1,1,1)
    gmm_bin.weights_ = np.array([bin_w]).flatten()
    return gmm_bin

def anglegaus_2_anglecos(mu,sigma2):
    """
    Transform the parameters of the Gaussian kernel to an equivalent kernel in terms of cosine
    (as commonly used in classical angular representations).

    Parameters:
    - mu (float): Mean of the Gaussian function in the angular space (angles).
    - sigma2 (float): Variance of the Gaussian function in the angular space (angles^2).

    Returns:
    - w_fit (float): Fitted parameter for the cosine kernel.
    - overlap_ratio (float): Ratio of the overlap between the Gaussian and cosine kernels.
    """

    def cos_kernel(x, w):
        return ((1 + np.cos(x - xs)) ** w) * (2 ** (1 - w))
    def gaus_kernel(x, A, mu, sigma):
        return A * np.exp(-(x - mu)**2 / (2 * sigma**2))
    A = 2  # Amplitude
    
    # Transform degrees into radians
    mu=np.radians(mu)
    sigma=np.radians(np.sqrt(sigma2))
    # We will adjust between gaussians between 10 sigmas
    x = np.linspace(mu-10*sigma, mu+10*sigma, 10000)
    # Generate y values using the cosine function with Gaussian noise
    y = gaus_kernel(x, A, mu, sigma)
    # Set the center of the function to mu
    xs=mu
    # Fit the data to a Gaussian function
    popt, _ = curve_fit(cos_kernel, x, y)
    # Extract the parameters of the Gaussian function
    w_fit = popt
    # Compute the overlap between the gaussian and cosine kernels
    # Calculate the values of both functions
    y_cos  = cos_kernel(x, w_fit)
    y_gaus = gaus_kernel(x, A, mu, sigma)
    # Calculate the areas under both curves
    area_cos = simps(y_cos, x)
    area_gaus = simps(y_gaus, x)
    # Calculate the overlap ratio
    overlap_ratio = area_gaus/area_cos
    return w_fit,overlap_ratio

def chemmat(symbols):
    """
    Compute the neighborhood matrix of the system.

    Parameters:
    - symbols (list): List containing the chemical elements.

    Returns:
    - telem (int): Number of elements.
    - indjk (numpy.ndarray): Atomic pair IDs matrix.
    - vecino (numpy.ndarray): Array containing atomic pair indices.
    """
    telem=len(symbols)
    indjk = np.zeros(telem, dtype=int)
    vecino = np.zeros((telem*(telem+1)//2,), dtype=int)
    for l in range(1,telem+1):
        indjk[l-1] = (l-1)*(2*telem-l)
    counter = 0
    print(" # Chemical diversity information :")
    print("  - Number of elements", telem)
    for l in range(telem):
        print("    Element ", l+1, " : ",symbols[l])
    print("  - Atomic pair IDs matrix :")
    for l in range(telem):
        for k in range(l, telem):
            counter += 1
            vecino[counter-1] = indjk[l] + k + 1
            if (l == k):
               print("    ",symbols[l],symbols[k],"          : ",vecino[counter-1])
            else:
               print("    ",symbols[l],symbols[k]," or ",symbols[k],symbols[l]," : ",vecino[counter-1])
    return telem, indjk, vecino

def save_angular_acsf_params(gmm=None,mean=None,std=None,file=None,atype=3,frac=1):
    """
    Save the main parameters of the angular ACSFs functions into an output file.
    Also, create plots and return a dictionary with the parameters of each of the selected angular ACSFs.

    Parameters:
    - gmm (object): Original GMM model.
    - mean (numpy.ndarray): Mean of the input data (to denormalize the parameters of the GMM models).
    - std (numpy.ndarray): Std of the input data (to denormalize the parameters of the GMM models).
    - file (str): Output file to save the data.
    - atype (int): Type of angular symmetry function to be employed.
    - frac (float): Percentage of angular ACSF to be used, goes from 0 to 1.

    Returns:
    - angular_acsf (list): List containing parameters of each selected angular ACSF.
    """
    angular_acsf=[]
    # Retrieve the data of the GMM components:
    n_gmm,gmm_mu,gmm_cov,gmm_w = gmm.n_components,gmm.means_, gmm.covariances_,gmm.weights_
    # Sort the GMMs by weight:
    # Grab only the percentage given by frac
    N = int(len(gmm_w) * frac)
    sorted_indices = np.argsort(-gmm_w)
    gmm_mu = gmm_mu[sorted_indices][:N]
    gmm_w = gmm_w[sorted_indices][:N]
    gmm_cov = gmm_cov[sorted_indices][:N]
    # Denormalize the data
    gmm_mu = gmm_mu * std + mean
    dim=gmm_mu.shape[1]
    covariances = []
    for l in range(N):
        covariance_norm = gmm_cov[l]
        covariance = np.zeros((dim, dim))
        for m in range(dim):
            for n in range(dim):
                covariance[m][n] = covariance_norm[m][n] * std[m] * std[n]
        covariances.append(covariance)
    gmm_cov=np.array(covariances)
    # Get the main parameters in the 3d space:
    mu_angle=[]
    mu_rij=[]
    mu_rik=[]
    sigma_angle=[]
    sigma_rij=[]
    sigma_rik=[]
    for i in range(N):
        mu_angle.append(gmm_mu[i][0])
        mu_rij.append(gmm_mu[i][1])
        mu_rik.append(gmm_mu[i][2])
        sigma_angle.append(np.sqrt(gmm_cov[i,0,0]))
        sigma_rij.append(np.sqrt(gmm_cov[i,1,1]))
        sigma_rik.append(np.sqrt(gmm_cov[i,2,2]))
    # Create the parameters of the angular ACSF
    angular_acsf=[]
    if (atype == 3): # Heavily modified angular ACSF
       rs=[]
       xi=[]
       eta=[]
       theta_s=[]
       print(" # Mapping the exponential parameters to the cosine kernel")
       for i in range(N):
           # Get the xi values
           val,overlap=anglegaus_2_anglecos(mu_angle[i],(sigma_angle[i])**2)
           xi.append(val[0])
           # As we rely on a numerical mapping, the overlap between the observed and reconstructed angular functions is computed
           if overlap > 1.1 or overlap < 0.9:
              print(" ! Warning, poor overlap between the cosine and exponential kernels in function ",i+1," : ", '{0:.4f}'.format(overlap))
           # Get the rs parameters: (obtained as the average value)
           rs.append((mu_rij[i]+mu_rik[i])/2)
           # Get the eta parameters: (obtained as the average value)
           sigma2dum=((sigma_rij[i] + sigma_rik[i])/2)**2
           eta.append(1/(2*sigma2dum))
           # Get the angular shifts:
           theta_s.append(mu_angle[i])
       counter=0
       with open(file,'w') as f:
            f.write(" Heavily Modified ACSF: R_S XI ETA THETA_S"+"\n")
            for gau in range(N): # print the gaussian components:
                f.write(f"{rs[gau]:12.4f} {xi[gau]:12.4f} {eta[gau]:12.4f}  {theta_s[gau]:12.4f} # GMM\n")
                angular_acsf.append((rs[gau],xi[gau],eta[gau],theta_s[gau]))
    elif (atype ==5): # Heavily modified angular ACSF with independent radial grids
       xi=[]
       theta_s=[]
       eta_ij=[]
       rs_ij=[]
       eta_ik=[]
       rs_ik=[]
       print(" # Mapping the exponential parameters to the cosine kernel")
       for i in range(N):
           # Get the xi values
           val,overlap=anglegaus_2_anglecos(mu_angle[i],(sigma_angle[i])**2)
           xi.append(val[0])
           # As we rely on a numerical mapping, the overlap between the observed and reconstructed angular functions is computed
           if overlap > 1.1 or overlap < 0.9:
              print(" ! Warning, poor overlap between the cosine and exponential kernels in function ",i+1," : ", '{0:.4f}'.format(overlap))
           # Get the rs parameters: 
           rs_ij.append(mu_rij[i])
           rs_ik.append(mu_rik[i]) 
           # Get the eta parameters: 
           eta_ij.append(1/(2*(sigma_rij[i]**2)))
           eta_ik.append(1/(2*(sigma_rik[i]**2)))
           # Get the angular shifts:
           theta_s.append(mu_angle[i])
       counter=0
       with open(file,'w') as f:
            f.write(" Heavily Modified ACSF (wirg): R_S (ij) R_S (ik) XI ETA (ij) ETA (ik) THETA_S"+"\n")
            for gau in range(N): # print the gaussian components:
                f.write(f"{rs_ij[gau]:12.4f} {rs_ik[gau]:12.4f} {xi[gau]:12.4f} {eta_ij[gau]:12.4f} {eta_ik[gau]:12.4f} {theta_s[gau]:12.4f} # GMM\n")
                angular_acsf.append((rs_ij[gau],rs_ik[gau],xi[gau],eta_ij[gau],eta_ik[gau],theta_s[gau]))
    elif (atype == 6): # Pairwise expansion of 3B terms (PE3B)
       rs_ij=[]
       rs_ik=[]
       rs_jk=[]
       eta_ij=[]
       eta_ik=[]
       eta_jk=[]
       for i in range(N):
           # Get the rs parameters:
           rs_ij.append(mu_rij[i])
           rs_ik.append(mu_rik[i])
           mu_rjk = np.sqrt(mu_rij[i]**2 + mu_rik[i]**2 - 2 * mu_rij[i]*mu_rjk[i]*np.cos(np.radians(mu_angle[i])))
           rs_jk.append(mu_rjk)
           print(mu_rij[i],mu_rik[i],mu_angle[i],mu_rjk)
           # Get the eta parameters:
           theta_sin=np.sin(np.radians(mu_angle[i]))
           theta_cos=np.cos(np.radians(mu_angle[i]))
           sigma_rjk = np.sqrt((mu_rik[i]*theta_sin*sigma_rij[i])**2 + (mu_rij[i]*theta_sin*sigma_rik[i])**2 + (mu_rik[i]*mu_rij[i]*theta_cos*np.radians(sigma_angle[i]))**2)
           print(sigma_rij[i],sigma_rik[i],sigma_rjk)
           eta_ij.append(1/(2*(sigma_rij[i]**2)))
           eta_ik.append(1/(2*(sigma_rik[i]**2)))
           eta_jk.append(1/(2*(sigma_rjk**2)))
       counter=0
       with open(file,'w') as f:
            f.write(" PE3B: R_S (ij) R_S (ik) R_S (jk) ETA (ij) ETA (ik) ETA (jk)"+"\n")
            for gau in range(N): # print the gaussian components:
                f.write(f"{rs_ij[gau]:12.4f} {rs_ik[gau]:12.4f} {rs_jk[gau]:12.4f} {eta_ij[gau]:12.4f} {eta_ik[gau]:12.4f} {eta_jk[gau]:12.4f} # GMM\n")
                angular_acsf.append((rs_ij[gau],rs_ik[gau],rs_jk[gau],eta_ij[gau],eta_ik[gau],eta_jk[gau]))
    return angular_acsf

def create_input_ang(elem,pair,symbols,ang_acsf,atype):
    """
    Create the input.ang file.

    Parameters:
    - elem (str): Element.
    - pair (tuple): Pair of neighboring elements.
    - symbols (list): List containing the chemical elements.
    - ang_acsf (list): List containing the parameters of the angular ACSFs.
    - atype (int): Type of angular ACSF to be used (3 or 5)

    # 3 Heavily modified angular function
    # 5 Heavily modified angular function with independent radial grids
    """

    # Get the ids of the atom and the neighboring pair
    index_atom = symbols.index(elem)+1  
    telem=len(symbols)
    indjk = np.zeros(telem, dtype=int)
    for l in range(1,telem+1):
        indjk[l-1] = (l-1)*(2*telem-l)
    index_atomi= symbols.index(pair[0])
    index_atomj= symbols.index(pair[1])
    if (index_atomi > index_atomj):
       index_pair = indjk[index_atomj] + index_atomi + 1
    else:
       index_pair = indjk[index_atomi] + index_atomj + 1
    num=len(ang_acsf)
    with open("input.ang",'a+') as f:
         f.write(str(index_atom) + " "+str(index_pair)+" "+str(num)+"\n")
         if (atype == 3): # Heavily Modified:
             for gaus in ang_acsf:
                 f.write("{:.6f} {:.6f} {:.6f} {:.6f}\n".format(gaus[0], gaus[1], gaus[2], gaus[3]))
         if (atype == 5): # Heavily Modified with independent radial grids:
             for gaus in ang_acsf: 
                 f.write("{:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(gaus[0], gaus[1], gaus[2], gaus[3], gaus[4], gaus[5]))
    return None

def create_input_ang_new_format(elem,pair,symbols,ang_acsf,atype):
    """
    Create the input.ang file using a more modern format.

    Parameters:
    - elem (str): Element.
    - pair (tuple): Pair of neighboring elements.
    - symbols (list): List containing the chemical elements.
    - ang_acsf (list): List containing the parameters of the angular ACSFs.
    - atype (int): Type of angular ACSF to be used.

    # Angular ACSF Types:
    #   3: Heavily modified angular function
    #   5: Heavily modified angular function with independent radial grids

    This is a modified version of create_input_ang that uses a more modern format.

    """
    num=len(ang_acsf)
    with open("input.ang",'a+') as f:
         f.write(str(elem) + " "+str(pair)+" "+str(num)+"\n")
         if (atype == 3): # Heavily Modified:
             for gaus in ang_acsf:
                 f.write("{:.6f} {:.6f} {:.6f} {:.6f}\n".format(gaus[0], gaus[1], gaus[2], gaus[3]))
         if (atype == 5): # Heavily Modified with independent radial grids:
             for gaus in ang_acsf: 
                 f.write("{:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(gaus[0], gaus[1], gaus[2], gaus[3], gaus[4], gaus[5]))
    return None

def create_input_ang_header(atype,rcut,max_ang):
    """
    Create the header of the input ang file.

    Parameters:
    - atype (int): Type of angular function.
    - rcut (float): Cutoff radius.
    - max_ang (int): Maximum number of angular functions.
    """

    with open('input.ang', 'r') as f:
        existing_content = f.read()
    # Open the file in write mode, which truncates the file
    with open('input.ang', 'w') as f:
        # Write the new content at the beginning of the file
        f.write(str(atype)+"\n")
        f.write(str(rcut)+"\n")
        f.write(str(max_ang)+"\n")
        # Write the previously read content to the file
        f.write(existing_content)
    return None

def create_input_ang_header_new_format(atype,rcut):
    """
    Create the header of the input ang file using a more modern format.

    Parameters:
    - atype (int): Type of angular function.
    - rcut (float): Cutoff radius.
    """

    with open('input.ang', 'r') as f:
        existing_content = f.read()
    # Open the file in write mode, which truncates the file
    with open('input.ang', 'w') as f:
        # Write the new content at the beginning of the file
        f.write(str(atype)+"\n")
        f.write(str(rcut)+"\n")
        # Write the previously read content to the file
        f.write(existing_content)
    return None

def plot_angular_acsf(atype=3,rcut=7.0,grad=200,gang=250,rs_ang=None,xi_ang=None,eta_ang=None,lambda_ang=None,theta_ang=None,file=None,title=None):
    """
    Plot the activation of angular ACSF functions in a polar plot.
    Currently, it only works for heavily modified angular functions 
    (type=3) and it plots a fuzzy representation in terms of the 
    mean radial and angular values.

    Parameters:
    - atype (int): Type of angular ACSF function to be used.
    - rcut (float): Cutoff radius in angstroms.
    - grad (int): Number of points for the radial grid.
    - gang (int): Number of points for the angular grid.
    - rs_ang (float): Radial shift (rs).
    - xi_ang (float): Xi value (exponent of the cosine kernel).
    - eta_ang (float): Eta value (exponent of the radial kernel).
    - lambda_ang (float): Lambda value (-1 or 1).
    - theta_ang (float): Shift of the angular part in degrees.
    - file (str): Image file where the plot will be saved.
    - title (str): Title of the image file.

    Returns:
    - None
    """
 
    # Convert the theta_ang degrees to radians (only for ACSF=3)
    if (atype == 3): 
       theta_s=np.radians(theta_ang)
    else:
       # This function is currently only available for symmetry functions of type 3.
       print(" INFO: plot_angular_acsf only works for angular ACSFs of type 3.")
       return None
    # Create the radial and angular grids
    radial_x=np.linspace(0,rcut,grad)     # in angstroms
    angular_x=np.linspace(0,np.pi,gang)   # in radians from 0 to pi
    r_values = []
    theta_values = []
    acsf_values = []
    for r in radial_x:
        for theta in angular_x:
            # Compute the value of the cutoff function
            fcut = fcutoff(r,rcut,cut_type='soft')
            # Compute angular contributions to ACSF
            if (atype == 1):
               fval = ((2**(1-xi_ang)) * 
                       (((1 + lambda_ang*np.cos(theta))**xi_ang) * np.exp(-eta_ang*((r-rs_ang)**2 + 
                       (r-rs_ang)**2 + (r-rs_ang)**2))) * fcut * fcut * fcut)
            elif (atype == 2):
               fval = ((2**(1-xi_ang)) * 
                        (((1 + lambda_ang*np.cos(theta))**xi_ang) * np.exp(-eta_ang*((r-rs_ang)**2 + 
                        (r-rs_ang)**2))) * fcut * fcut)
            elif (atype == 3):
               fval = ((2**(1-xi_ang)) * 
                        (((1 + np.cos(theta-theta_s)) ** xi_ang) * np.exp(-eta_ang*((((r+r)/2)-rs_ang)**2))) * 
                        fcut * fcut)
            r_values.append(r)
            theta_values.append(theta)
            acsf_values.append(fval)
    # Convert the lists to numpy arrays
    r_values = np.array(r_values)
    theta_values = np.array(theta_values)
    acsf_values = np.array(acsf_values)
    # Plotting the polar plot
    fig = plt.figure()
    if (title is not None): fig.suptitle(title)
    ax = fig.add_subplot(111, projection='polar')
    # The values should be adequately normalized between 0 and 1, so there is no need to adjust the scale
    scp = ax.scatter(theta_values, r_values, c=acsf_values, cmap='gnuplot')
    # Set the number of radial gridlines
    num_radial_grids = 5
    radii = np.linspace(0, r_values.max(), num_radial_grids)
    ax.set_yticks(radii)
    # Set the labels for the radial gridlines (optional)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    # Show the plot
    if file is None:
       plt.show()
    else:
       plt.savefig(file)
    plt.close()
    return None
   
def plot_angular_activations(data,elem,pair,atype=3,rcut=7.0,rs_ang=None,xi_ang=None,eta_ang=None,lambda_ang=None,theta_ang=None,file=None,title=None,ps=0.005,cmap_polar='bwr'):
    """
    Plot the activation of angular ACSF functions in a polar plot (rij,angle).
    The color is given by the value taken by the angular symmetry function.
    As such, the resultant plot shows the regions of the angular space in which
    a given symmetry function gets activated.

    Parameters:
    - data (list): List of variables containing the data (angle, d_ij, d_ik).
    - elem (str): Symbol of the central element of the trio (e.g., 'C').
    - pair (str): Pair of atoms forming the trio (e.g., 'OH' for forming 'C-O-H').
    - atype (int): Type of angular ACSF function to be used.
    - rcut (float): Cutoff radius in angstroms.
    - rs_ang (float or list): Radial shift (rs). For atype=5, it should be a list (rs_ij,rs_ik)
    - xi_ang (float): Xi value (exponent of the cosine kernel).
    - eta_ang (float or list): Eta value (exponent of the radial kernel). For atype=5, it should be a list (eta_ij, eta_ik).
    - lambda_ang (float): Lambda value (-1 or 1).
    - theta_ang (float): Shift of the angular part in degrees.
    - file (str): Image file where the plot will be saved.
    - title (str): Title of the image file.
    - ps (float): Maximum value of the point size.
    - cmap_polar (str): Colormap for polar plot.
    """
    # Convert the theta_ang degrees to radians
    try:   
       theta_s=np.radians(theta_ang)
    except NameError:
       pass
    # Retrieve the data
    angles = []
    dist_ij = []
    dist_ik = []
    dist_ik_color = []
    angle, d_ij, d_ik = zip(*data)
    angles.extend(angle)
    dist_ij.extend(d_ij)
    dist_ik.extend(d_ik)
    angles = np.array(angles)
    dist_ij = np.array(dist_ij)
    dist_ik = np.array(dist_ik)
    # Create boolean mask for filtering the data based on the cutoff
    mask = (dist_ij <= rcut) & (dist_ik <= rcut)
    angles_filtered = angles[mask]
    dist_ij_filtered = dist_ij[mask]
    dist_ik_filtered = dist_ik[mask]
    # Create a color variable for the polar plot
    dist_ratio_color = np.array(dist_ik_filtered/dist_ij_filtered)
    # Compute the activations of the angular ACSFs
    acsf_values = []
    r_val=[]
    a_val=[]
    val_val=[]
    for n in np.arange(angles_filtered.shape[0]):
        rij=dist_ij_filtered[n]
        rik=dist_ik_filtered[n]
        theta=np.radians(angles_filtered[n])
        rjk = np.sqrt(rij**2 + rik**2 - 2*rij*rik*np.cos(theta))
        fcutij = fcutoff(rij,rcut,cut_type='soft')
        fcutik = fcutoff(rik,rcut,cut_type='soft')
        fcutjk = fcutoff(rjk,rcut,cut_type='soft')
        if (atype == 1):
                fval = ((2**(1-xi_ang)) * 
                         (((1 + lambda_ang*np.cos(theta))**xi_ang) * 
                          np.exp(-eta_ang*((rij-rs_ang)**2 + (rik-rs_ang)**2 + (rjk-rs_ang)**2))) * fcutij * fcutik * fcutjk)
                acsf_values.append(fval)
        elif (atype == 2):
                fval = ((2**(1-xi_ang)) * 
                         (((1 + lambda_ang*np.cos(theta))**xi_ang) * 
                          np.exp(-eta_ang*((rij-rs_ang)**2 + (rik-rs_ang)**2))) * fcutij * fcutik)
                acsf_values.append(fval)
        elif (atype == 3):
                fval = ((2**(1-xi_ang)) * 
                         (((1 + np.cos(theta-theta_s)) ** xi_ang) * 
                          np.exp(-eta_ang*((((rij+rik)/2)-rs_ang)**2))) * fcutij * fcutik)
                r_val.append(rij)
                a_val.append(theta)
                val_val.append(fval)
                acsf_values.append(fval)
        elif (atype ==5): # Heavily modified angular function with independent radial grids
                fval = ((2**(1-xi_ang)) * 
                         (((1 + np.cos(theta-theta_s)) ** xi_ang) * 
                          np.exp(-eta_ang[0]*((rij-rs_ang[0])**2)) *  
                          np.exp(-eta_ang[1]*((rik-rs_ang[1])**2))) * fcutij * fcutik)
                r_val.append(rij)
                a_val.append(theta)
                val_val.append(fval)
                acsf_values.append(fval)
    acsf_values=np.array(acsf_values)
    # Radial axis
    fig = plt.figure()
    if (title is not None): fig.suptitle(title)
    fig_polar = fig.add_subplot(111, projection='polar')
    fig_polar.set_rlim([0, rcut])
    rticks =[]
    for i in np.arange(0,int(rcut)):
         rticks.append(i)
    fig_polar.set_rticks(rticks)
    fig_polar.set_rlabel_position(135)
    # Angular axis
    fig_polar.set_thetamin(0)
    fig_polar.set_thetamax(180)
    # Plot the data
    scp = fig_polar.scatter(np.radians(angles_filtered), dist_ij_filtered, c=acsf_values, cmap=cmap_polar, s=ps, alpha=1)
    # Set the number of radial gridlines
    num_radial_grids = 5
    radii = np.linspace(0, rcut, num_radial_grids)
    fig_polar.set_yticks(radii)
    # Show the plot
    if file is None:
       plt.show()
    else:
       plt.savefig(file)
    plt.close()
    return None
