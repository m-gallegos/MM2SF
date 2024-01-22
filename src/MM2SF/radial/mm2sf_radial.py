import numpy as np
from scipy.stats import norm
from itertools import combinations
import sys as sys
from scipy.spatial.distance import cdist
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy import stats
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from MM2SF.basics.functions import *

def radial_selector_tailormade(trjname=None, nbins=1000, nmax=15, max_iter=10000, bw=None, smooth='no',
                            rcut=7.0, trj_step=1, cut_type='hard', ndecomp=2,
                            over_thres=0.005, aux="yes", rtype=1, new_format=True):
    """
    Process trajectory data using Gaussian Mixture Model (GMM) analysis.

    Parameters:
    - trjname (str): Path to the trajectory file (XYZ format in Angstroms).
    - nbins (int): Number of bins to be used for the histograms.
    - nmax (int): Maximum number of Gaussian functions to look for.
    - max_iter (int): Maximum number of iterations for GMM.
    - bw (float): Bandwidth method for smoothing. Can be a float, 'scott', or 'silverman'.
    - smooth (str): Whether to smooth the data before passing it to the GMM models ('yes' or 'no').
    - rcut (float): Cutoff radius, in Angstroms.
    - trj_step (int): Step used to sample the geometries of the trajectory file (sampling frequency).
    - cut_type (str): Cutoff type ('hard' or 'soft').
    - ndecomp (int): Number of Gaussians in which each GMM component will be further decomposed.
    - over_thres (float): Overlap threshold to include auxiliary functions.
    - aux (str): Whether to include auxiliary functions ('yes' or 'no').
    - rtype (int): Type of radial (used for printing the input.rad file).
    - new_format (logic): use old (deprecated) or new format to store the ACSF angular parameters.

    """
    print(" # Spatial distribution strategy    = tailor-made")
    # Initialize some variables
    if bw is None: bw=50/float(nbins)
    gmm_sub=None
    aux_gaussians=None

    # Extract the geometries from the trjfile
    print(" # Will read data from ", trjname, " every ", str(trj_step), " steps.")
    print(" # Cutoff radius                    =", rcut)
    print(" # Cutoff type                      =", cut_type)
    print(" # Auxiliary functions              =", aux)
    if (aux == "yes"):
       print(" # Overlap threshold for auxiliars  =", over_thres)
    print(" # Data smoothing                   =", smooth)
    if (smooth == "yes"):
       print(" # BW for data smoothing            =", bw)
    print(" # Maximum number of GMM components =", nmax)
    print(" # Maximum number of GMM iterations =", max_iter)
    print(" # GMM sub-components of a GMM      =", ndecomp)
    print(" # Number of bins for 2D histograms =", nbins)
    print(" ")
    geometries = read_xyz_trajectory_wstep(trjname,sampling_step=trj_step)
    
    # Compute the radial distribution
    print(" # Computing the distance distribution........START!")
    symbols,dist_dists=compute_distance_distribution(geometries)
    clean_files('radial')
    if not new_format: create_input_type(symbols) # input.type file for the computation of ACSF features
    print(" # Computing the distance distribution........DONE!")
    max_rad=0
    for idx1 in np.arange(0,len(symbols)):
        for idx2 in np.arange(idx1,len(symbols)):
            elemi=str(symbols[idx1])
            elemj=str(symbols[idx2])
            print("---------------------------------------")
            print(" #          "+elemi+elemj+"          # ")
            print("---------------------------------------")
            dist=dist_dists[elemi][elemj]
            # Discard null values (appearing for same element distribution)
            mask = dist != 0
            dist=dist[mask]
            print(" # Discarding null values from distance distribution.")
            if len(dist) == 0 :
               print(" # Not enough data to perform the GMM clustering.")
               continue
            # Parse the distribution to 2D data:
            x,y=extract_distribution(dist, nbins)
            x= np.reshape(x, (-1, 1))
            y= np.reshape(y, (-1, 1))
            xydist=np.hstack((x, y))
            # Apply the cutoff function
            print(" # Applying cutoff function.")
            xydist_cut=apply_cut(xydist,rcut,cut_type=cut_type)
            # Smooth the data (if desired)
            if (smooth == "yes"):
               print(" # Applying a KDE smoothing to the distribution.")
               xydist_cut_kde,samples=smooth_data(xydist_cut,bw)
               xydist_cut_kde=normalize(xydist_cut_kde)
            # Normalize the distribution
            print(" # Normalizing the distribution.")
            xydist=normalize(xydist)
            xydist_cut=normalize(xydist_cut)
            # Recreate raw data from the distribution
            if (smooth == "yes"):
               dist_dat=xy2raw(xydist_cut_kde)
            else:
               dist_dat=xy2raw(xydist_cut)
            # Find the best number of GMM radial components
            print(" # Optimizing GMM models..............")
            n_comp=find_opt_components(dist_dat,nmax,max_iter)
            print(" # Ideal number of GMM components :",n_comp)
            # Find the parameters of the best GMM fitting
            gmm,g_means,g_covariances,g_weights,sum_gau,grid=find_gmm_params(dist_dat,n_comp,max_iter,nbins)
            print(" # Parameters of the individual components of the GMM model :")
            print(" ")
            gmm_sub=print_gmm_components(gmm,decomp=ndecomp)
            out_title=elemi+"-"+elemj+" radial distribution"
            # Add auxiliary functions (optional)
            if (aux == "yes"):  
               aux_gaussians = compute_auxiliar_gaussians(gmm, (0,rcut),over_thres)
               for t in range(len(aux_gaussians)):
                   val=aux_gaussians[t]
                   mean=val[0]
                   variance=val[1]**2
                   print(f"    # Auxiliar component {t}: ")
                   print("       -mean      : "+ ' '.join([f'{mean:.6f}'.center(10+7)]))
                   print("       -variance  : "+ ' '.join([f'{variance:.6f}'.center(10+7) ]))
               outname=str(elemi)+str(elemj)+"_radial_aux.png"
               plot_aux(rcut=rcut,gmm=gmm,aux_gaussians=aux_gaussians,file=outname,title=out_title)
            # Plot the radial distributions
            if (smooth == "yes"):
               outname=str(elemi)+str(elemj)+"_radial_dist_smooth.png"
               plot_gmm_components(gmm,samples.reshape(-1,1),nbins,title=out_title,file=outname)
            else:
               outname=str(elemi)+str(elemj)+"_radial_dist.png"
               plot_gmm_components(gmm,dist.reshape(-1,1),nbins,title=out_title,file=outname)
            print(" # 2D representation of GMM components saved to ", outname)
            # Create the ACSF radial features
            outname=str(elemi)+str(elemj)+".rad"
            rad_acsf=save_radial_acsf_params(gmm=gmm,gmm_sub=gmm_sub,aux=aux_gaussians,file=outname)
            if len(rad_acsf) > max_rad: max_rad = len(rad_acsf)
            print(" # Radial ACSF parameters saved to              ", outname)
            if new_format:
               create_input_rad_new_format(elemi,elemj,symbols,rad_acsf)
            else:
               create_input_rad(elemi,elemj,symbols,rad_acsf)
    # Create the header of the input.rad file
    if new_format: 
       create_input_rad_header_new_format(rtype,rcut)
    else:
       create_input_rad_header(rtype,rcut,max_rad)
    return None


def radial_selector_displaced(trjname=None, nbins=1000, nmax=15, max_iter=10000, bw=None, smooth='no',
                                 rcut=7.0, trj_step=1, cut_type='hard', over_thres=0.005,
                                 rtype=1, sigma_scale=3, new_format=True):
    """
    This function performs a displaced Gaussian selection on a trajectory.

    Parameters:
    - trjname (str): Path to the trajectory file (XYZ format in Angstroms).
    - nbins (int): Number of bins for histograms.
    - nmax (int): Maximum number of Gaussian functions to look for.
    - max_iter (int): Maximum number of iterations for GMM.
    - bw (float, optional): Bandwidth for smoothing; if None, calculated as 50/float(nbins).
    - smooth (str): Whether to smooth the data before passing it to the GMM models ('yes' or 'no').
    - rcut (float): Cutoff radius in Angstroms.
    - trj_step (int): Step used to sample the geometries.
    - cut_type (str): Cutoff type ('hard' or 'soft').
    - over_thres (float): Overlap threshold to include auxiliary functions.
    - rtype (int): Type of radial.
    - sigma_scale (int): Scaling factor for the standard deviation of the Gaussian used during the
                         displacement of the radial ACSF terms.
    - new_format (logic): use old (deprecated) or new format to store the ACSF angular parameters.

    Note: 
    Currently, auxiliary or decomposed functions are not available when using displaced radial ACSFs.

    This function represents an adapted version of the conventional radial selector.
    In this modification, a displacement is strategically applied to Gaussian functions,
    aiming to mitigate the challenge posed by the inherent symmetry imposed by Gaussian functions
    centered within clusters.

    Returns:
    None
    """
    
    # Initialize some variables
    print(" # Spatial distribution strategy    = displaced")
    if bw is None: bw=50/float(nbins)
    ndecomp=0         
    aux="no"         
    gmm_sub=None
    aux_gaussians=None

    # Extract the geometries from the trjfile
    print(" # Will read data from ", trjname, " every ", str(trj_step), " steps.")
    print(" # Cutoff radius                    =", rcut)
    print(" # Cutoff type                      =", cut_type)
    print(" # Auxiliar functions               =", aux)
    if (aux == "yes"):
       print(" # Overlap threshold for auxiliars  =", over_thres)
    print(" # Data smoothing                   =", smooth)
    if (smooth == "yes"):
       print(" # BW for data smoothing            =", bw)
    print(" # Maximum number of GMM components =", nmax)
    print(" # Maximum number of GMM iterations =", max_iter)
    print(" # GMM sub-components of a GMM      =", ndecomp)
    print(" # Number of bins for 2D histograms =", nbins)
    print(" # ALPHA                            =", sigma_scale)
    print(" ")
    geometries = read_xyz_trajectory_wstep(trjname,sampling_step=trj_step)
    
    # Compute the radial distribution
    print(" # Computing the distance distribution........START!")
    symbols,dist_dists=compute_distance_distribution(geometries)
    clean_files('radial')
    if not new_format: create_input_type(symbols) # input.type file for the computation of ACSF features
    print(" # Computing the distance distribution........DONE!")
    max_rad=0
    for idx1 in np.arange(0,len(symbols)):
        for idx2 in np.arange(idx1,len(symbols)):
            elemi=str(symbols[idx1])
            elemj=str(symbols[idx2])
            print("---------------------------------------")
            print(" #          "+elemi+elemj+"          # ")
            print("---------------------------------------")
            dist=dist_dists[elemi][elemj]
            # Discard null values (appearing for same element distribution)
            mask = dist != 0
            dist=dist[mask]
            print(" # Discarding null values from distance distribution.")
            if len(dist) == 0 :
               print(" # Not enough data to perform the GMM clustering.")
               continue
            # Parse the distribution to 2D data:
            x,y=extract_distribution(dist, nbins)
            x= np.reshape(x, (-1, 1))
            y= np.reshape(y, (-1, 1))
            xydist=np.hstack((x, y))
            # Apply a cutoff function
            print(" # Applying cutoff function.")
            xydist_cut=apply_cut(xydist,rcut,cut_type=cut_type)
            # Smooth the data (if desired)
            if (smooth == "yes"):
               print(" # Applying a KDE smoothing to the distribution.")
               xydist_cut_kde,samples=smooth_data(xydist_cut,bw)
               xydist_cut_kde=normalize(xydist_cut_kde)
            # Normalize the distribution
            print(" # Normalizing the distribution.")
            xydist=normalize(xydist)
            xydist_cut=normalize(xydist_cut)
            # Recreate raw data from the distribution
            if (smooth == "yes"):
               dist_dat=xy2raw(xydist_cut_kde)
            else:
               dist_dat=xy2raw(xydist_cut)
            # Find the best number of GMM radial components
            print(" # Optimizing GMM models..............")
            n_comp=find_opt_components(dist_dat,nmax,max_iter)
            print(" # Ideal number of GMM components :",n_comp)
            # Find the parameters of the best GMM fitting
            gmm,g_means,g_covariances,g_weights,sum_gau,grid=find_gmm_params(dist_dat,n_comp,max_iter,nbins)
            # Find the parameters of the best GMM fitting
            gmm=displace_gaus(gmm,sigma_scale)
            print(" # Parameters of the individual components of the GMM model :")
            print(" ")
            gmm_sub=print_gmm_components(gmm,decomp=ndecomp)
            out_title=elemi+"-"+elemj+" radial distribution"
            # Add auxiliar functions (optional)
            if (aux == "yes"):  
               aux_gaussians = compute_auxiliar_gaussians(gmm, (0,rcut),over_thres)
               for t in range(len(aux_gaussians)):
                   val=aux_gaussians[t]
                   mean=val[0]
                   variance=val[1]**2
                   print(f"    # Auxiliar component {t}: ")
                   print("       -mean      : "+ ' '.join([f'{mean:.6f}'.center(10+7)]))
                   print("       -variance  : "+ ' '.join([f'{variance:.6f}'.center(10+7) ]))
               outname=str(elemi)+str(elemj)+"_radial_aux.png"
               plot_aux(rcut=rcut,gmm=gmm,aux_gaussians=aux_gaussians,file=outname,title=out_title)
            # Plot the radial distributions
            if (smooth == "yes"):
               outname=str(elemi)+str(elemj)+"_radial_dist_smooth.png"
               plot_gmm_components(gmm,samples.reshape(-1,1),nbins,title=out_title,file=outname)
            else:
               outname=str(elemi)+str(elemj)+"_radial_dist.png"
               plot_gmm_components(gmm,dist.reshape(-1,1),nbins,title=out_title,file=outname)
            print(" # 2D representation of GMM components saved to ", outname)
            # Create the ACSF radial features 
            outname=str(elemi)+str(elemj)+".rad"
            rad_acsf=save_radial_acsf_params(gmm=gmm,gmm_sub=gmm_sub,aux=aux_gaussians,file=outname)
            if len(rad_acsf) > max_rad: max_rad = len(rad_acsf)
            print(" # Radial ACSF parameters saved to              ", outname)
            if new_format:
               create_input_rad_new_format(elemi,elemj,symbols,rad_acsf)
            else:
               create_input_rad(elemi,elemj,symbols,rad_acsf)
    #Create the header of the input.rad file:
    if new_format:
       create_input_rad_header_new_format(rtype,rcut)
    else:
       create_input_rad_header(rtype,rcut,max_rad)
    return None
    

def radial_selector_binary(trjname=None, nbins=1000, nmax=15, max_iter=10000, bw=None, smooth='no',
                              rcut=7.0, trj_step=1, cut_type='hard', over_thres=0.005,
                              rtype=1, alpha=3, beta=0.5, new_format=True):
    """
    This function performs a binary Gaussian selection on a trajectory.

    Parameters:
    - trjname (str): Path to the trajectory file (XYZ format in Angstroms).
    - nbins (int): Number of bins for histograms.
    - nmax (int): Maximum number of Gaussian functions to look for.
    - max_iter (int): Maximum number of iterations for GMM.
    - bw (float, optional): Bandwidth for smoothing; if None, calculated as 50/float(nbins).
    - smooth (str): Whether to smooth the data before passing it to the GMM models ('yes' or 'no').
    - rcut (float): Cutoff radius in Angstroms.
    - trj_step (int): Step used to sample the geometries.
    - cut_type (str): Cutoff type ('hard' or 'soft').
    - over_thres (float): Overlap threshold to include auxiliary functions.
    - rtype (int): Type of radial.
    - alpha (int): Scaling factor for the standard deviation of the Gaussian used for displacement.
    - beta (float): Scaling factor for the standard deviation of the Gaussian used to determine the overlap.
                    of the binary gaussians in the center of the cluster.
    - new_format (logic): use old (deprecated) or new format to store the ACSF angular parameters.

    Note: 
    Currently, auxiliary or decomposed functions are not available when using displaced radial ACSFs.

    This function signifies a modification of the standard radial selector,
    introducing a novel approach where two binary Gaussian functions are generated for each cluster.
    This method accurately delineates the upper and lower bounds of the Gaussian functions,
    without imposing artificial symmetries.

    Returns:
    None
    """

    # Initialize some variables
    print(" # Spatial distribution strategy    = binary")
    if bw is None: bw=50/float(nbins)
    ndecomp=0         
    aux="no"         
    gmm_sub=None
    aux_gaussians=None

    # Extract the geometries from the trjfile
    print(" # Will read data from ", trjname, " every ", str(trj_step), " steps.")
    print(" # Cutoff radius                    =", rcut)
    print(" # Cutoff type                      =", cut_type)
    print(" # Auxiliar functions               =", aux)
    if (aux == "yes"):
       print(" # Overlap threshold for auxiliars  =", over_thres)
    print(" # Data smoothing                   =", smooth)
    if (smooth == "yes"):
       print(" # BW for data smoothing            =", bw)
    print(" # Maximum number of GMM components =", nmax)
    print(" # Maximum number of GMM iterations =", max_iter)
    print(" # GMM sub-components of a GMM      =", ndecomp)
    print(" # Number of bins for 2D histograms =", nbins)
    print(" # ALPHA                            =", alpha)
    print(" # BETA                             =", beta)
    print(" ")
    geometries = read_xyz_trajectory_wstep(trjname,sampling_step=trj_step)
    
    # Compute the radial distribution
    print(" # Computing the distance distribution........START!")
    symbols,dist_dists=compute_distance_distribution(geometries)
    clean_files('radial')
    if not new_format: create_input_type(symbols)
    print(" # Computing the distance distribution........DONE!")
    max_rad=0
    for idx1 in np.arange(0,len(symbols)):
        for idx2 in np.arange(idx1,len(symbols)):
            elemi=str(symbols[idx1])
            elemj=str(symbols[idx2])
            print("---------------------------------------")
            print(" #          "+elemi+elemj+"          # ")
            print("---------------------------------------")
            dist=dist_dists[elemi][elemj]
            # Discard null values (appearing for same element distribution)
            mask = dist != 0
            dist=dist[mask]
            print(" # Discarding null values from distance distribution.")
            if len(dist) == 0 :
               print(" # Not enough data to perform the GMM clustering.")
               continue
            # Parse the distribution to 2D data:
            x,y=extract_distribution(dist, nbins)
            x= np.reshape(x, (-1, 1))
            y= np.reshape(y, (-1, 1))
            xydist=np.hstack((x, y))
            # Apply the cutoff function
            print(" # Applying cutoff function.")
            xydist_cut=apply_cut(xydist,rcut,cut_type=cut_type)
            # Smooth the data (if desired)
            if (smooth == "yes"):
               print(" # Applying a KDE smoothing to the distribution.")
               xydist_cut_kde,samples=smooth_data(xydist_cut,bw)
               xydist_cut_kde=normalize(xydist_cut_kde)
            # Normalize the distribution
            print(" # Normalizing the distribution.")
            xydist=normalize(xydist)
            xydist_cut=normalize(xydist_cut)
            # Recreate raw data from the distribution
            if (smooth == "yes"):
               dist_dat=xy2raw(xydist_cut_kde)
            else:
               dist_dat=xy2raw(xydist_cut)
            # Find the best number of GMM radial components 
            print(" # Optimizing GMM models..............")
            n_comp=find_opt_components(dist_dat,nmax,max_iter)
            print(" # Ideal number of GMM components :",n_comp)
            # Find the parameters of the best GMM fitting
            gmm,g_means,g_covariances,g_weights,sum_gau,grid=find_gmm_params(dist_dat,n_comp,max_iter,nbins)
            # Create the binary gaussians
            gmm=binary_gaus(gmm, alpha=alpha,beta=beta)
            print(" # Parameters of the individual components of the GMM model :")
            print(" ")
            gmm_sub=print_gmm_components(gmm,decomp=ndecomp)
            out_title=elemi+"-"+elemj+" radial distribution"
            # Add auxiliar functions (optional)
            if (aux == "yes"):  
               aux_gaussians = compute_auxiliar_gaussians(gmm, (0,rcut),over_thres)
               for t in range(len(aux_gaussians)):
                   val=aux_gaussians[t]
                   mean=val[0]
                   variance=val[1]**2
                   print(f"    # Auxiliar component {t}: ")
                   print("       -mean      : "+ ' '.join([f'{mean:.6f}'.center(10+7)]))
                   print("       -variance  : "+ ' '.join([f'{variance:.6f}'.center(10+7) ]))
               outname=str(elemi)+str(elemj)+"_radial_aux.png"
               plot_aux(rcut=rcut,gmm=gmm,aux_gaussians=aux_gaussians,file=outname,title=out_title)
            # Plot the radial distributions
            if (smooth == "yes"):
               outname=str(elemi)+str(elemj)+"_radial_dist_smooth.png"
               plot_gmm_components(gmm,samples.reshape(-1,1),nbins,title=out_title,file=outname)
            else:
               outname=str(elemi)+str(elemj)+"_radial_dist.png"
               plot_gmm_components(gmm,dist.reshape(-1,1),nbins,title=out_title,file=outname)
            print(" # 2D representation of GMM components saved to ", outname)
            # Create the ACSF radial features
            outname=str(elemi)+str(elemj)+".rad"
            rad_acsf=save_radial_acsf_params(gmm=gmm,gmm_sub=gmm_sub,aux=aux_gaussians,file=outname)
            if len(rad_acsf) > max_rad: max_rad = len(rad_acsf)
            print(" # Radial ACSF parameters saved to              ", outname)
            if new_format:
               create_input_rad_new_format(elemi,elemj,symbols,rad_acsf)
            else:
               create_input_rad(elemi,elemj,symbols,rad_acsf)
    #Create the header of the input.rad file:
    if new_format:
       create_input_rad_header_new_format(rtype,rcut)
    else:
       create_input_rad_header(rtype,rcut,max_rad)
    return None

def radial_selector_even(trjname=None, nbins=1000, nmax=15, max_iter=10000, bw=None, smooth='no',
                                             rcut=7.0, trj_step=1, cut_type='hard', ndecomp=4, over_thres=0.005,
                                             aux='no', rtype=1, new_format=True):
    """
    This function performs a Gaussian selection with uniformly distributed Gaussians in space.

    Parameters:
    - trjname (str): Path to the trajectory file (XYZ format in Angstroms).
    - nbins (int): Number of bins for histograms.
    - nmax (int): Maximum number of Gaussian functions to look for.
    - max_iter (int): Maximum number of iterations for GMM.
    - bw (float, optional): Bandwidth for smoothing; if None, calculated as 50/float(nbins).
    - smooth (str): Whether to smooth the data before passing it to the GMM models ('yes' or 'no').
    - rcut (float): Cutoff radius in Angstroms.
    - trj_step (int): Step used to sample the geometries.
    - cut_type (str): Cutoff type ('hard' or 'soft').
    - ndecomp (int): Number of Gaussians in which each GMM component will be further decomposed.
    - over_thres (float): Overlap threshold to include auxiliary functions.
    - aux (str): Include auxiliary functions ('yes' or 'no').
    - rtype (int): Type of radial.
    - new_format (logic): use old (deprecated) or new format to store the ACSF angular parameters.

    Note:

    This function represents a version that employs uniformly distributed Gaussian functions in space.

    Returns:
    None
    """
    # Initialize some variables
    print(" # Spatial distribution strategy    = even")
    if bw is None: bw=50/float(nbins)
    gmm_sub=None
    aux_gaussians=None
    # Extract the geometries from the trjfile
    print(" # Will read data from ", trjname, " every ", str(trj_step), " steps.")
    print(" # Cutoff radius                    =", rcut)
    print(" # Cutoff type                      =", cut_type)
    print(" # Auxiliar functions               =", aux)
    if (aux == "yes"):
       print(" # Overlap threshold for auxiliars  =", over_thres)
    print(" # Data smoothing                   =", smooth)
    if (smooth == "yes"):
       print(" # BW for data smoothing            =", bw)
    print(" # Maximum number of GMM components =", nmax)
    print(" # Maximum number of GMM iterations =", max_iter)
    print(" # GMM sub-components of a GMM      =", ndecomp)
    print(" # Number of bins for 2D histograms =", nbins)
    print(" ")
    geometries = read_xyz_trajectory_wstep(trjname,sampling_step=trj_step)
    
    # Compute the radial distribution
    print(" # Computing the distance distribution........START!")
    symbols,dist_dists=compute_distance_distribution(geometries)
    clean_files('radial')
    if not new_format: create_input_type(symbols)
    print(" # Computing the distance distribution........DONE!")
    max_rad=0
    for idx1 in np.arange(0,len(symbols)):
        for idx2 in np.arange(idx1,len(symbols)):
            elemi=str(symbols[idx1])
            elemj=str(symbols[idx2])
            print("---------------------------------------")
            print(" #          "+elemi+elemj+"          # ")
            print("---------------------------------------")
            dist=dist_dists[elemi][elemj]
            # Discard null values (appearing for same element distribution)
            mask = dist != 0
            dist=dist[mask]
            print(" # Discarding null values from distance distribution.")
            if len(dist) == 0 :
               print(" # Not enough data to perform the GMM clustering.")
               continue
            # Parse the distribution to 2D data:
            x,y=extract_distribution(dist, nbins)
            x= np.reshape(x, (-1, 1))
            y= np.reshape(y, (-1, 1))
            xydist=np.hstack((x, y))
            # Apply the cutoff function
            print(" # Applying cutoff function.")
            xydist_cut=apply_cut(xydist,rcut,cut_type=cut_type)
            # Smooth the data (if desired)
            if (smooth == "yes"):
               print(" # Applying a KDE smoothing to the distribution.")
               xydist_cut_kde,samples=smooth_data(xydist_cut,bw)
               xydist_cut_kde=normalize(xydist_cut_kde)
            # Normalize the distribution
            print(" # Normalizing the distribution.")
            xydist=normalize(xydist)
            xydist_cut=normalize(xydist_cut)
            # Recreate raw data from the distribution
            if (smooth == "yes"):
               dist_dat=xy2raw(xydist_cut_kde)
            else:
               dist_dat=xy2raw(xydist_cut)
            # Find the best number of GMM radial components
            print(" # Optimizing GMM models..............")
            n_comp=find_opt_components(dist_dat,nmax,max_iter)
            print(" # Ideal number of GMM components :",n_comp)
            # Find the parameters of the best GMM fitting
            gmm,g_means,g_covariances,g_weights,sum_gau,grid=find_gmm_params(dist_dat,n_comp,max_iter,nbins)
            gmm_sub=print_gmm_components(gmm,decomp=ndecomp)
            # Add auxiliar functions (optional)
            if (aux == "yes"):  
               aux_gaussians = compute_auxiliar_gaussians(gmm, (0,rcut),over_thres)
            # Create the ACSF radial features (evenly spaced)
            if (gmm_sub is None):
               num_rad_acsf=len(g_means)
            else:
               num_rad_acsf=len(gmm_sub)
            if (aux_gaussians is not None): num_rad_acsf += len(aux_gaussians)
            # Create the evenly-sampled gaussians 
            rad_mean,rad_sdev=even_gaussians(rcut, num_rad_acsf)
            rad_acsf=[]
            for count in range(len(rad_sdev)):
                rad_acsf.append([rad_mean[count],1/(2*rad_sdev[count]**2)])
            if num_rad_acsf > max_rad: max_rad = num_rad_acsf
            if new_format:
               create_input_rad_new_format(elemi,elemj,symbols,rad_acsf)
            else:
               create_input_rad(elemi,elemj,symbols,rad_acsf)
    #Create the header of the input.rad file:
    if new_format:
       create_input_rad_header_new_format(rtype,rcut)
    else:
       create_input_rad_header(rtype,rcut,max_rad)
    return None
