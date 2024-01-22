import sys as sys
import numpy as np
import itertools
from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import time
from MM2SF.basics.functions import *

def angular_selector(trjname=None, rcut=3.5, nbins=500, trj_step=20, nmax=20, max_iter=100000000,
                      cv_type='full', gmm_crit="bicconv", atype=3, afrac=0.75, percbic=30, percdiff=40,
                      new_format=True):
    """
    This function processes the angular distribution.

    Parameters:
    - trjname (str): Path to the trajectory file (XYZ format in Angstroms).
    - rcut (float): Cutoff radius (in Angstroms).
    - nbins (int): Number of bins for 2D histograms.
    - trj_step (int): Step used to sample the geometries.
    - nmax (int): Maximum number of Gaussian functions to look for.
    - max_iter (int): Maximum number of iterations for GMM.
    - cv_type (str): GMM covariance type.
    - gmm_crit (str): Criterion employed for selecting the number of clusters ('bicmin' or 'bicconv').
    - atype (int): Type of angular ACSF to be employed (3, heavily modified, or 5, pairwise expansion).
    - afrac (float): Percentage of angular functions to take (0 to 1).
    - percbic (float): BIC score percentile (required if gmm_crit is "bicconv").
    - percdiff (float): BIC diff percentile (required if gmm_crit is "bicconv").
    - new_format (logic): use old (deprecated) or new format to store the ACSF angular parameters.

    """
    
    print(" # Spatial distribution strategy    = tailor-made")
    # Extract the geometries from the trjfile
    print(" # Will read data from ", trjname, " every ", str(trj_step), " steps.")
    print(" # Cutoff radius                    =", rcut)
    print(" # Maximum number of GMM components =", nmax)
    print(" # Maximum number of GMM iterations =", max_iter)
    print(" # GMM covariance type              =", cv_type)
    print(" # Optimum GMM criterion            =", gmm_crit)
    if (atype != 3 and atype !=5):
       raise AssertionError("Currently, the angular function types can be 3 and 5, only!")
    print(" # Angular ACSF type                =", atype)

    # Initialize some variables
    if (gmm_crit == "bicconv"):
       try: 
          percbic
       except NameError: percbic=60
       try: 
          percdiff
       except NameError: percdiff=30
       print("  - BIC score percentile            =", percbic)
       print("  - BIC diff percentile             =", percdiff)
    else:
       percbic=None
       percdiff=None
    print(" # Number of bins for 2D histograms =", nbins)
    print(" ")
    geometries = read_xyz_trajectory_wstep(trjname,sampling_step=trj_step)
    start_time=time.time()

    # Compute the angular distribution (accounting for redundancies)
    print(" # Computing the angular distribution........START!")
    symbols,pepe=compute_angles_wijk(geometries)
    print(" # Computing the angular distribution........DONE!")
    end_time=time.time()
    # Get the local neighboring matrix
    telem,indjk,vecino=chemmat(symbols)
    clean_files('angular')
    if not new_format: create_input_type(symbols)  # input.type for the computation of the ACSF features
    # Iterate over all the combinations between elements and atomic pairs
    max_ang=0  # (comes from a deprecated issue of the ACSF generator code)
    for elem, pairs in pepe.items():
        for pair, data in pairs.items():
           if len(data) > 0:
            print("---------------------------------------")
            print(" #           "+elem+pair+"           # ")
            print("---------------------------------------")
            print(" # Number of points before rcut filter ",len(data))
            # Retrieve the data and filter it
            data=filter3Ddata(data,rcut,pair)
            n_samples=len(data)
            print(" # Number of points after rcut filter ",n_samples)
            if (n_samples <= nmax*3):
               print(" # WARNING !: NOT ENOUGH DATA TO RUN THE GMM DECOMPOSITION, THIS COMBINATION WILL BE SKIPPED")
               continue
            # Save the (observed) angular distribution
            outname=str(elem)+str(pair)+"_real_ang_dist.png"
            plot_angular_trios(data,elem,pair,rcut,nbins,cmap_corr='jet',cmap_polar='gnuplot',ps=0.005,file=outname)
            print(" # Real angular distribution saved to ", outname)
            # Normalize the data
            data_norm,data_mean,data_std=norm_3ddata(data)
            print(" # Data normalization................ DONE!")
            # Look for the best number of components
            print(" # Optimizing GMM models..............")
            outname=str(elem)+str(pair)+"_bic_score.png"
            n_comp=find_opt_components3D(data_norm,nmax=nmax,max_iter=max_iter,cv_type=cv_type,crit=gmm_crit,percbic=percbic,percdiff=percdiff,file=outname)
            print(" # Ideal number of GMM components :",n_comp)
            # Fit the GMM to the best number of components
            gmm,g_means,g_covariances,g_weights=find_gmm_params3D(data_norm,n_comp,max_iter,cv_type=cv_type)
            # Print the parameters of the GMM functions in the denormalized scale
            print(" # Parameters of the individual components of the GMM model :")
            print(" ")
            print_3Dgmm_components(gmm,data_std,data_mean)
            # Plot the individual GMMM components in the 3D space
            outname=str(elem)+str(pair)+"_3D_dist.png"
            plot_3Dgmm_components(gmm,data_norm,data_mean,data_std,file=outname)
            print(" # 3D representation of GMM components saved to ", outname)
            # Use the entire GMM to recreate the angular distribution and reconstruct the previous plots
            samples = gmm.sample(n_samples=n_samples)[0]
            samples = samples*data_std + data_mean
            outname=str(elem)+str(pair)+"_gmm_ang_dist.png"
            plot_angular_trios(samples,elem,pair,rcut,nbins,cmap_corr='jet',cmap_polar='gnuplot',ps=0.005,file=outname)
            print(" # GMM estimated angular distribution saved to  ", outname)
            # Save the angular ACSFs to a file:
            outname=str(elem)+str(pair)+".ang"
            # Note: the arguments to be parsed depend on the actual functional form of the angular ACSF kernel
            ang_acsf=save_angular_acsf_params(gmm=gmm,mean=data_mean,std=data_std,file=outname,atype=atype,frac=afrac)
            if len(ang_acsf) > max_ang: max_ang = len(ang_acsf)
            print(" # Angular ACSF parameters saved to              ", outname)
            # Plot the angular ACSFs and their activations (only available for cosine-based kernel)
            sym_count=0
            for sym in ang_acsf:
                sym_count +=1
                outname=str(elem)+str(pair)+"ang_"+str(sym_count)+".png"
                outtitle=str(elem)+str(pair)+" ACSF "+str(sym_count)
                plot_angular_acsf(atype=atype,rcut=rcut,grad=200,gang=250,rs_ang=sym[0],xi_ang=sym[1],eta_ang=sym[2],lambda_ang=None,theta_ang=sym[3],file=outname,title=outtitle)
                outname=str(elem)+str(pair)+"ang_"+str(sym_count)+"_act.png"
                plot_angular_activations(data,elem,pair,atype=atype,rcut=rcut,rs_ang=sym[0],xi_ang=sym[1],eta_ang=sym[2],lambda_ang=None,theta_ang=sym[3],file=outname,title=outtitle,ps=0.05,cmap_polar='Blues')
            if new_format: 
               create_input_ang_new_format(elem,pair,symbols,ang_acsf,atype)
            else: 
               create_input_ang(elem,pair,symbols,ang_acsf,atype)
    #Create the header of the input.ang file:
    if new_format:
       create_input_ang_header_new_format(atype,rcut)
    else: 
       create_input_ang_header(atype,rcut,max_ang)
    return None
