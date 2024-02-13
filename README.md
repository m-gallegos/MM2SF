# MM2SF
This repository gathers the MM2SF code along with some instructions and README files.

MM2SF is a simple code to create an optimum collection of Atom Centered Symmetry Functions (ACSFs) **[1-2]** for a chemical system. The code uses a Gaussian Mixture Model (GMM) to decompose the characteristic chemical space of a molecule, as provided by a molecular dynamics simulation or normal mode sampling, into well-defined clusters. Then the parameters of the symmetry functions are automatically selected to accurately describe each of the latter domains of the chemical space. Currently, the code is designed to explore, solely, the radial and angular landscapes, resulting in two-body,

```math
G^{rad}_{i} = \sum^{N}_{j \ne i} e^{-\eta(r_{ij}-r_{s})^{2}} \cdot fc(r_{ij}),
```

and three-body symmetry functions,

```math
G^{ang}_{i} = 2^{1-\xi} \sum^{N}_{j,k \ne i} (1 + cos(\theta_{ijk}-\theta_{s}))^{\xi}
\cdot exp \left [ -\eta \left ( \frac{r_{ij} + r_{ik}}{2} -r_s\right )^2 \right ] \cdot f_c(r_{ij}) \cdot f_c(r_{ik}),
```

in the particular case of the latter, a heavily modified functional form **[3]** is used.

# Installation

To avoid incompatibilities, it is generally recommended to work on Python environments:

    python -m venv venv
    source /venv/bin/activate  (Linux) or venv\Scripts\activate      (Windows)

MM2SF can be easily installed using the pip Python package manager:

    pip install git+https://github.com/m-gallegos/MM2SF.git

Alternatively, one can download the zip file from the MM2SF GitHub and run the following command:

    pip install MM2SF-main.zip

**Note: MM2SF requires a Python version >=3.7.3, <=3.11.5 and one of the latest pip package managers (at least 23.3.2).**

# Execution

MM2SF is designed for seamless execution from the command line, necessitating only a trajectory file containing XYZ Cartesian Coordinates in Angstroms. To ensure an accurate selection of ACSF parameters, it is crucial that the trajectory file reflects a comprehensive sampling of the potential energy landscape within the target systems.

For this purpose, employing Molecular Dynamics (MD) or Normal Mode Sampling (NMS) proves highly beneficial. In this context, conducting simulations at relatively elevated temperatures is generally recommended to facilitate a representative exploration of the conformational space. This approach enhances the reliability of ACSF parameter selection, contributing to the overall effectiveness of MM2SF in capturing the intricate dynamics of the molecular system.

## Self-Optimization of Radial ACSFs

Here, we present an illustrative example showcasing the straightforward utilization of MM2SF to systematically explore the radial space within a molecular framework. This example demonstrates how MM2SF can be effortlessly employed to construct a comprehensive collection of radial symmetry functions, effectively describing the intricate spatial characteristics of the molecular system.

First, the MM2SF package, along with its radial module, must be imported:

    import MM2SF as mm2sf
    from MM2SF.radial import *

Then the radial selector is invoked. As previously mentioned, this process necessitates a trajectory file that consolidates XYZ Cartesian coordinates from the various geometries explored during the sampling.

### Tailor-made distribution
    
    mm2sf_radial.radial_selector_tailormade(trjname="./alanine_capped_AMBER_traj_500K.xyz",nbins=1000, 
                            nmax=15,max_iter=10000,bw=None,smooth='no',rcut=7.0,trj_step=100, 
                            cut_type='hard',ndecomp=2,over_thres=0.005,aux="yes",new_format=True)
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
    - new_format (logic): use old (deprecated) or new format to store the ACSF angular parameters.


### Displaced distribution

    mm2sf_radial.radial_selector_displaced(trjname="./alanine_capped_AMBER_traj_500K.xyz", nbins=1000, 
                            nmax=15, max_iter=10000, bw=None, smooth='no',rcut=7.0, trj_step=1, 
                            cut_type='hard', over_thres=0.005,sigma_scale=3,new_format=True)

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
    - sigma_scale (int): Scaling factor for the standard deviation of the Gaussian used during the
                         displacement of the radial ACSF terms.
    - new_format (logic): use old (deprecated) or new format to store the ACSF angular parameters.


This function presents a customized iteration of the conventional radial selector. In this adaptation, a deliberate displacement is introduced to Gaussian functions, strategically addressing the inherent symmetry challenge posed by Gaussian functions centered within clusters. The modification aims to enhance the effectiveness of the radial selector in capturing nuanced features of the atomic environment.

### Binary distribution
                            
    mm2sf_radial.radial_selector_binary(trjname="./alanine_capped_AMBER_traj_500K.xyz", nbins=1000, 
                            nmax=15, max_iter=10000, bw=None, smooth='no',rcut=7.0, trj_step=1, 
                            cut_type='hard', over_thres=0.005, alpha=3, beta=0.5,new_format=True)
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
    - alpha (int): Scaling factor for the standard deviation of the Gaussian used for displacement.
    - beta (float): Scaling factor for the standard deviation of the Gaussian used to determine the overlap.
                    of the binary gaussians in the center of the cluster.
    - new_format (logic): use old (deprecated) or new format to store the ACSF angular parameters.                   

This function signifies a deviation from the standard radial selector, introducing a novel approach wherein two binary Gaussian functions are generated for each cluster. This innovative method precisely delineates the upper and lower bounds of the Gaussian functions, eliminating the imposition of artificial symmetries and providing a more accurate representation of the atomic environment.
    
### Even distribution
                            
    mm2sf_radial.radial_selector_even(trjname="./alanine_capped_AMBER_traj_500K.xyz", nbins=1000, 
                            nmax=15, max_iter=10000, bw=None, smooth='no',rcut=7.0, trj_step=1, 
                            cut_type='hard', ndecomp=4, over_thres=0.005, aux='no',new_format=True)
  
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
    - new_format (logic): use old (deprecated) or new format to store the ACSF angular parameters.
    
    
This function represents a version that employs uniformly distributed Gaussian functions in space (the same number as that corresponding to the tailor-made functions with a given number of decompositions and auxiliary functions (optional) will be uniformly spread within the cutoff radius). 

## Self-Optimization of Angular ACSFs

Here, we present an illustrative example showcasing the straightforward utilization of MM2SF to systematically explore the angular space within a molecular framework. This example demonstrates how MM2SF can be effortlessly employed to construct a comprehensive collection of angular symmetry functions, effectively describing the intricate spatial characteristics of the molecular system. 

First, the MM2SF package, along with its angular module, must be imported:

    import MM2SF as mm2sf
    from MM2SF.angular import *

Then the angular selector is invoked. As previously mentioned, this process necessitates a trajectory file that consolidates XYZ Cartesian coordinates from the various geometries explored during the sampling. Currently, only the standard (tailor-made) distribution scheme is implemented.

### Tailor-made distribution

    mm2sf_angular.angular_selector(trjname="./alanine_capped_AMBER_traj_500K.xyz", rcut=3.5, nbins=500, trj_step=250, nmax=20, max_iter=100000000,
                 cv_type='full', gmm_crit="bicconv", afrac=0.75, percbic=30, percdiff=40,new_format=True)

    Parameters:
    - trjname (str): Path to the trajectory file (XYZ format in Angstroms).
    - rcut (float): Cutoff radius (in Angstroms).
    - nbins (int): Number of bins for 2D histograms.
    - trj_step (int): Step used to sample the geometries.
    - nmax (int): Maximum number of Gaussian functions to look for.
    - max_iter (int): Maximum number of iterations for GMM.
    - cv_type (str): GMM covariance type.
    - gmm_crit (str): Criterion employed for selecting the number of clusters ('bicmin' or 'bicconv').
    - afrac (float): Percentage of angular functions to take (0 to 1).
    - percbic (float): BIC score percentile (required if gmm_crit is "bicconv").
    - percdiff (float): BIC diff percentile (required if gmm_crit is "bicconv").
    - new_format (logic): use old (deprecated) or new format to store the ACSF angular parameters.

## Computation of the optimized ACSF features


In addition to the optimization of ACSF parameters, MM2SF incorporates a built-in module for computing the final ACSF descriptors of a specified molecule or a collection of molecules. This functionality relies on an input file that consolidates the concatenated XYZ Cartesian coordinates of the target molecules for which ACSF features will be computed. It is worth noting that this input file can include additional columns corresponding to atomic properties. If such columns are present in the initial XYZ file, MM2SF seamlessly incorporates them into the final output files. This capability significantly streamlines the creation of final databases for practical machine learning purposes.

    8
    # XYZ coordinates of a molecule along with atomic properties
    N      -1.251576605435      -1.083935829497       0.026316267168  prop1
    C      -1.858843050383       0.202786044207       0.047861361978  prop2
    N      -1.007350896530       1.334075589503      -0.059527127050  prop3
    C       0.154282232245      -1.366733347294       0.006918469122  prop4
    C       0.890689092262      -0.199589588701      -0.065892436565  prop5
    C       0.358885284422       1.111356713435       0.056261088273  prop6
    N       2.272815954059       0.013059813526      -0.088651269865  prop7
    C       2.493157268062       1.373499159718       0.109470989373  prop8


First, the MM2SF package, along with its acsf computation module, must be imported:

    import MM2SF as mm2sf
    from MM2SF.acsf import *
    
Then the ACSF features can be computed by calling the built-in compute_ACSF function. The latter requires the XYZ file along with the input.rad and input.ang files (gathering the optimized ACSF parameters, previously generated by MM2SF).

    mm2sf_sfc.compute_ACSF(trjname="./alanine_capped_AMBER_traj_500K.xyz",irad="./input.rad",iang="./input.ang",new_format=True)    
    
    Parameters:
    - trjname (str): Name of the trajectory file with .xyz extension (as XYZ coordinates).
    - irad (str): path to the input.rad file (file specifying the parameters for 
                  constructing radial symmetry functions.)
    - iang (str): path to the input.and file (file specifying the parameters for 
                  constructing angular symmetry functions.)
    - new_format (logic): use old (deprecated) or new formats for the creation of the ACSF files.
                  When using the old format, the input.type file must be provided.

Once executed, the code will produce two files gathering the same information but using slightly different formattings:

    _acsf_nn.out: atom chemical identity, atomic properties in order (if any), ACSF features.
    _acsf_GP.out (it uses an extended XYZ format): for each molecule, the following format is used:
    22 # Number of atoms
       # Blank line
    O -1.284000 -1.369000 -0.514000 None 0.000000 0.000000 0.000011 0.020075 .... (Atom identity, XZY coordinates, Atomic properties and ACSF features).
    
Finally, it is worth noting that MM2SF also accounts for an old formatting (deprecated) for the computation of the ACSF features. Such a formatting relies on unique integer identifiers to refer to the atomic pairs or atomic trios. If desired, the ACSF features can be computed with this deprecated format as:

    mm2sf_sfc.compute_ACSF(trjname="./alanine_capped_AMBER_traj_500K.xyz",itype="./input.type",irad="./input.rad",iang="./input.ang",new_format=False)

    -Note : requires an additional file (input.type), generated by MM2SF if selected new_format=False during the optimization of the radial and angular features.

Notice that the input.rad and input.ang files used in this case have to be generated by setting the **new_format=False** flag during the exploration of the radial and angular spaces. Both the current and deprecated formats will produce exactly the same features so the use of the latter is generally discouraged for the general user.

# References

**[1]** J. Behler , The Journal of Chemical Physics, 134, 074106 (2011).

**[2]** J. Behler and M. Parrinello, Physical Review Letters, 98, 146401 (2007).

**[3]** J. S. Smith, O. Isayev and A. E. Roitberg, Chemical Science, 8, 3192–3203 (2017).
