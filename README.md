# MM2SF
This repository gathers the MM2SF code along with some instructions and readme files.

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

MM2SF can be easily installed using the pip Python package manager:

    pip install git+https://github.com/m-gallegos/MM2SF.git

Alternatively, one can download the zip file from the MM2SF GitHub and run the following command:

    pip install MM2SSF-main.zip

# Execution

MM2SF can be directly executed from the command line. 

# References

**[1]** J. Behler , The Journal of Chemical Physics, 134, 074106 (2011).

**[2]** J. Behler and M. Parrinello, Physical Review Letters, 98, 146401 (2007).

**[3]** J. S. Smith, O. Isayev and A. E. Roitberg, Chemical Science, 8, 3192–3203 (2017).
