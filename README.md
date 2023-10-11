# MM2SF
This repository gathers the MM2SF code along with some instructions and readme files.

MM2SF is a simple code to create an optimum collection of Atom Centered Symmetry Functions (ACSFs) **[1-2]** for a chemical system. The code uses a Gaussian Mixture Model (GMM) to decompose the characteristic chemical space of a molecule, as provided by a molecular dynamics simulation or normal mode sampling, into well-defined clusters. Then the parameters of the symmetry functions are automatically selected to accurately describe each of the latter domains of the chemical space. Currently, the code is designed to explore, solely, the radial and angular landscapes, resulting in two-body and three-body symmetry functions.



# References

**[1]** J. Behler , The Journal of Chemical Physics, 134, 074106 (2011).

**[2]** J. Behler and M. Parrinello, Physical Review Letters, 98, 146401 (2007).
