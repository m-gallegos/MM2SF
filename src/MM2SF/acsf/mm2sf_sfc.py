import numpy as np
import sys as sys
import os
from MM2SF.acsf.sfc_functions import *

def compute_ACSF(trjname=None,itype=None,irad=None,iang=None,new_format=True):
    """
    Compute ACSF (Atom-Centered Symmetry Functions) based on input geometry files.

    Parameters:
    - trjname (str): Name of the trajectory file with .xyz extension (as XYZ coordinates).
    - itype (str): path to the input.type file (file containing information about the 
                  number and types of different chemical elements in the database).
    - irad (str): path to the input.rad file (file specifying the parameters for 
                  constructing radial symmetry functions.)
    - iang (str): path to the input.and file (file specifying the parameters for 
                  constructing angular symmetry functions.)
    - new_format (logic): use old (deprecated) or new formats for the creation of the ACSF files.
                  When using the old format, the input.type file must be provided.

    Notes:
    - The program requires input files (input.type (optional), input.rad, input.ang) to run.
    - Each geometry file (*.xyz) should contain the number of atoms on the first line,
      followed by the x, y, z coordinates (in angstroms) of the geometry. Additionally,
      properties can be appended as additional columns.
    - The value of theta_s is read in degrees and converted to radians.
    """
    print(" # Will read molecular data from       ", trjname)
    print(" # Will read radial    data from       ", irad)
    print(" # Will read angular   data from       ", iang)
    if not new_format:   # Old format
       print(" # Will read type      data from       ", itype)
       # Read input.type
       telem,tipo = read_itype(itype)
       # Read input.rad
       type_rad,rcut_rad,radmax,rs_rad,eta_rad,nrad = read_irad(irad,telem)
       # Read input.ang
       indjk,vecino = generate_indjk(telem)
       type_ang, rcut_ang, angmax, idum, rs_ang, xi_ang, eta_ang, lambda_ang, rs_ang_ij, \
       rs_ang_ik, eta_ang_ij, eta_ang_ik, theta_s, nang = read_iang(iang,telem,indjk,vecino)
       # Clean output files
       outname_gp, outname_nn= clean_out_files(trjname)
       # Compute the ACSF features
       trj2sf(trjname,telem,tipo,type_rad,rcut_rad,radmax,rs_rad,eta_rad,nrad,indjk,vecino,type_ang,rcut_ang, 
             angmax,idum,rs_ang,xi_ang,eta_ang,lambda_ang,rs_ang_ij,rs_ang_ik,eta_ang_ij,eta_ang_ik,theta_s,
             nang,outname_gp,outname_nn)
    else:                # New format
       # Read input.rad
       elements, type_rad,rcut_rad,rs_rad,eta_rad,nrad = read_irad_new_format(irad)
       # Get the ordered pairs
       ordered_pairs= order_pairs(elements)
       # Read input.ang
       pairs, type_ang, rcut_ang, idum, rs_ang, xi_ang, eta_ang, lambda_ang, rs_ang_ij, \
       rs_ang_ik, eta_ang_ij, eta_ang_ik, theta_s, nang = read_iang_new_format(iang)
       # Clean output files
       outname_gp, outname_nn= clean_out_files(trjname)
       # Compute the ACSF features
       trj2sf_new_format(trjname,elements,ordered_pairs,pairs,type_rad,rcut_rad,rs_rad,eta_rad,nrad,type_ang,rcut_ang, 
             idum,rs_ang,xi_ang,eta_ang,lambda_ang,rs_ang_ij,rs_ang_ik,eta_ang_ij,eta_ang_ik,theta_s,
             nang,outname_gp,outname_nn)
    return None
