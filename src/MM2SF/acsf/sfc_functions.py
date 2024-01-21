import numpy as np
import sys as sys
import os

def order_pairs(elements):
   ordered_pairs=[]
   for l in elements:
       for k in elements:
           if (k >= l):
               ordered_pairs.append(l+k)
   return ordered_pairs

def flatten_dict(dictionary):
    flattened_list = []
    for key, value in dictionary.items():
        if isinstance(value, dict):
            flattened_list.extend(flatten_dict(value))
        else:
            flattened_list.append(value)
    return flattened_list

def append2GPR(outname,name,label,coord,prop,acsf):
    """
    Append data to the output ACSF file for Gaussian Process Regression (GPR) training.

    Parameters:
    - outname (str): The name of the output file to which data will be appended.
    - name (str): Name or identifier associated with the data.
    - label (List): List of atom labels.
    - coord (numpy.ndarray): Array of shape (n_atoms, 3) representing atomic coordinates.
    - prop (List): List of properties associated with each atom.
    - acsf (List): List of Atomic Clustered Support Functions (ACSF) associated with each atom.

    Returns:
    - None: The function appends the data to the specified output file.
    """
    with open(outname,'a+') as f:
         natoms=len(label)
         f.write(str(natoms)+"\n")
         f.write(name)
         #if isinstance(prop[0][0], float):
         if prop[0] is not None:
            for i in np.arange(0,natoms):
                #f.write(f"{label[i]} {coord[i,0]:.6f} {coord[i,1]:.6f} {coord[i,2]:.6f} {prop[i]:.8f} ")
                f.write(f"{label[i]} {coord[i,0]:.6f} {coord[i,1]:.6f} {coord[i,2]:.6f} ")
                f.write(" ".join([f"{value:.8f}" for value in prop[i]]))
                f.write(" ")
                f.write(" ".join([f"{value:.6f}" for value in acsf[i]]))
                f.write("\n")
         else:
            for i in np.arange(0,natoms):
                f.write(f"{label[i]} {coord[i,0]:.6f} {coord[i,1]:.6f} {coord[i,2]:.6f} {prop[i]} ")
                f.write(" ".join([f"{value:.6f}" for value in acsf[i]]))
                f.write("\n")
    return None

def append2NN(outname,label,prop,acsf):
    """
    Append data to the output ACSF file for Neural Network (NN) training.

    Parameters:
    - outname (str): The name of the output file to which data will be appended.
    - label (List): List of labels associated with the data.
    - prop (List): List of properties associated with each data point.
    - acsf (List): List of Atomic Clustered Support Functions (ACSF) associated with each data point.

    Returns:
    - None: The function appends the data to the specified output file.
    """
    natoms=len(label)
    with open(outname,'a+') as f:
         #if isinstance(prop[0][0], float):
         if prop[0] is not None:
            for i in np.arange(0,natoms):
                #f.write(f"{label[i]},{prop[i]:.8f}, ")
                f.write(f"{label[i]},")
                f.write(",".join([f"{value:.8f}" for value in prop[i]]))
                f.write(", ")
                f.write(",".join([f"{value:.6f}" for value in acsf[i]]))
                f.write("\n")
         else:
            for i in np.arange(0,natoms):
                f.write(f"{label[i]},{prop[i]}, ")
                f.write(",".join([f"{value:.6f}" for value in acsf[i]]))
                f.write("\n")
    return None
    
def intpos(vec, nele, val):
    for i in range(nele):
        if vec[i] == val:
            return i + 1
    return 1

def chemclas(eti, telem, tipo):
    clas = 0
    for i in range(telem):
        if eti == tipo[i]:
            clas = i + 1
            break
    if clas == 0:
        print(f"Fatal Error: {eti} not recognized as a known element.")
        raise SystemExit("Error Termination")
    return clas

def distance(atomi, atomj):
    dist = 0.0
    for i in range(3):
        dist += (atomj[i] - atomi[i])**2
    dist = np.sqrt(dist)
    return dist

def cutoff(cut, dist):
    pi = np.pi
    if dist > cut:
        fcut = 0.0
    elif dist <= cut:
        fcut = 0.5 * (np.cos(pi * dist / cut) + 1.0)
    return fcut

def dotproduct(vec1, vec2):
    dot = np.dot(vec1, vec2)
    return dot

atomicnumber = {
    "H":   1, "He":  2, "Li":  3, "Be":  4, "B":   5,
    "C":   6, "N":   7, "O":   8, "F":   9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P":  15,
    "S":  16, "Cl": 17, "Ar": 18, "K":  19, "Ca": 20,
    "Sc": 21, "Ti": 22, "V":  23, "Cr": 24, "Mn": 25,
    "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
    "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35,
    "Kr": 36, "Rb": 37, "Sr": 38, "Y":  39, "Zr": 40,
    "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45,
    "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
    "Sb": 51, "Te": 52, "I":  53, "Xe": 54, "Cs": 55,
    "Ba": 56, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60,
    "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65,
    "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70,
    "Lu": 71, "Hf": 72, "Ta": 73, "W":  74, "Re": 75,
    "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80,
    "Tl": 81, "Pb": 82, "Bi": 83, "Th": 90, "Pa": 91,
    "U":  92, "Np": 93, "Pu": 94, "Am": 95, "Cm": 96,
    "Bk": 97, "Cf": 98, "Es": 99, "Fm": 100, "Md": 101,
    "No": 102, "Lr": 103, "Rf": 104, "Db": 105, "Sg": 106,
    "Bh": 107, "Hs": 108, "Mt": 109, "Ds": 110, "Rg": 111,
    "Cn": 112, "Nh": 113, "Fl": 114, "Mc": 115, "Lv": 116,
    "Ts": 117}

def atomfactor(Z):
    factor = Z
    return factor

def pairfactor(Zi, Zj):
    factor = Zi * Zj
    return factor

def read_itype(itype):
    with open(itype, 'r') as f:
        telem = int(f.readline().strip().split()[0]) # Number of different element types
        tipo = [f.readline().strip() for _ in range(telem)] # List of element types
    return telem,tipo

def read_irad(irad,telem):
    """
    Read parameters for radial symmetry functions from the specified file.

    Parameters:
    - file_path (str): Path to the input file containing radial symmetry function parameters (default: "input.rad").

    Returns:
    - type_rad (int): Type of radial symmetry function.
    - rcut_rad (float): Cutoff radius for radial symmetry functions.
    - radmax (int): Maximum number of radial symmetry functions.
    - rs_rad (ndarray): Array of radial symmetry function distances.
    - eta_rad (ndarray): Array of radial symmetry function eta values.
    - nrad (ndarray): Array containing the number of radial symmetry functions for each element pair.
    """
    with open(irad, "r") as f:
        type_rad = int(f.readline().strip().split()[0])
        rcut_rad = float(f.readline().strip().split()[0])
        radmax = int(f.readline().strip().split()[0])

        if type_rad == 1:
            print(' # Type of Radial Symmetry Function    = Normal')
            rs_rad = np.zeros((telem, telem, radmax), dtype=np.float64)
            eta_rad = np.zeros((telem, telem, radmax), dtype=np.float64)
            nrad = np.zeros((telem, telem), dtype=np.int32)

            for i in range(1, telem**2 + 1):
                ni, nj, idum = map(int, f.readline().strip().split()[:3])
                nrad[ni-1, nj-1] = idum

                for k in range(1, idum + 1):
                    rs, eta = map(float, f.readline().strip().split()[:2])
                    rs_rad[ni-1, nj-1, k-1] = rs
                    eta_rad[ni-1, nj-1, k-1] = eta

        elif type_rad == 2:
            print(' # Type of Radial Symmetry Function    = Z-Weighted')
            rs_rad = np.zeros((telem, 1, radmax), dtype=np.float64)
            eta_rad = np.zeros((telem, 1, radmax), dtype=np.float64)
            nrad = np.zeros((telem, 1), dtype=np.int32)

            for i in range(1, telem + 1):
                ni, idum = map(int, f.readline().strip().split()[:2])
                nrad[ni-1, 0] = idum

                for k in range(1, idum + 1):
                    rs, eta = map(float, f.readline().strip().split()[:2])
                    rs_rad[ni-1, 0, k-1] = rs
                    eta_rad[ni-1, 0, k-1] = eta

        else:
            raise ValueError("Unrecognizable Radial Symmetry Function Type")

    return type_rad, rcut_rad, radmax, rs_rad, eta_rad, nrad

def read_irad_new_format(irad):
    """
    Read parameters for radial symmetry functions from the specified file.

    Parameters:
    - file_path (str): Path to the input file containing radial symmetry function parameters (default: "input.rad").

    Returns:
    - type_rad (int): Type of radial symmetry function.
    - rcut_rad (float): Cutoff radius for radial symmetry functions.
    - rs_rad (ndarray): Array of radial symmetry function distances.
    - eta_rad (ndarray): Array of radial symmetry function eta values.
    - nrad (ndarray): Array containing the number of radial symmetry functions for each element pair.
    """

    elements=[]
    with open(irad, "r") as f:
        type_rad = int(f.readline().strip().split()[0])
        rcut_rad = float(f.readline().strip().split()[0])
        if type_rad == 1:
            print(' # Type of Radial Symmetry Function    = Normal')
            # Initialize the dictionaries
            rs_rad  = {}
            eta_rad = {}
            while True:
               line  = f.readline().strip().split()
               if len(line) == 0 : break
               elem_i= line[0]
               elem_j= line[1]
               if (elem_i not in elements):
                  elements.append(elem_i)
                  rs_rad[elem_i] = {}
                  eta_rad[elem_i] = {}
               rs_rad[elem_i][elem_j] = []
               eta_rad[elem_i][elem_j] = []
               nrad  = int(line[2])
               for i in range(nrad):
                   line=f.readline().strip().split()
                   rs_rad[elem_i][elem_j].append(float(line[0]))
                   eta_rad[elem_i][elem_j].append(float(line[1]))
        elif type_rad == 2:
            print(' # Type of Radial Symmetry Function    = Z-Weighted')
            # Initialize the dictionaries
            rs_rad  = {}
            eta_rad = {}
            while True:
               line  = f.readline().strip().split()
               if len(line) == 0 : break
               elem_i= line[0]
               if (elem_i not in elements):
                  elements.append(elem_i)
                  rs_rad[elem_i] = []
                  eta_rad[elem_i] = []
               nrad  = int(line[1])
               for i in range(nrad):
                   line=f.readline().strip().split()
                   rs_rad[elem_i].append(float(line[0]))
                   eta_rad[elem_i].append(float(line[1]))
        else:
            raise ValueError("Unrecognizable Radial Symmetry Function Type")
        
    elements=sorted(elements)
    return elements, type_rad, rcut_rad, rs_rad, eta_rad, nrad


def read_iang(iang,telem,indjk,vecino):
    """
    Read input data from the 'input.ang' file and store relevant information.

    Parameters:
    - telem (int): Number of elements.

    Returns:
    - type_ang (int): Type of Angular Symmetry Function.
    - rcut_ang (float): Cutoff radius for angular functions.
    - angmax (int): Maximum number of distinct angular functions per atom pair.
    - idum (int): Dummy variable used in calculations.
    - rs_ang (ndarray): Array for storing rs values for angular functions.
    - xi_ang (ndarray): Array for storing xi values for angular functions.
    - eta_ang (ndarray): Array for storing eta values for angular functions.
    - lambda_ang (ndarray): Array for storing lambda values for angular functions.
    - theta_s (ndarray): Array for storing theta_s values for angular functions.
    - nang (ndarray): Array for storing the maximum number of angular functions for each atom pair.
    """
  

    # Initialize the variables
    type_ang=None;rcut_ang=None;angmax=None;idum=None;rs_ang=None;xi_ang=None;
    eta_ang=None;lambda_ang=None;rs_ang_ij=None;rs_ang_ik=None;xi_ang=None; 
    eta_ang_ij=None;eta_ang_ik=None;theta_s=None;nang=None
   
    # Read the data
    with open(iang, "r") as f:
         type_ang = int(f.readline().strip().split()[0])  
         rcut_ang = float(f.readline().strip().split()[0]) 
         angmax = int(f.readline().strip().split()[0]) 
         idum = (telem*(telem+1)//2)
         if type_ang == 1:
            print(' # Type of Angular Symmetry Function   = Normal')
            rs_ang = np.zeros((telem, idum, angmax))
            xi_ang = np.zeros((telem, idum, angmax))
            eta_ang = np.zeros((telem, idum, angmax))
            lambda_ang = np.zeros((telem, idum, angmax))
            nang = np.zeros((telem, idum),dtype=int)
            while True:
                try:
                  ni, nj, idum = map(int, f.readline().strip().split()[:3]) 
                  # Estimate the position of the atomic pair in the neighboring matrix
                  pepe = intpos(vecino, telem*(telem+1)//2, nj)-1 
                  # Assign the value of nang maximum for the atomic trio
                  nang[ni-1, pepe] = idum 
                  # Read the rs,xi,eta and lambda
                  for k in range(1, idum + 1):
                      rs_ang[ni-1, pepe, k-1], xi_ang[ni-1, pepe, k-1], eta_ang[ni-1, pepe, k-1], lambda_ang[ni-1, pepe, k-1] = map(float, f.readline().strip().split()[:4]) 
                except StopIteration:
                   break
                except ValueError:
                   break
         elif type_ang == 2:
            print(' # Type of Angular Symmetry Function   = Modified')
            rs_ang = np.zeros((telem, idum, angmax))
            xi_ang = np.zeros((telem, idum, angmax))
            eta_ang = np.zeros((telem, idum, angmax))
            lambda_ang = np.zeros((telem, idum, angmax))
            nang = np.zeros((telem, idum),dtype=int)
            while True:
                try:
                  ni, nj, idum = map(int, f.readline().strip().split()[:3])
                  pepe = intpos(vecino, telem * (telem + 1) // 2, nj)-1
                  nang[ni-1, pepe] = idum
                  for k in range(1, idum + 1):
                      rs_ang[ni-1, pepe, k-1], xi_ang[ni-1, pepe, k-1], eta_ang[ni-1, pepe, k-1], lambda_ang[ni-1, pepe, k-1] = map(float, f.readline().strip().split()[:4])
                except StopIteration:
                   break
                except ValueError:
                   break
         elif type_ang == 3:
             print(' # Type of Angular Symmetry Function   = Heavily Modified')
             rs_ang = np.zeros((telem, idum, angmax))
             xi_ang = np.zeros((telem, idum, angmax))
             eta_ang = np.zeros((telem, idum, angmax))
             theta_s = np.zeros((telem, idum, angmax))
             nang = np.zeros((telem, idum),dtype=int)
             while True:
                try:
                   ni, nj, idum = map(int, f.readline().strip().split()[:3])
                   pepe = intpos(vecino, telem * (telem + 1) // 2, nj)-1
                   nang[ni-1, pepe] = idum
                   for k in range(1, idum + 1):
                       rs_ang[ni-1, pepe, k-1], xi_ang[ni-1, pepe, k-1], eta_ang[ni-1, pepe, k-1], theta_s[ni-1, pepe, k-1] = map(float, f.readline().strip().split()[:4])
                except StopIteration:
                   break
                except ValueError:
                   break
         elif type_ang == 4:
             print(' # Type of Angular Symmetry Function   = Z-Weighted')
             idum = 1
             rs_ang = np.zeros((telem, idum, angmax))
             xi_ang = np.zeros((telem, idum, angmax))
             eta_ang = np.zeros((telem, idum, angmax))
             lambda_ang = np.zeros((telem, idum, angmax))
             nang = np.zeros((telem, idum),dtype=int)
             for i in range(telem):
                ni,  idum = map(int, f.readline().strip().split()[:2])
                nang[ni-1, 0] = idum
                for k in range(1,idum+1):
                     rs_ang[ni-1, 0, k-1], xi_ang[ni-1, 0, k-1], eta_ang[ni-1, 0, k-1], lambda_ang[ni-1, 0, k-1] = map(float, f.readline().strip().split()[:4])
         elif type_ang == 5:
             print(' # Type of Angular Symmetry Function   = Heavily Modified with independent radial grids')
             rs_ang_ij = np.zeros((telem, idum, angmax))
             rs_ang_ik = np.zeros((telem, idum, angmax))
             xi_ang = np.zeros((telem, idum, angmax))
             eta_ang_ij = np.zeros((telem, idum, angmax))
             eta_ang_ik = np.zeros((telem, idum, angmax))
             theta_s = np.zeros((telem, idum, angmax))
             nang = np.zeros((telem, idum),dtype=int)
             while True:
                try:
                   ni, nj, idum = map(int, f.readline().strip().split()[:3])
                   pepe = intpos(vecino, telem * (telem + 1) // 2, nj)-1
                   nang[ni-1, pepe] = idum
                   for k in range(1, idum + 1):
                       rs_ang_ij[ni-1, pepe, k-1], rs_ang_ik[ni-1, pepe, k-1], xi_ang[ni-1, pepe, k-1], eta_ang_ij[ni-1, pepe, k-1],eta_ang_ik[ni-1, pepe, k-1], theta_s[ni-1, pepe, k-1] = map(float, f.readline().strip().split()[:6])
                except StopIteration:
                   break
                except ValueError:
                   break
         else:
             raise ValueError('Unrecognizable Angular Symmetry Function Type')

    return type_ang, rcut_ang, angmax, idum, rs_ang, xi_ang, eta_ang, lambda_ang, rs_ang_ij, rs_ang_ik, eta_ang_ij, eta_ang_ik, theta_s, nang

def read_iang_new_format(iang):
    """
    Read input data from the 'input.ang' file and store relevant information.

    Returns:
    - type_ang (int): Type of Angular Symmetry Function.
    - rcut_ang (float): Cutoff radius for angular functions.
    - idum (int): Dummy variable used in calculations.
    - rs_ang (ndarray): Array for storing rs values for angular functions.
    - xi_ang (ndarray): Array for storing xi values for angular functions.
    - eta_ang (ndarray): Array for storing eta values for angular functions.
    - lambda_ang (ndarray): Array for storing lambda values for angular functions.
    - theta_s (ndarray): Array for storing theta_s values for angular functions.
    - nang (ndarray): Array for storing the maximum number of angular functions for each atom pair.
    """

    # Initialize the variables
    type_ang=None;rcut_ang=None;idum=None;rs_ang=None;xi_ang=None;
    eta_ang=None;lambda_ang=None;rs_ang_ij=None;rs_ang_ik=None;xi_ang=None; 
    eta_ang_ij=None;eta_ang_ik=None;theta_s=None;nang=None
    pairs=[]
    with open(iang, "r") as f:
         # Read type and cutoff radius
         type_ang = int(f.readline().strip().split()[0])  
         rcut_ang = float(f.readline().strip().split()[0]) 
         if type_ang in [1,2] :
            if (type_ang == 1): print(' # Type of Angular Symmetry Function   = Normal')
            if (type_ang == 2): print(" # Type of Angular Symmetry Function   = Modified")
            # Initialize the dictionary
            rs_ang  = {}
            xi_ang  = {}
            eta_ang = {}
            lambda_ang = {}
            while True:
               line=f.readline().strip().split()
               if (len(line) == 0) : break
               elem_i = line[0]
               if elem_i not in rs_ang:
                  rs_ang[elem_i]    ={}
                  xi_ang[elem_i]    ={}
                  eta_ang[elem_i]   ={}
                  lambda_ang[elem_i]={}
               pair   = line[1]
               if pair not in rs_ang[elem_i]: pairs.append(pair)
               rs_ang[elem_i][pair]=[]
               xi_ang[elem_i][pair]=[]
               eta_ang[elem_i][pair]=[]
               lambda_ang[elem_i][pair]=[]
               nang   = int(line[2])
               for i in range(nang):
                   line=f.readline().strip().split()
                   rs_ang[elem_i][pair].append(float(line[0]))
                   xi_ang[elem_i][pair].append(float(line[1]))
                   eta_ang[elem_i][pair].append(float(line[2]))
                   lambda_ang[elem_i][pair].append(float(line[3]))
         elif type_ang == 3:
            print(" # Type of Angular Symmetry Function   = Heavily Modified")
            # Initialize the dictionary
            rs_ang  = {}
            xi_ang  = {}
            eta_ang = {}
            theta_s = {}
            while True:
               line=f.readline().strip().split()
               if (len(line) == 0) : break
               elem_i = line[0]
               if elem_i not in rs_ang:
                  rs_ang[elem_i]    ={}
                  xi_ang[elem_i]    ={}
                  eta_ang[elem_i]   ={}
                  theta_s[elem_i]={}
               pair   = line[1]
               if pair not in rs_ang[elem_i]: pairs.append(pair)
               rs_ang[elem_i][pair]=[]
               xi_ang[elem_i][pair]=[]
               eta_ang[elem_i][pair]=[]
               theta_s[elem_i][pair]=[]
               nang   = int(line[2])
               for i in range(nang):
                   line=f.readline().strip().split()
                   rs_ang[elem_i][pair].append(float(line[0]))
                   xi_ang[elem_i][pair].append(float(line[1]))
                   eta_ang[elem_i][pair].append(float(line[2]))
                   theta_s[elem_i][pair].append(float(line[3]))
         elif type_ang == 4:
            print(" # Type of Angular Symmetry Function   = Z-Weighted")
            # Initialize the dictionary
            rs_ang  = {}
            xi_ang  = {}
            eta_ang = {}
            lambda_ang = {}
            while True:
               line=f.readline().strip().split()
               if (len(line) == 0) : break
               elem_i = line[0]
               if elem_i not in rs_ang:
                  rs_ang[elem_i]    =[]
                  xi_ang[elem_i]    =[]
                  eta_ang[elem_i]   =[]
                  lambda_ang[elem_i]=[]
               nang   = int(line[1])
               for i in range(nang):
                   line=f.readline().strip().split()
                   rs_ang[elem_i].append(float(line[0]))
                   xi_ang[elem_i].append(float(line[1]))
                   eta_ang[elem_i].append(float(line[2]))
                   lambda_ang[elem_i].append(float(line[3]))
         elif type_ang == 5:
            print(" # Type of Angular Symmetry Function   = Heavily Modified with independent radial grids")
            # Initialize the dictionaries
            rs_ang_ij  = {}
            rs_ang_ik  = {}
            xi_ang  = {}
            eta_ang_ij = {}
            eta_ang_ik = {}
            theta_s = {}
            while True:
               line=f.readline().strip().split()
               if (len(line) == 0) : break
               elem_i = line[0]
               if elem_i not in rs_ang_ij:
                  rs_ang_ij[elem_i]    ={}
                  rs_ang_ik[elem_i]    ={}
                  xi_ang[elem_i]    ={}
                  eta_ang_ij[elem_i]   ={}
                  eta_ang_ik[elem_i]   ={}
                  theta_s[elem_i]={}
               pair   = line[1]
               if pair not in rs_ang_ij[elem_i]: pairs.append(pair)
               rs_ang_ij[elem_i][pair]=[]
               rs_ang_ik[elem_i][pair]=[]
               xi_ang[elem_i][pair]=[]
               eta_ang_ij[elem_i][pair]=[]
               eta_ang_ik[elem_i][pair]=[]
               theta_s[elem_i][pair]=[]
               nang   = int(line[2])
               for i in range(nang):
                   line=f.readline().strip().split()
                   rs_ang_ij[elem_i][pair].append(float(line[0]))
                   rs_ang_ik[elem_i][pair].append(float(line[1]))
                   xi_ang[elem_i][pair].append(float(line[2]))
                   eta_ang_ij[elem_i][pair].append(float(line[3]))
                   eta_ang_ik[elem_i][pair].append(float(line[4]))
                   theta_s[elem_i][pair].append(float(line[5]))
         else:
             raise ValueError('Unrecognizable Angular Symmetry Function Type')
    pairs=sorted([*set(pairs)])
    return pairs, type_ang, rcut_ang, idum, rs_ang, xi_ang, eta_ang, lambda_ang, rs_ang_ij, rs_ang_ik, eta_ang_ij, eta_ang_ik, theta_s, nang

def generate_indjk(telem):
    """
    Generate arrays indjk and vecino based on the given number of elements (telem).

    Parameters:
    - telem (int): Number of elements.

    Returns:
    - indjk (ndarray): Array of indices for vecino calculation.
    - vecino (ndarray): Array storing vecino indices.
    """
    indjk = np.zeros(telem, dtype=int)
    vecino = np.zeros((telem * (telem + 1) // 2,), dtype=int)

    for i in range(1, telem + 1):
        indjk[i - 1] = (i - 1) * (2 * telem - i)

    counter = 0
    for i in range(telem):
        for j in range(i, telem):
            counter += 1
            vecino[counter - 1] = indjk[i] + j + 1

    return indjk, vecino

def clean_out_files(infile):
    outname_gp=infile+"_acsf_GP.out"
    outname_nn=infile+"_acsf_nn.out"
    try:
       os.remove(outname_gp)
       os.remove(outname_nn)
    except FileNotFoundError:
       pass
    return outname_gp,outname_nn

def trj2sf(geom,telem,tipo,type_rad,rcut_rad,radmax,rs_rad,eta_rad,nrad,indjk,vecino,type_ang,rcut_ang,
          angmax,idum,rs_ang,xi_ang,eta_ang,lambda_ang,rs_ang_ij,rs_ang_ik,eta_ang_ij,eta_ang_ik,theta_s,
          nang,outname_gp,outname_nn):
    """
    Compute the ACSF features
    """

    # Initialize some variables
    contador  = 0  
    fconcat   = 0.0  
    geomcount = 0
    with open(geom, 'r') as f:
        while True:
           print(" # Running the parse for geometry ",geomcount+1)
           try: 
              natom = int(f.readline().strip().split()[0])  
           except IndexError: 
              break
           coord = []
           label = []
           prop = []
           name=f.readline()
           for p in range(natom):
               line = f.readline().strip().split()
               nprop=int(len(line)-4)
               label.append(line[0])
               coord.append([float(x) for x in line[1:4]])
               if nprop > 0:
                  fproplist=[]
                  for lgdum in range(4,4+nprop):
                      fproplist.append(float(line[lgdum]))
                  prop.append(fproplist)
               else:
                  prop.append(None)
           coord = np.array(coord)
           prop=np.array(prop)
           # Compute radial ACSF
           radial = np.zeros((natom, telem, radmax))
           for i in range(natom):
               telem_i = chemclas(label[i],telem,tipo)
               for j in range(natom):
                   # Take into account atomic pairs formed by different atoms (j not equal to i)
                   if j != i: 
                       telem_j = chemclas(label[j],telem,tipo)
                       Zj = atomicnumber[label[j]]
                       factor = atomfactor(Zj)
                       rij = coord[j] - coord[i]  
                       distij = distance(coord[i], coord[j])
                       fcutij = cutoff(rcut_rad, distij)
                       if type_rad == 1:
                           for p in range(nrad[telem_i-1, telem_j-1]):
                               fpepe = np.exp(-eta_rad[telem_i-1, telem_j-1, p]*(distij-rs_rad[telem_i-1, telem_j-1, p])**2)*fcutij
                               radial[i, telem_j-1, p] += fpepe
                       elif type_rad == 2:
                           for p in range(nrad[telem_i-1, 0]):
                               fpepe = factor*np.exp(-eta_rad[telem_i-1, 0, p]*(distij-rs_rad[telem_i-1, 0, p])**2)*fcutij
                               radial[i, 0, p] += fpepe
           # Compute angular ACSF
           angular = np.zeros((natom, telem*(telem+1)//2, angmax))
           for i in range(natom):
               telem_i = chemclas(label[i],telem,tipo)
               for j in range(0,natom):
                   for k in range(j+1,natom):
                       if i != j and j != k and i != k:
                           # Determine the distance vectors between atom i and the other two atoms
                           rij = coord[j] - coord[i]
                           rik = coord[k] - coord[i]
                           rjk = coord[k] - coord[j]
                           # Compute the dot product of rij and rik
                           dot = np.dot(rij, rik)
                           # Compute the distances between the atoms i, j, and k
                           distij = np.linalg.norm(coord[i] - coord[j])
                           distik = np.linalg.norm(coord[i] - coord[k])
                           distjk = np.linalg.norm(coord[j] - coord[k])
                           # Apply a cutoff function to the distances
                           fcutij = cutoff(rcut_ang,distij)
                           fcutik = cutoff(rcut_ang,distik)
                           fcutjk = cutoff(rcut_ang,distjk)
                           # Compute the angle between rij and rik
                           theta = np.arccos(dot / (distij * distik))
                           if ((type_ang == 1) or (type_ang == 2) or (type_ang == 3) or (type_ang == 5)):
                               telem_j = chemclas(label[j], telem, tipo)
                               telem_k = chemclas(label[k], telem, tipo)
                               neigh_id = 0
                               if (telem_k >= telem_j):
                                   neigh = indjk[telem_j-1] + telem_k
                               else:
                                   neigh = indjk[telem_k-1] + telem_j
                               for u in range(telem*(telem+1)//2):
                                   if (vecino[u] == neigh):
                                       neigh_id = u+1
                               if (neigh_id == 0):
                                   raise Exception('FATAL ERROR IDENTIFYING THE NEIGHBOR PAIR NEIGH_ID=0')
                           elif (type_ang == 4):
                               Zj = atomicnumber[label[j]]
                               Zk = atomicnumber[label[k]]
                               factor = pairfactor(Zj, Zk)
                           # Define the atomic pair:
                           pair = label[j] + label[k] if label[j] <= label[k] else label[k] + label[j]
                           # Compute angular contributions to ACSF
                           if (type_ang == 1):
                               for p in range(nang[telem_i-1, neigh_id-1]):
                                   fpepe = ((2**(1-xi_ang[telem_i-1, neigh_id-1, p])) * 
                                            (((1 + lambda_ang[telem_i-1, neigh_id-1, p]*np.cos(theta))**xi_ang[telem_i-1, neigh_id-1, p]) * 
                                             np.exp(-eta_ang[telem_i-1, neigh_id-1, p]*((distij-rs_ang[telem_i-1, neigh_id-1, p])**2 + 
                                                                                   (distik-rs_ang[telem_i-1, neigh_id-1, p])**2 + 
                                                                                   (distjk-rs_ang[telem_i-1, neigh_id-1, p])**2))) * 
                                            fcutij * fcutik * fcutjk)
                                   angular[i, neigh_id-1, p] += fpepe
                           elif (type_ang == 2):
                               for p in range(nang[telem_i-1, neigh_id-1]):
                                   fpepe = ((2**(1-xi_ang[telem_i-1, neigh_id-1, p])) * 
                                            (((1 + lambda_ang[telem_i-1, neigh_id-1, p]*np.cos(theta))**xi_ang[telem_i-1, neigh_id-1, p]) * 
                                             np.exp(-eta_ang[telem_i-1, neigh_id-1, p]*((distij-rs_ang[telem_i-1, neigh_id-1, p])**2 + 
                                                                                   (distik-rs_ang[telem_i-1, neigh_id-1, p])**2))) * 
                                            fcutij * fcutik)
                                   angular[i, neigh_id-1, p] += fpepe
                           elif (type_ang == 3):
                               for p in range(nang[telem_i-1, neigh_id-1]):
                                   fpepe = ((2**(1-xi_ang[telem_i-1, neigh_id-1, p])) * 
                                            (((1 + np.cos(theta-np.radians(theta_s[telem_i-1, neigh_id-1, p]))) ** xi_ang[telem_i-1, neigh_id-1, p]) * 
                                             np.exp(-eta_ang[telem_i-1, neigh_id-1, p]*((((distij+distik)/2)-rs_ang[telem_i-1, neigh_id-1, p])**2))) * 
                                            fcutij * fcutik)
                                   angular[i, neigh_id-1, p] += fpepe
                           elif (type_ang == 4):
                               # WACSF
                               neigh_id = 1  # As we are working with WACSF
                               for p in range(nang[telem_i-1, neigh_id-1]):
                                   fpepe = factor * (2.0**(1.0 - xi_ang[telem_i-1, neigh_id-1, p])) * \
                                       (((1.0 + lambda_ang[telem_i-1, neigh_id-1, p] * np.cos(theta))**xi_ang[telem_i-1, neigh_id-1, p]) *
                                        np.exp(-eta_ang[telem_i-1, neigh_id-1, p] * ((distij - rs_ang[telem_i-1, neigh_id-1, p])**2 + 
                                        (distik - rs_ang[telem_i-1, neigh_id-1, p])**2 + (distjk - rs_ang[telem_i-1, neigh_id-1, p])**2))) * \
                                       fcutij * fcutik * fcutjk
                                   angular[i, 0, p] += fpepe
                           elif (type_ang == 5):     # Heavily modified angular with independent radial grids   
                               for p in range(nang[telem_i-1, neigh_id-1]):
                                   # Homoatomic case, the pairs must be ordered!
                                   if (telem_j == telem_k ):  # Account for the order!
                                      if (distik > distij):
                                          fpepe = ((2**(1-xi_ang[telem_i-1, neigh_id-1, p])) * 
                                                   (((1 + np.cos(theta-np.radians(theta_s[telem_i-1, neigh_id-1, p]))) ** xi_ang[telem_i-1, neigh_id-1, p]) * 
                                                    np.exp(-eta_ang_ij[telem_i-1, neigh_id-1, p]*((distij-rs_ang_ij[telem_i-1, neigh_id-1, p])**2))   *
                                                    np.exp(-eta_ang_ik[telem_i-1, neigh_id-1, p]*((distik-rs_ang_ik[telem_i-1, neigh_id-1, p])**2))) *
                                                   fcutij * fcutik)
                                      else:
                                          fpepe = ((2**(1-xi_ang[telem_i-1, neigh_id-1, p])) * 
                                                   (((1 + np.cos(theta-np.radians(theta_s[telem_i-1, neigh_id-1, p]))) ** xi_ang[telem_i-1, neigh_id-1, p]) * 
                                                    np.exp(-eta_ang_ij[telem_i-1, neigh_id-1, p]*((distik-rs_ang_ij[telem_i-1, neigh_id-1, p])**2))   *
                                                    np.exp(-eta_ang_ik[telem_i-1, neigh_id-1, p]*((distij-rs_ang_ik[telem_i-1, neigh_id-1, p])**2))) *
                                                   fcutij * fcutik)
                                   # Heteroatomic case
                                   else:
                                      if (label[j] == pair[0]):
                                          fpepe = ((2**(1-xi_ang[telem_i-1, neigh_id-1, p])) * 
                                                   (((1 + np.cos(theta-np.radians(theta_s[telem_i-1, neigh_id-1, p]))) ** xi_ang[telem_i-1, neigh_id-1, p]) * 
                                                    np.exp(-eta_ang_ij[telem_i-1, neigh_id-1, p]*((distij-rs_ang_ij[telem_i-1, neigh_id-1, p])**2))   *
                                                    np.exp(-eta_ang_ik[telem_i-1, neigh_id-1, p]*((distik-rs_ang_ik[telem_i-1, neigh_id-1, p])**2))) *
                                                   fcutij * fcutik)
                                      else:
                                          fpepe = ((2**(1-xi_ang[telem_i-1, neigh_id-1, p])) * 
                                                   (((1 + np.cos(theta-np.radians(theta_s[telem_i-1, neigh_id-1, p]))) ** xi_ang[telem_i-1, neigh_id-1, p]) * 
                                                    np.exp(-eta_ang_ij[telem_i-1, neigh_id-1, p]*((distik-rs_ang_ij[telem_i-1, neigh_id-1, p])**2))   *
                                                    np.exp(-eta_ang_ik[telem_i-1, neigh_id-1, p]*((distij-rs_ang_ik[telem_i-1, neigh_id-1, p])**2))) *
                                                   fcutij * fcutik)
                                   angular[i, neigh_id-1, p] += fpepe
           # Obtain final ACSF
           acsf=[]
           for i in range(natom):
               telem_i=chemclas(label[i], telem, tipo)
               atom_acsf=[]
               atom_acsf.clear()
               if type_rad == 1:
                  rad_acsf= [radial[i, k, p] for k in range(telem) for p in range(nrad[telem_i-1, k])]
               elif type_rad == 2:
                  rad_acsf= [radial[i, 0, p] for p in range(nrad[telem_i-1, 0])]
               if type_ang in [1, 2, 3, 5]:
                  ang_acsf = [(angular[i, k, p]) for k in range(telem*(telem+1)//2) for p in range(nang[telem_i-1, k])]
               elif type_ang == 4:
                  ang_acsf = [(angular[i, 0, p]) for p in range(nang[telem_i-1, 0])]
               # Create final list and list of lists
               for val in rad_acsf: atom_acsf.append(val)
               for val in ang_acsf: atom_acsf.append(val)
               acsf.append(atom_acsf)
           # Append the data to the file
           append2GPR(outname_gp,name,label,coord,prop,acsf)
           append2NN(outname_nn,label,prop,acsf)
           geomcount+=1
    return None 

def trj2sf_new_format(geom,elements,ordered_pairs,pairs,type_rad,rcut_rad,rs_rad,eta_rad,nrad,type_ang,rcut_ang,
          idum,rs_ang,xi_ang,eta_ang,lambda_ang,rs_ang_ij,rs_ang_ik,eta_ang_ij,eta_ang_ik,theta_s,
          nang,outname_gp,outname_nn):
    """
    Compute the ACSF features
    """

    contador = 0  
    fconcat = 0.0  
    geomcount=0
    with open(geom, 'r') as f:
        while True:
           print(" # Running the parse for geometry ",geomcount+1)
           try: 
              natom = int(f.readline().strip().split()[0])  
           except IndexError: 
              break
           coord = []
           label = []
           prop = []
           name=f.readline()
           for p in range(natom):
               line = f.readline().strip().split()
               nprop=int(len(line)-4)
               label.append(line[0])
               coord.append([float(x) for x in line[1:4]])
               if nprop > 0:
                  fproplist=[]
                  for lgdum in range(4,4+nprop):
                      fproplist.append(float(line[lgdum]))
                  prop.append(fproplist)
               else:
                  prop.append(None)
           coord = np.array(coord)
           prop=np.array(prop)
    
           # Compute radial ACSF
           radial=[]
           for i in range(natom):
               # Radial ACSF for atom i
               if (type_rad == 1):
                   radial_atom_i = {}
                   for neigh in rs_rad[label[i]].keys(): 
                       radial_atom_i[neigh] = {}
                       for radacsf in range(len(rs_rad[label[i]][neigh])):
                           radial_atom_i[neigh][radacsf] = 0.0
               elif (type_rad == 2):
                   radial_atom_i = {}
                   for radacsf in range(len(rs_rad[label[i]])):
                       radial_atom_i[radacsf] = 0.0
               for j in range(natom):
                   # Take into account atomic pairs formed by different atoms (j not equal to i)
                   if j != i:  
                       Zj = atomicnumber[label[j]]
                       factor = atomfactor(Zj)
                       rij = coord[j] - coord[i]  
                       distij = distance(coord[i], coord[j])
                       fcutij = cutoff(rcut_rad, distij)
                       if type_rad == 1:
                           for p in range(len(rs_rad[label[i]][label[j]])):
                               fpepe = np.exp(-eta_rad[label[i]][label[j]][p]*(distij-rs_rad[label[i]][label[j]][p])**2)*fcutij
                               radial_atom_i[label[j]][p] += fpepe           
                       elif type_rad == 2:
                           for p in range(len(rs_rad[label[i]])):
                               fpepe = factor*np.exp(-eta_rad[label[i]][p]*(distij-rs_rad[label[i]][p])**2)*fcutij
                               radial_atom_i[p] += fpepe           
               radial.append(flatten_dict(radial_atom_i))
           # Compute angular ACSF
           angular=[]
           for i in range(natom):
               if (type_ang in [1,2,3,5]):
                   angular_atom_i = {}
                   for neigh in xi_ang[label[i]].keys(): 
                       angular_atom_i[neigh] = {}
                       for angacsf in range(len(xi_ang[label[i]][neigh])):
                           angular_atom_i[neigh][angacsf] = 0.0
               elif (type_ang == 4):
                   angular_atom_i = {}
                   for angacsf in range(len(xi_ang[label[i]])):
                       angular_atom_i[angacsf] = 0.0
               for j in range(0,natom):
                   for k in range(j+1,natom):
                       if i != j and j != k and i != k:
                           # Determine the distance vectors between atom i and the other two atoms
                           rij = coord[j] - coord[i]
                           rik = coord[k] - coord[i]
                           rjk = coord[k] - coord[j]
                           # Compute the dot product of rij and rik
                           dot = np.dot(rij, rik)
                           # Compute the distances between the atoms i, j, and k
                           distij = np.linalg.norm(coord[i] - coord[j])
                           distik = np.linalg.norm(coord[i] - coord[k])
                           distjk = np.linalg.norm(coord[j] - coord[k])
                           # Apply a cutoff function to the distances
                           fcutij = cutoff(rcut_ang,distij)
                           fcutik = cutoff(rcut_ang,distik)
                           fcutjk = cutoff(rcut_ang,distjk)
                           # Compute the angle between rij and rik and centered at i
                           theta = np.arccos(dot / (distij * distik))
                           Zj = atomicnumber[label[j]]
                           Zk = atomicnumber[label[k]]
                           factor = pairfactor(Zj, Zk)
                           # Define the atomic pair:
                           pair_jk = label[j] + label[k] if label[j] <= label[k] else label[k] + label[j]
                           if (type_ang == 4):
                                 for p in range(len(xi_ang[label[i]])):
                                     fpepe = factor * (2.0**(1.0 - xi_ang[label[i]][p])) *(((1.0 + lambda_ang[label[i]][p] * np.cos(theta))**xi_ang[label[i]][p]) * np.exp(-eta_ang[label[i]][p] * ((distij - rs_ang[label[i]][p])**2 +  (distik - rs_ang[label[i]][p])**2 + (distjk - rs_ang[label[i]][p])**2))) * fcutij * fcutik * fcutjk
                                     angular_atom_i[p] += fpepe           
                           elif (pair_jk in xi_ang[label[i]].keys()):
                             # Compute angular contributions to ACSF
                             if (type_ang == 1):
                                for p in range(len(xi_ang[label[i]][pair_jk])):
                                     fpepe = ((2**(1-xi_ang[label[i]][pair_jk][p])) * 
                                              (((1 + lambda_ang[label[i]][pair_jk][p]*np.cos(theta))**xi_ang[label[i]][pair_jk][p]) * 
                                               np.exp(-eta_ang[label[i]][pair_jk][p]*((distij-rs_ang[label[i]][pair_jk][p])**2 + 
                                                                                     (distik-rs_ang[label[i]][pair_jk][p])**2 + 
                                                                                     (distjk-rs_ang[label[i]][pair_jk][p])**2))) * fcutij * fcutik * fcutjk)
                                     angular_atom_i[pair_jk][p] += fpepe
                             elif (type_ang == 2):
                                for p in range(len(xi_ang[label[i]][pair_jk])):
                                     fpepe = ((2**(1-xi_ang[label[i]][pair_jk][p])) * 
                                              (((1 + lambda_ang[label[i]][pair_jk][p]*np.cos(theta))**xi_ang[label[i]][pair_jk][p]) * 
                                               np.exp(-eta_ang[label[i]][pair_jk][p]*((distij-rs_ang[label[i]][pair_jk][p])**2 + 
                                                                                     (distik-rs_ang[label[i]][pair_jk][p])**2))) * 
                                              fcutij * fcutik)
                                     angular_atom_i[pair_jk][p] += fpepe
                             elif (type_ang == 3):
                                for p in range(len(xi_ang[label[i]][pair_jk])):
                                     fpepe = ((2**(1-xi_ang[label[i]][pair_jk][p])) * 
                                              (((1 + np.cos(theta-np.radians(theta_s[label[i]][pair_jk][p]))) ** xi_ang[label[i]][pair_jk][p]) * 
                                               np.exp(-eta_ang[label[i]][pair_jk][p]*((((distij+distik)/2)-rs_ang[label[i]][pair_jk][p])**2))) * 
                                              fcutij * fcutik)
                                     angular_atom_i[pair_jk][p] += fpepe
                             elif (type_ang == 5):     # Heavily modified angular with independent radial grids   
                                 distances=np.array([distij,distik])
                                 # Homoatomic case
                                 if (label[j] == label[k]): 
                                      distik = np.max(distances)
                                      distij = np.min(distances)
                                 else:
                                      if (label[j] == pair_jk[0]):
                                         distij=distances[0]
                                         distik=distances[1]
                                      else:
                                         distij=distances[1]
                                         distik=distances[0]
                                 for p in range(len(xi_ang[label[i]][pair_jk])):
                                   fpepe = ((2**(1-xi_ang[label[i]][pair_jk][p])) * 
                                            (((1 + np.cos(theta-np.radians(theta_s[label[i]][pair_jk][p]))) ** xi_ang[label[i]][pair_jk][p]) * 
                                             np.exp(-eta_ang_ij[label[i]][pair_jk][p]*((distij-rs_ang_ij[label[i]][pair_jk][p])**2))   *
                                             np.exp(-eta_ang_ik[label[i]][pair_jk][p]*((distik-rs_ang_ik[label[i]][pair_jk][p])**2))) *
                                            fcutij * fcutik)
                                   angular_atom_i[pair_jk][p] += fpepe
               # Store angular ACSFs to a list
               lista=[]
               if (type_ang == 4):
                   angular.append(flatten_dict(angular_atom_i))
               else:
                   for x in ordered_pairs:
                       if x in angular_atom_i.keys():
                          for y in angular_atom_i[x].keys():
                              lista.append(angular_atom_i[x][y])
                   angular.append(lista)
           # Obtain final ACSF
           acsf=[]
           for i in range(natom):
               # Create final list and list of lists
               atom_acsf=[]
               for val in radial[i]: atom_acsf.append(val)
               for val in angular[i]: atom_acsf.append(val)
               acsf.append(atom_acsf)
           # Append the data to the file
           append2GPR(outname_gp,name,label,coord,prop,acsf)
           append2NN(outname_nn,label,prop,acsf)
           geomcount+=1
    return None 
    
