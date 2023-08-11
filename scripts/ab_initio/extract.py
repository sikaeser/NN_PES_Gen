#!/usr/bin/env python3


from ase.units import *
import re
from ase.io import read, write
import os
import numpy as np


file_list = []
for path, dirs, files in os.walk("out"):
    for file in files:
        if file.endswith(".out"):
            file_list.append(path+"/"+file)

Nmax = 11 #needs to be adapted

num = len(file_list)
N = np.zeros([num], dtype=int)
E = np.zeros([num], dtype=float)
Q = np.zeros([num], dtype=float)
D = np.zeros([num, 3], dtype=float)
Z = np.zeros([num, Nmax], dtype=int)
R = np.zeros([num, Nmax, 3], dtype=float)
F = np.zeros([num, Nmax, 3], dtype=float)

#reference mp2/aug-cc-pVTZ energies for the single H, C, O atoms. The corresponding value
#will be subtracted per corresponding atom in the molecule. If other atoms are contained
#in the molecule, the list needs to be extended.
Eref = np.zeros([10], dtype=float)
Eref[1] = -0.499821176024
Eref[6] = -37.759560677467
Eref[8] = -74.959294147330

'''
For B3LYP/cc-pVDZ:
H: -0.497858658764 hartree
C: -37.830617391474 hartree
O: -75.039041613326 hartree

For CCSD(T)-F12/aug-cc-pVTZ-F12
H: -0.499946213283 hartree
C: -37.788204984713 hartree
O: -75.000839553994 hartree
'''

index = 0
for file in file_list:
    # open file and read contents
    with open(file, "r") as f:
        contents = f.read().splitlines()

    #search for CARTESIAN COORDINATES:
    linenumber = [i for i,line in enumerate(contents) if re.search(' geometry={', line)][0]
    linenumber1 = [i for i,line in enumerate(contents) if re.search(' }', line)][0]
    Ntmp = linenumber1-linenumber-1
    Ztmp = []
    Rtmp = []
    for line in contents[linenumber+1:linenumber+1+Ntmp]:
        l, x, y, z = line.split()
        if l == 'H':
            Ztmp.append(1)
        elif l == 'C':
            Ztmp.append(6)
        elif l == 'O':
            Ztmp.append(8)
        else:
            print("UNKNOWN LABEL", l)
            quit()
        Rtmp.append([float(x), float(y), float(z)])

    #search for forces:
    linenumberF = [i for i,line in enumerate(contents) if re.search('FORCEX', line)][Ntmp]

    Ftmp = []
    for line in contents[linenumberF+1:linenumberF+1+Ntmp]:
        x, y, z = line.split()
        Ftmp.append([float(x), float(y), float(z)])

    #search for DIPOLE MOMENT:
    linenumberDX = [i for i,line in enumerate(contents) if re.search('DIPOLEX', line)][1]
    Dx = float(contents[linenumberDX].split()[2])
    linenumberDY = [i for i,line in enumerate(contents) if re.search('DIPOLEY', line)][1]
    Dy = float(contents[linenumberDY].split()[2])
    linenumberDZ = [i for i,line in enumerate(contents) if re.search('DIPOLEZ', line)][1]
    Dz = float(contents[linenumberDZ].split()[2])
    Dtmp = [Dx, Dy, Dz]
    
    #search for FINAL SINGLE POINT ENERGY:
    linenumberE = [i for i,line in enumerate(contents) if re.search('MP2ENERGY', line)][1]
    Etmp = float(contents[linenumberE].split()[2])

    #subtract asymptotics
    for z in Ztmp:
        Etmp -= Eref[z]

    #search for TOTAL CHARGE
    linenumberQ = [i for i,line in enumerate(contents) if re.search('symmetry,nosym', line)][0]
    Qtmp = float(contents[linenumberQ+1].split(",")[1].split("=")[1])    

    N[index] = Ntmp
    E[index] = Etmp
    Q[index] = Qtmp
    D[index,:] = np.asarray(Dtmp)
    Z[index,:Ntmp] = np.asarray(Ztmp)
    R[index,:Ntmp,:] = np.asarray(Rtmp)
    F[index,:Ntmp,:] = np.asarray(Ftmp)

    index += 1
    if index%100 == 0:
        print(index/num*100,"%")


#unit conversion
E *= Hartree
D *= Debye
F *= Hartree/Bohr

np.savez_compressed("dataset.npz", N=N, E=E, Q=Q, D=D, Z=Z, R=R, F=F)


