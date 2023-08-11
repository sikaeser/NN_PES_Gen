#!/usr/bin/env python3

#imports
import argparse
from ase import Atoms
import numpy as np
from ase.visualize import view
from ase.io import read, write
from ase.optimize import *
from ase import units
from NNCalculator.NNCalculator import *
from ase.md.langevin import Langevin
from ase.io.trajectory import Trajectory


#parse command line arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser._action_groups.pop()
required = parser.add_argument_group("required arguments")
required.add_argument("-i", "--input",   type=str,   help="input xyz",  required=True)
optional = parser.add_argument_group("optional arguments")
optional.add_argument("--charge",  type=float, help="total charge", default=0.0)
optional.add_argument("--fmax",  type=float, help="maximal force", default=0.0005)
from os.path import splitext

args = parser.parse_args()
filename, extension = splitext(args.input)
print("input ", args.input)

#read input file (molecular structure to predict) and create
#an atoms object
atoms = read(args.input)

#setup calculator object, which in this case is the NN calculator
#it is important that it is setup with the settings as used in the
#training procedure.
calc = NNCalculator(
    checkpoint="../models_clPhOH/clphoh.meta.mp2.631g.3000_a", #load the model you want to used
    atoms=atoms,
    charge=args.charge,
    F=128,
    K=64,
    num_blocks=5,
    num_residual_atomic=2,
    num_residual_interaction=3,
    num_residual_output=1,
    sr_cut=10.0,
    use_electrostatic=True,
    use_dispersion=True)         

#attach the calculator object (used to describe the atomic interaction) to the atoms object
atoms.set_calculator(calc)

# chose optimization algorithm (MDMin, BFGS, FIRE)
algorithm = BFGS
dyn = algorithm(atoms)



#create constraint
from ase.constraints import FixInternals

#dihedral = atoms.get_dihedral(2, 3, 11, 12)
#print(dihedral)

#create traj object where to save the scan
traj = Trajectory("ClPhOH_scan.traj", "w")

#save minimum geometry
traj.write(atoms)

#loop over the dihedral and save scan
#at each step constrain the dihedral and optimize remaining part of molecule
angle = 1.0# atoms.get_dihedral(2, 3, 11, 12)
for i in range(180):
    dihedral_indices1 = [2, 3, 11, 12]
    atoms.rotate_dihedral(dihedral_indices1, -angle)
    dihedral1 = [atoms.get_dihedral(*dihedral_indices1)* np.pi / 180, dihedral_indices1]
    c = FixInternals(dihedrals=[dihedral1])
    atoms.set_constraint(c)

    dyn = BFGS(atoms)
    dyn.run(fmax=0.0005)
    traj.write(atoms)

    


