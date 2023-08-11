#!/usr/bin/env python3

# imports
import argparse
from ase import Atoms
from ase.io import read, write
from ase.optimize import *
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase.io.trajectory import Trajectory
from os.path import splitext
from NNCalculator.NNCalculator import *

'''
script to run adaptive sampling using an ENSEMBLE OF PhysNet models
Usage python adaptive_sampling_md.py -i opt_structure.xyz -o outputfilename

Path to models as well as hyperparameters need to be adapted.
'''


#parse command line arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser._action_groups.pop()
required = parser.add_argument_group("required arguments")
required.add_argument("-i", "--input",   type=str,   help="input xyz",  required=True)
required.add_argument("-o", "--output",  type=str,   help="output xyz", required=True)
optional = parser.add_argument_group("optional arguments")
optional.add_argument("--charge",  type=float, help="total charge", default=0.0)
optional.add_argument("--temperature", type=float, help="Set momenta corresponding to a temperature T", default=2000)
optional.add_argument("--timestep",  type=float, help="timestep for Langevin algorithm", default=0.1)
optional.add_argument("--friction",  type=float, help="friction coeff for Langevin algorithm", default=0.02)
optional.add_argument("--interval",  type=float, help="interval", default=10)
optional.add_argument("--energythreshold",  type=float, help="when stdev is larger than this, write", default=0.0215) 
args = parser.parse_args()
print("input ", args.input)
print("output", args.output)


#read input file 
atoms = read(args.input)

#shift such that structure is centered around origin
atoms.set_positions(atoms.get_positions()-np.mean(atoms.get_positions(), axis=0))


#setup calculator object, which in this case is the NN calculator
#it is important that it is setup with the settings as used in the
#training procedure.
calc = NNCalculator(
    checkpoint=["../models_clPhOH/clphoh.meta.mp2.631g.3000_a","../models_clPhOH/clphoh.meta.mp2.631g.3000_b"], #load the ENSEMBLE of models you want to used
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


#setup calculator (which will be used to describe the atomic interactions)
atoms.set_calculator(calc)                   

# Set the momenta corresponding to a temperature T
MaxwellBoltzmannDistribution(atoms, args.temperature * units.kB)

# define the algorithm for MD: here Langevin alg. with with a time step of 0.1 fs,
# the temperature T and the friction coefficient to 0.02 atomic units.
dyn = Langevin(atoms, args.timestep * units.fs, args.temperature * units.kB, args.friction)
traj = Trajectory(args.output + '.traj', 'w', atoms)


# run the dynamics
counter = 0
for i in range(1000000000000):
    dyn.run(1)
    if i % args.interval == 0:
        epot = atoms.get_potential_energy() / len(atoms)
        ekin = atoms.get_kinetic_energy() / len(atoms)
        print(i, counter, epot, ekin, epot+ekin, calc.energy_stdev)
        if calc.energy_stdev > args.energythreshold: # 0.043 is approx 1 kcal/mol and we want to be better than that!
            traj.write()
            counter = counter + 1
        if counter > 999:
            break

