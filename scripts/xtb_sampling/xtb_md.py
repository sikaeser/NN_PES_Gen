#!/usr/bin/env python3

# imports
import argparse
from ase import Atoms
from ase.io import read, write
from ase.optimize import *
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, ZeroRotation, Stationary
from ase.md.langevin import Langevin
from ase.io.trajectory import Trajectory
from os.path import splitext
from ase.md.verlet import VelocityVerlet
import numpy as np

from xtb.ase.calculator import XTB
#parse command line arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser._action_groups.pop()
required = parser.add_argument_group("required arguments")
required.add_argument("-i", "--input",   type=str,   help="input xyz",  required=True)
optional = parser.add_argument_group("optional arguments")
optional.add_argument("--label",   type=str,   help="prefix of calculator files",  default="calc_xtb1/md")
optional.add_argument("--charge",  type=float, help="total charge", default=0.0)
optional.add_argument("--temperature", type=float, help="Set momenta corresponding to a temperature T", default=2000)
optional.add_argument("--timestep",  type=float, help="timestep for Langevin algorithm", default=0.1)
optional.add_argument("--friction",  type=float, help="friction coeff for Langevin algorithm", default=0.02)
optional.add_argument("--magmom",  type=int, help="magnetic moment (number of unpaired electrons)", default=0)
optional.add_argument("--steps",  type=int, help="number of steps in md", default=500000)
optional.add_argument("--interval",  type=float, help="interval for saving snapshots", default=100)

args = parser.parse_args()
filename, extension = splitext(args.input)


#read input file 
atoms = read(args.input)


#setup calculator
calc = XTB(
    label=args.label,
    charge=args.charge,
    magmom=args.magmom)

#setup calculator (which will be used to describe the atomic interactions)
atoms.set_calculator(calc)                   

#run an optimization
BFGS(atoms).run(fmax=0.0001)

# Set the momenta corresponding to a temperature T
MaxwellBoltzmannDistribution(atoms, args.temperature * units.kB)
ZeroRotation(atoms)
Stationary(atoms)

# define the algorithm for MD: here Langevin alg. with with a time step of 0.1 fs,
# the temperature T and the friction coefficient to 0.02 atomic units.
dyn = Langevin(atoms, args.timestep * units.fs, args.temperature * units.kB, args.friction)
#dyn = VelocityVerlet(atoms, args.timestep * units.fs)


def printenergy(a=atoms):  # store a reference to atoms in the definition.
    """Function to print the potential, kinetic and total energy."""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    print('Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
          'Etot = %.3feV' % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin))
    
# save the positions of all atoms after every Xth time step.
traj = Trajectory(str(args.temperature)+ 'K_md_' + filename + '.traj', 'w', atoms)

#equilibration
for i in range(10000):
    if i%100 == 0:
        print("Equilibration Step: ", i)
    dyn.run(1)



# run the dynamics
for i in range(args.steps):
    dyn.run(1)
    if i%args.interval == 0:
        #epot = atoms.get_potential_energy() / len(atoms)
        #ekin = atoms.get_kinetic_energy() / len(atoms)
        print("Production Step: ", i)
        traj.write()


