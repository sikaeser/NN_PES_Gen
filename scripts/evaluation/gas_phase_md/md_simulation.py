#!/usr/bin/env python3

# imports
import argparse
from ase import Atoms
from ase.io import read, write
from ase.optimize import *
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, ZeroRotation, Stationary
from ase.md.langevin import Langevin
from ase.md.verlet import VelocityVerlet
from ase.io.trajectory import Trajectory
from os.path import splitext
from NNCalculator.NNCalculator import *
import time

'''
script to run an MD simulation using a PhysNet model
Usage python md_simulation.py -i opt_structure.xyz -o outputfilename

Path to models as well as hyperparameters need to be adapted.

The MD settings can be changed from the command line (e.g. --mdtype NVE) or
by changing the defaults.
'''


#parse command line arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser._action_groups.pop()
required = parser.add_argument_group("required arguments")
required.add_argument("-i", "--input",   type=str,   help="input xyz",  required=True)
required.add_argument("-o", "--output",  type=str,   help="output xyz", required=True)
optional = parser.add_argument_group("optional arguments")
optional.add_argument("--charge",  type=float, help="total charge", default=0.0)
optional.add_argument("--temperature", type=float, help="Set momenta corresponding to a temperature T", default=300)
optional.add_argument("--timestep",  type=float, help="timestep for integration algorithm", default=0.1)
optional.add_argument("--friction",  type=float, help="friction coeff for Langevin algorithm", default=0.02)    #if NVT is chosen
optional.add_argument("--interval",  type=float, help="interval", default=10)                                   #interval to save snapshots of the md
optional.add_argument("--equilsteps",  type=int, help="equilibration steps", default=1000)                      #steps for an equilibration phase during the md
optional.add_argument("--prodsteps",  type=int, help="production steps", default=50000)                        #steps for the production phase during the md
optional.add_argument("--mdtype",  type=str, help="NVE or NVT", default="NVE")                                  #type of md simulation to be run

args = parser.parse_args()
print("input ", args.input)
print("output", args.output)

print()
print("++++++++++++++++++++++++++")
print("MD Settings:")
print("MD Type", args.mdtype)
print("Time step", args.timestep)
print("Temperature", args.temperature)
print("Equilibration steps", args.equilsteps)
print("Production steps", args.prodsteps)
print("++++++++++++++++++++++++++")


#read input file 
atoms = read(args.input)

#shift such that structure is centered around origin
atoms.set_positions(atoms.get_positions()-np.mean(atoms.get_positions(), axis=0))


#setup calculator object, which in this case is the NN calculator
#it is important that it is setup with the settings as used in the
#training procedure.
calc = NNCalculator(
    checkpoint="../models_clPhOH/clphoh.meta.mp2.631g.3000_a", #load the model you want to use
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

# Optimize initial geometry
BFGS(atoms).run(fmax=0.00001)

#preparing the traj object to save the trajectory in a given interval 
traj = Trajectory(args.output + '.traj', 'w', atoms)

# Set the momenta corresponding to a temperature T
MaxwellBoltzmannDistribution(atoms, args.temperature * units.kB)

if args.mdtype == "NVE":
    ZeroRotation(atoms) #supress rotation and translation
    Stationary(atoms)
    #define algorithm for MD:
    dyn = VelocityVerlet(atoms, args.timestep * units.fs)

elif args.mdtype == "NVT":   
    # define the algorithm for MD:
    dyn = Langevin(atoms, args.timestep * units.fs, args.temperature * units.kB, args.friction)



t = time.time()
for i in range(args.equilsteps):
    dyn.run(1)
    if i % 100 == 0:
        print("Equilibration at step: " + str(i))
        print(time.time() - t, "s")
        t = time.time()
        
        epot = atoms.get_potential_energy()
        ekin = atoms.get_kinetic_energy()
        etot = epot + ekin
        print("Total energy:", np.asscalar(etot), " eV")

# run the dynamics
counter = 0
for i in range(args.prodsteps):
    dyn.run(1)
    if i % 100 == 0:
        print("Production run, at step: " + str(i))
        print(time.time() - t, "s")
        t = time.time()
        epot = atoms.get_potential_energy()
        ekin = atoms.get_kinetic_energy()
        etot = epot + ekin
        print("Total energy:", np.asscalar(etot), " eV")

    if i % args.interval == 0:
        traj.write(atoms)


