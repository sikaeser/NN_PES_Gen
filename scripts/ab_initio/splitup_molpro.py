#!/usr/bin/env python3

import argparse
from ase import Atoms
from ase.io import read, write
from ase.io.trajectory import Trajectory
from os.path import splitext
import io, os
import numpy as np

#parse command line arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser._action_groups.pop()
required = parser.add_argument_group("required arguments")
required.add_argument("-i", "--input",   type=str,   help="input traj",  required=True)
optional = parser.add_argument_group("optional arguments")
optional.add_argument("-o", "--output",   type=str,   help="output folder name",  default="inp_molpro")
args = parser.parse_args()

#get current working directory and make a scratch 
#directory
path = os.getcwd()
path = path + '/' + args.output
if not os.path.exists(path): os.makedirs(path)
filename, extension = splitext(args.input)


# read input file
traj = Trajectory(args.input)


# define function to translate atomic number to atomic representations
# MIGHT NEED FURTHER STATEMENTS
def atomic_numbers_to_labels(Z):
	labels = []
	for z in Z:
		if z == 1:
			labels.append('H')
		elif z == 6: 
			labels.append('C')
		elif z == 8:
			labels.append('O')
		elif z == 17:
			labels.append('Cl')
		else:
			print("UNSUPPORTED ATOMIC NUMBER", z)
			quit()
	return labels

def calculate_COC(pos, num):
    R = np.zeros(3)
    for i in range(pos.shape[0]):
        R += num[i] * pos[i,:]
    R /= sum(num)
    return R



# get atomic numbers and atomic positions for every structure
for i in range(len(traj)):
    atoms = traj[i]
    string = filename + '_%03d' % (i,) +'.inp'
    completeName = os.path.join(path, string)


    
# calculate centre of charge
    pos = atoms.get_positions()
    num = atoms.get_atomic_numbers()
    R = calculate_COC(pos, num)
    pos = pos - R




# write necessary header for Molpro input file
    with open(completeName, "w") as f:
        f.write('!MP2/6-31G(d,p)\n')
        f.write('memory,500,m\n')
        f.write('symmetry,nosym\n')
        f.write('wf,charge=0,spin=0\n')
        f.write('basis=6-31G(d,p)\n')
        f.write('geometry={\n')


        labels = atomic_numbers_to_labels(atoms.get_atomic_numbers())
        for a in range(len(labels)):
            x, y, z = pos[a,:]
            l = labels[a]
            f.write(' ' + l + '    ' + str(x) + '    ' + str(y) + '   ' +  str(z) +"\n")

        f.write('}\n\n')
        f.write('!perform calculations\n')
        f.write('hf\n')
        f.write('mp2\n')
        f.write('force;varsav\n\n')
        f.write('!store results\n')
        f.write('mp2energy = energy\n')
        f.write('dipolex   = dmx*TODEBYE\n')
        f.write('dipoley   = dmy*TODEBYE\n')
        f.write('dipolez   = dmz*TODEBYE\n')
        f.write('forcex    = -gradx\n')
        f.write('forcey    = -grady\n')
        f.write('forcez    = -gradz\n\n')
        f.write('!display results\n')
        f.write('show[1,f30.16],mp2energy\n')
        f.write('show[3,F30.16],dipolex,dipoley,dipolez\n')
        f.write('table,forcex,forcey,forcez\n')
        f.write('digits,16,16,16\n')
