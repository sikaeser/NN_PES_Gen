#!/usr/bin/env python3


import argparse
import numpy as np
from ase import io
from ase.io.trajectory import Trajectory
from ase.io import read, write
from os.path import splitext
import ase.units as units

'''
Usage: python3 analyze_energy.py -i dataset.npz
'''

#parse command line arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser._action_groups.pop()
required = parser.add_argument_group("required arguments")
required.add_argument("-i", "--input",   type=str,   help="dataset",  required=True)
optional = parser.add_argument_group("optional arguments")
optional.add_argument("--charge",  type=float, help="total charge", default=0.0)


args = parser.parse_args()
filename, extension = splitext(args.input)

import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(2)
data = np.load(args.input)
E = data["E"]*23.0605
Z = data["Z"]
Z_unique = np.unique(Z, axis=0)

if len(E) > 1000:
    ax1.hist(E,int(len(E)/50))

    ax1.set_xlabel('$E$ [Kcal/mol]')
    ax1.set_ylabel('Count')

    ax2.scatter(range(len(E)),E)
    ax2.set_ylabel('$E$ [Kcal/mol]')
    ax2.set_xlabel('Index')
    plt.show()

else:
    plt.plot(E)
    plt.show()


#beingsaved.savefig('energyrange.eps', format='eps', dpi=1200)


