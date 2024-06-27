#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import argparse

#data
data_all = np.genfromtxt("hydrogen_oxalate_mp2_avtz_gen2_22200_a.dat")*23.0605


# definitions for the axes
left, width = 0.15, 0.65
bottom, height = 0.1, 0.65
spacing = 0.03


rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom + height + spacing, width, 0.2]

# start with a rectangular Figure
plt.figure(figsize=(8, 8))

ax1 = plt.axes(rect_scatter)
ax1.tick_params(direction='in', top=True, right=True)
ax0 = plt.axes(rect_histx)
ax0.tick_params(direction='in', labelbottom=False)

ax1.set_aspect('equal', adjustable='box')
# Fontsize
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

#plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('font', size=SMALL_SIZE, weight='bold')  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#plot section
#global params



plt.rcParams['lines.markersize'] = 12
#plt.rcParams['axes.linewidth'] = 20000
for axis in ['top','bottom','left','right']:
  ax0.spines[axis].set_linewidth(2)
for axis in ['top','bottom','left','right']:
  ax1.spines[axis].set_linewidth(2)



ax0.axhline(0.0, linestyle='--', color='grey')




ax0.tick_params(width=2, length=10, labelsize=15)
ax0.tick_params(which='minor',width=1, length=5, color='black', labelsize=15)
ax1.tick_params(width=2, length=10, labelsize=15)
ax1.tick_params(which='minor',width=1, length=5, color='black', labelsize=15)


"""
ax0.set_xlim(-730, -280)
ax0.set_ylim(-50, 50, 25)
ax1.set_ylim(-730, -280)
ax1.set_xlim(-730, -280)
ax1.set_yticks(np.arange(-700, -300+1, 100))
"""

ax0.set_ylabel('$\\Delta$ (kcal/mol)', fontsize=20,fontweight='bold')
ax1.set_ylabel('$E_{\\rm PhysNet}$ (kcal/mol)', fontsize=20,fontweight='bold')
ax1.set_xlabel('$E_{\\rm MP2}$ (kcal/mol)', fontsize=20,fontweight='bold')

#fam corr
ax1.scatter(data_all[:, 0], data_all[:, 1], marker="o",facecolors='none', edgecolors='#0d2f5c', s=7.5)
ax0.scatter(data_all[:, 0], data_all[:, 0] - data_all[:, 1], marker="o",facecolors='none', edgecolors='#0d2f5c', s=7.5)

plt.show()
quit()

plt.savefig('errcorr_physnet_vs_mp2.png',bbox_inches='tight', dpi=300)

plt.show()
quit()


