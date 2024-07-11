# from audioop import avg
# from distutils.log import error
# from turtle import distance
from turtle import position
import numpy as np
# import pandas as pdy7
# import math
import matplotlib.pyplot as plt
from load_data import *

from itertools import combinations
from scipy import interpolate


fig_size = (9,4)
font = {
        'font.weight' : 'normal',
        'font.size'   : 11}
plt.rcParams.update(font)

hydrus_list = ['mixed']
PINN_data = load_different_soils(hydrus_list, 'PINN')
Finite_diff_data = load_different_soils(hydrus_list, 'Finite_diff')
all_data = [Finite_diff_data, PINN_data]


# left column PINN, right column Finite diff
# Plot error by avg distances between sensors
fig, axs = plt.subplots(len(hydrus_list),2, sharex=True, sharey=False, figsize=fig_size, layout="constrained")
axs=np.expand_dims(axs,0)
for i in range(len(hydrus_list)):
    for j in range(2):
        
        x = all_data[j][hydrus_list[i]]['avg_distances']
        y = all_data[j][hydrus_list[i]]['avg_distances_errors']
        axs[i,j].plot(x, y, linewidth=3)
        name = ' '.join(hydrus_list[i].title().split('_'))
        # axs[i,j].set_title(name)
        axs[i,j].annotate('('+chr(ord('a')+2*i+j)+')', xy=(0.05, 0.85),
                xycoords='axes fraction',
                fontweight = 'bold', fontsize = 18)

fig.supxlabel('Distance [cm]')
fig.supylabel(r'$\epsilon^K$')
pad = 25
axs[0,0].annotate('Architecture 1', xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                ha='center', va='baseline', fontweight = 'bold', fontsize = 18)

axs[0,1].annotate('Architecture 2', xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                ha='center', va='baseline', fontweight = 'bold', fontsize = 18)

# fig.tight_layout()
plt.savefig('Mixed - Errors by avg distance between sensors.tiff', dpi=600, pil_kwargs={"compression": "tiff_lzw"})
# plt.savefig('Mixed - Errors by avg distance between sensors.png', dpi=600)

# Plot error of equidistant sensors   
fig, axs = plt.subplots(len(hydrus_list),2, sharex=True, sharey=False, figsize=fig_size, layout="constrained")
axs=np.expand_dims(axs,0)

for i in range(len(hydrus_list)):
    for j in range(2):
        
        x = all_data[j][hydrus_list[i]]['equidistances']
        y = all_data[j][hydrus_list[i]]['equidistances_errors']
        axs[i,j].plot(x, y, linewidth=3)
        name = ' '.join(hydrus_list[i].title().split('_'))
        # axs[i,j].set_title(name)
        axs[i,j].annotate('('+chr(ord('a')+2*i+j)+')', xy=(0.05, 0.85),
                xycoords='axes fraction',
                fontweight = 'bold', fontsize = 18)

fig.supxlabel('Distance [cm]')
fig.supylabel(r'$\epsilon^K$')
pad = 25
axs[0,0].annotate('Architecture 1', xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                ha='center', va='baseline', fontweight = 'bold', fontsize = 18)

axs[0,1].annotate('Architecture 2', xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                ha='center', va='baseline', fontweight = 'bold', fontsize = 18)

plt.savefig('Mixed - Errors of equidistant sensors.tiff', dpi=600, pil_kwargs={"compression": "tiff_lzw"})
# plt.savefig('Mixed - Errors of equidistant sensors.png', dpi=600)

# Plot error by avg depth  
fig, axs = plt.subplots(len(hydrus_list),2, sharex=True, sharey=False, figsize=fig_size, layout="constrained")
axs=np.expand_dims(axs,0)

for i in range(len(hydrus_list)):
    for j in range(2):
        
        x = all_data[j][hydrus_list[i]]['avg_depth']
        y = all_data[j][hydrus_list[i]]['avg_depth_errors']
        axs[i,j].plot(x, y, linewidth=3)
        name = ' '.join(hydrus_list[i].title().split('_'))
        # axs[i,j].set_title(name)
        axs[i,j].annotate('('+chr(ord('a')+2*i+j)+')', xy=(0.05, 0.85),
                xycoords='axes fraction',
                fontweight = 'bold', fontsize = 18)
fig.supxlabel('Distance [cm]')
fig.supylabel(r'$\epsilon^K$')
pad = 25
axs[0,0].annotate('Architecture 1', xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                ha='center', va='baseline', fontweight = 'bold', fontsize = 18)

axs[0,1].annotate('Architecture 2', xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                ha='center', va='baseline', fontweight = 'bold', fontsize = 18)

# fig.tight_layout()

plt.savefig('Mixed - Errors by avg depth.tiff', dpi=600, pil_kwargs={"compression": "tiff_lzw"})
# plt.savefig('Mixed - Errors by avg depth.png', dpi=600)

# Plot best config  
fig, axs = plt.subplots(len(hydrus_list),2, sharex=False, sharey=False, figsize=fig_size, layout="constrained")
axs=np.expand_dims(axs,0)

for i in range(len(hydrus_list)):
    for j in range(2):
        
        x = all_data[j][hydrus_list[i]]['ref_moist']
        y = all_data[j][hydrus_list[i]]['best_K']
        ref_K = all_data[j][hydrus_list[i]]['ref_K']
        axs[i,j].plot(x, y, linewidth=3, color = 'tab:blue', label='Estimated')
        axs[i,j].plot(x, ref_K, linewidth=3, color = 'r', label='Ground Truth')
        name = ' '.join(hydrus_list[i].title().split('_'))
        # axs[i,j].set_title(name)
        axs[i,j].annotate('('+chr(ord('a')+2*i+j)+')', xy=(0.05, 0.85),
                xycoords='axes fraction',
                fontweight = 'bold', fontsize = 18)
axs[0,0].legend(loc='upper center')
fig.supxlabel('Distance [cm]')
fig.supylabel(r'$K(\theta)\quad [cm/hr]$')

pad = 25
axs[0,0].annotate('Architecture 1', xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                ha='center', va='baseline', fontweight = 'bold', fontsize = 18)

axs[0,1].annotate('Architecture 2', xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                ha='center', va='baseline', fontweight = 'bold', fontsize = 18)

plt.savefig('Mixed - best config.tiff', dpi=600, pil_kwargs={"compression": "tiff_lzw"})
# plt.savefig('Mixed - best config.png', dpi=600)

# Plot error by middle sensor depth  
fig, axs = plt.subplots(len(hydrus_list),2, sharex=True, sharey=False, figsize=fig_size, layout="constrained")
axs=np.expand_dims(axs,0)

for i in range(len(hydrus_list)):
    for j in range(2):
        
        x = all_data[j][hydrus_list[i]]['s1_depth']
        y = all_data[j][hydrus_list[i]]['s1_depth_errors']
        axs[i,j].plot(x, y, linewidth=3)
        name = ' '.join(hydrus_list[i].title().split('_'))
        # axs[i,j].set_title(name)
        axs[i,j].annotate('('+chr(ord('a')+2*i+j)+')', xy=(0.05, 0.85),
                xycoords='axes fraction',
                fontweight = 'bold', fontsize = 18)
fig.supxlabel('Distance [cm]')
fig.supylabel(r'$\epsilon^K$')
pad = 25
axs[0,0].annotate('Architecture 1', xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                ha='center', va='baseline', fontweight = 'bold', fontsize = 18)

axs[0,1].annotate('Architecture 2', xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                ha='center', va='baseline', fontweight = 'bold', fontsize = 18)


plt.savefig('Mixed - Errors by middle sensor depth.tiff', dpi=600, pil_kwargs={"compression": "tiff_lzw"})
# plt.savefig('Mixed - Errors by middle sensor depth.png', dpi=600)


# Plot median K   
fig, axs = plt.subplots(len(hydrus_list),2, sharex=False, sharey=False, figsize=fig_size, layout="constrained")
axs=np.expand_dims(axs,0)

for i in range(len(hydrus_list)):
    for j in range(2):
        
        ref_moist = all_data[j][hydrus_list[i]]['ref_moist']
        ref_K = all_data[j][hydrus_list[i]]['ref_K']
        median_K = all_data[j][hydrus_list[i]]['median_K']
        first_quart_K = all_data[j][hydrus_list[i]]['first_quart_K']
        third_quart_K = all_data[j][hydrus_list[i]]['third_quart_K']
        axs[i,j].plot(ref_moist, median_K, color = 'tab:blue', linewidth=3, label='Median')
        axs[i,j].fill_between(ref_moist, first_quart_K, third_quart_K, color = [(0.30,0.66,1)], label=r'$1^{st} to 3^{rd} quartile$')
        axs[i,j].plot(ref_moist, ref_K, 'r', label='Ground Truth')
        name = ' '.join(hydrus_list[i].title().split('_'))
        
        axs[i,j].annotate('('+chr(ord('a')+2*i+j)+')', xy=(0.05, 0.85),
                xycoords='axes fraction',
                fontweight = 'bold', fontsize = 18)
axs[0,0].legend(loc='upper center')
fig.supxlabel(r'$\theta \quad [cm^3/cm^3]$')
fig.supylabel(r'$K(\theta)\quad [cm/hr]$')
pad = 25
axs[0,0].annotate('Architecture 1', xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                ha='center', va='baseline', fontweight = 'bold', fontsize = 18)

axs[0,1].annotate('Architecture 2', xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                ha='center', va='baseline', fontweight = 'bold', fontsize = 18)

plt.savefig('Mixed - median K.tiff', dpi=600, pil_kwargs={"compression": "tiff_lzw"})
# plt.savefig('Mixed - median K.png', dpi=600)

plt.show()

plt.pause(0.1)