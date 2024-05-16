import os 
import matplotlib.pyplot as plt 
import pandas as pd 
import csv
import numpy as np
import time
from datetime import datetime as dt 
import mplhep as hep
import pickle 
import json
hep.set_style("ATLAS")



# Making plot analysis after having registered the results from the predictions 
# by saved models 


# All data are saved by using pickle, it directly saves the dataframe structure 



# 1- Events for HGBC and DANN on the same plot (for the best threshold, it's better)
# 2- Zmax (do some plots in the region close to the best threshold to be more precise)
# 3- best lambda (see later)
# 4- average of the Zmax over three TES : 0.97 1.00 and 1.03
# 5 - Hist analysis 




# 2- Zmax (do some plots in the region close to the best threshold to be more precise)

current_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(current_dir, "threshold.pkl")


threshold_data = pickle.load(open(data_file, "rb"))

threshold_list = threshold_list = np.linspace(0.85, 0.95, 20)
Z_list = threshold_data["significance regarding threshold for TES = 0.97"]

def annot_max(x,y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text= "threshold={:.3f}, Z_max={:.3f}".format(xmax, ymax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    # arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    # kw = dict(xycoords='data',textcoords="axes fraction",
    #           arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    kw = dict(xycoords='data',textcoords="axes fraction", bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.5,0.1), **kw)
    
    x_bounds = ax.get_xbound()
    y_bounds = ax.get_ybound()
    ax.set_xlim(x_bounds[0], x_bounds[1])
    ax.set_ylim(y_bounds[0], y_bounds[1])
    plt.vlines(xmax, y_bounds[0], ymax, colors = 'r', linestyles='dashed')
    plt.hlines(ymax, x_bounds[0] ,xmax, colors='r', linestyles='dashed')




fig_Z_threshold = plt.figure()
plt.plot(threshold_list, Z_list, 'b.')
plt.xlabel('threshold')
plt.ylabel('Significance')
# plt.legend(loc = 'lower right')
plt.title(f"TES = 0.97")
hep.atlas.text(loc=1, text = " ")

annot_max(threshold_list, Z_list)


plot_file_Z_theshold = os.path.join(current_dir, "DANN_Z_threshold_analysis_TES=0.97.png")


plt.savefig(plot_file_Z_theshold)
plt.close(fig_Z_threshold)