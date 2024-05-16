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



current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)



# 1- Events for HGBC and DANN on the same plot (for the best threshold, it's better)

events_DANN_datafile = os.path.join(current_dir, "events.pkl")
events_DANN_data = pickle.load(open(events_DANN_datafile, 'rb'))


events_HGBC_datafile = os.path.join(parent_dir, "HGBC_saved", "events.pkl")
events_HGBC_data = pickle.load(open(events_HGBC_datafile, 'rb'))


theta_list = events_HGBC_data["theta_list"]
s_list_HGBC = events_HGBC_data["s_list"]
b_list_HGBC = events_HGBC_data["b_list"]

s_list_DANN = events_DANN_data["s_list"]
b_list_DANN = events_DANN_data["b_list"]

std_s_HGBC = np.std(s_list_HGBC, ddof= 1)
std_s_DANN = np.std(s_list_DANN, ddof= 1)
std_b_HGBC = np.std(b_list_HGBC, ddof= 1)
std_b_DANN = np.std(b_list_DANN, ddof= 1)


fig_s = plt.figure()
plt.plot(theta_list, s_list_DANN, 'r.', label = f'DANN: std = {std_s_DANN:.5f}')
plt.plot(theta_list, s_list_HGBC, 'b.', label = f'HGBC: std = {std_s_HGBC:.5f}')
plt.xlabel('TES')
plt.ylabel('Number of signal events in the ROI')
hep.atlas.text(loc=1, text = " ")
plt.legend(loc = 'best')

plot_file_s = os.path.join(current_dir, "s_events_comparison")


plt.savefig(plot_file_s)
plt.close(fig_s)


fig_b = plt.figure()
plt.plot(theta_list, b_list_DANN, 'r.', label = f'DANN: std = {std_b_DANN:.5f}')
plt.plot(theta_list, b_list_HGBC, 'b.', label = f'HGBC: std = {std_b_HGBC:.5f}')
plt.xlabel('TES')
plt.ylabel('Number of background events in the ROI')
hep.atlas.text(loc=1, text = " ")
plt.legend(loc = 'best')

plot_file_b = os.path.join(current_dir, "b_events_comparison")


plt.savefig(plot_file_b)
plt.close(fig_b)



# 2- Zmax (do some plots in the region close to the best threshold to be more precise)



# threshold_data_file = os.path.join(current_dir, "threshold.pkl")
# threshold_data = pickle.load(open(threshold_data_file, "rb"))

# threshold_list = threshold_list = np.linspace(0.85, 0.95, 20)
# Z_list = threshold_data["significance regarding threshold for TES = 0.97"]

# def annot_max(x,y, ax=None):
#     xmax = x[np.argmax(y)]
#     ymax = y.max()
#     text= "threshold={:.3f}, Z_max={:.3f}".format(xmax, ymax)
#     if not ax:
#         ax=plt.gca()
#     bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
#     # arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
#     # kw = dict(xycoords='data',textcoords="axes fraction",
#     #           arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
#     kw = dict(xycoords='data',textcoords="axes fraction", bbox=bbox_props, ha="right", va="top")
#     ax.annotate(text, xy=(xmax, ymax), xytext=(0.5,0.1), **kw)
    
#     x_bounds = ax.get_xbound()
#     y_bounds = ax.get_ybound()
#     ax.set_xlim(x_bounds[0], x_bounds[1])
#     ax.set_ylim(y_bounds[0], y_bounds[1])
#     plt.vlines(xmax, y_bounds[0], ymax, colors = 'r', linestyles='dashed')
#     plt.hlines(ymax, x_bounds[0] ,xmax, colors='r', linestyles='dashed')




# fig_Z_threshold = plt.figure()
# plt.plot(threshold_list, Z_list, 'b.')
# plt.xlabel('threshold')
# plt.ylabel('Significance')
# # plt.legend(loc = 'lower right')
# plt.title(f"TES = 0.97")
# hep.atlas.text(loc=1, text = " ")

# annot_max(threshold_list, Z_list)


# plot_file_Z_theshold = os.path.join(current_dir, "DANN_Z_threshold_analysis_TES=0.97.png")


# plt.savefig(plot_file_Z_theshold)
# plt.close(fig_Z_threshold)