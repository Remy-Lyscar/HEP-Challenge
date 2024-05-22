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
# hep.set_style("ATLAS")



# Making plot analysis after having registered the results from the predictions 
# by saved models 


# All data are saved by using pickle, it directly saves the dataframe structure 



# 1- Events for HGBC and DANN on the same plot (for the best threshold, it's better)
# 2- Zmax (do some plots in the region close to the best threshold to be more precise)
# 3- best lambda (see later)



current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)



# 1- Events for HGBC and DANN on the same plot (for the best threshold, it's better)

# events_DANN_datafile = os.path.join(current_dir, "events.pkl")
# events_DANN_data = pickle.load(open(events_DANN_datafile, 'rb'))


# events_HGBC_datafile = os.path.join(parent_dir, "HGBC_saved", "HGBC_saved3 - default sklearn params, threshold = 0.952", "events.pkl")
# events_HGBC_data = pickle.load(open(events_HGBC_datafile, 'rb'))


# theta_list = events_HGBC_data["theta_list"]
# s_list_HGBC = events_HGBC_data["s_list"]
# b_list_HGBC = events_HGBC_data["b_list"]

# s_list_DANN = events_DANN_data["s_list"]
# b_list_DANN = events_DANN_data["b_list"]

# std_s_HGBC = np.std(s_list_HGBC, ddof= 1)
# std_s_DANN = np.std(s_list_DANN, ddof= 1)
# std_b_HGBC = np.std(b_list_HGBC, ddof= 1)
# std_b_DANN = np.std(b_list_DANN, ddof= 1)


# fig_s = plt.figure()
# plt.plot(theta_list, s_list_DANN, 'r.', label = f'DANN: std = {std_s_DANN:.5f}')
# plt.plot(theta_list, s_list_HGBC, 'b.', label = f'HGBC: std = {std_s_HGBC:.5f}')
# plt.xlabel('TES')
# plt.ylabel('Number of signal events in the ROI')
# # hep.atlas.text(loc=1, text = " ")
# plt.legend(loc = 'lower right')

# plot_file_s = os.path.join(current_dir, "s_events_comparison")


# plt.savefig(plot_file_s)
# plt.close(fig_s)


# fig_b = plt.figure()
# plt.plot(theta_list, b_list_DANN, 'r.', label = f'DANN: std = {std_b_DANN:.5f}')
# plt.plot(theta_list, b_list_HGBC, 'b.', label = f'HGBC: std = {std_b_HGBC:.5f}')
# plt.xlabel('TES')
# plt.ylabel('Number of background events in the ROI')
# # hep.atlas.text(loc=1, text = " ")
# plt.legend(loc = 'best')

# plot_file_b = os.path.join(current_dir, "b_events_comparison")


# plt.savefig(plot_file_b)
# plt.close(fig_b)


# #1bis - Events comparisn, relative fluctuation aroung mean -> better estimator of the stability of the model


# mean_s_HGBC = np.mean(s_list_HGBC)
# mean_s_DANN = np.mean(s_list_DANN)
# mean_b_DANN = np.mean(b_list_DANN)
# mean_b_HGBC = np.mean(b_list_HGBC)


# s_DANN = [s/mean_s_DANN for s in s_list_DANN]
# s_HGBC = [s/mean_s_HGBC for s in s_list_HGBC]
# b_DANN = [b/mean_b_DANN for b in b_list_DANN]
# b_HGBC = [b/mean_b_HGBC for b in b_list_HGBC]

# std_s_HGBC = np.std(s_HGBC, ddof= 1)
# std_s_DANN = np.std(s_DANN, ddof= 1)
# std_b_HGBC = np.std(b_HGBC, ddof= 1)
# std_b_DANN = np.std(b_DANN, ddof= 1)


# fig_s = plt.figure()
# plt.plot(theta_list, s_DANN, 'r.', label = f'DANN: std = {std_s_DANN:.5f}')
# plt.plot(theta_list, s_HGBC, 'b.', label = f'HGBC: std = {std_s_HGBC:.5f}')
# plt.xlabel('TES')
# plt.ylabel('delta_s')
# # hep.atlas.text(loc=1, text = " ")
# plt.legend(loc = 'best')

# plot_file_s = os.path.join(current_dir, "delta_s_events_comparison")


# plt.savefig(plot_file_s)
# plt.close(fig_s)


# fig_b = plt.figure()
# plt.plot(theta_list, b_DANN, 'r.', label = f'DANN: std = {std_b_DANN:.5f}')
# plt.plot(theta_list, b_HGBC, 'b.', label = f'HGBC: std = {std_b_HGBC:.5f}')
# plt.xlabel('TES')
# plt.ylabel('delta_b')
# # hep.atlas.text(loc=1, text = " ")
# plt.legend(loc = 'best')

# plot_file_b = os.path.join(current_dir, "delta_b_events_comparison")


# plt.savefig(plot_file_b)
# plt.close(fig_b)


# 2- Zmax (do some plots in the region close to the best threshold to be more precise)



# threshold_data_file = os.path.join(current_dir, "threshold.pkl")
# threshold_data = pickle.load(open(threshold_data_file, "rb"))

# threshold_list = threshold_list = np.linspace(0.8, 1, 20)
# Z_list = threshold_data["significance regarding threshold for TES = 1.03"]

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
# plt.title(f"TES = 1.03")
# # hep.atlas.text(loc=1, text = " ")

# annot_max(threshold_list, Z_list)


# plot_file_Z_theshold = os.path.join(current_dir, "DANN_Z_threshold_analysis_TES=1.03.png")


# plt.savefig(plot_file_Z_theshold)
# plt.close(fig_Z_threshold)



# plots of delta_mus as a function of lambda




deltamu_file50 = os.path.join(current_dir, "DANN_saved4 - 4_4_4_100_lambda50_epochs4_threshold = 0.945, old_div", "delta_mus.pkl")
deltamu_file10 = os.path.join(current_dir, "DANN_saved5 - lambda = 10_threshold= 0.960 , new_div", "delta_mus.pkl")
deltamu_file1= os.path.join(current_dir, "DANN_saved6 - lambda1_threshold= , new div", "delta_mus.pkl")
deltamu_file100 = os.path.join(current_dir, "DANN_saved6_lambda100_threshold =0.958 , nediv", "delta_mus.pkl")

deltamu_data50 = pickle.load(open(deltamu_file50, "rb"))
deltamu_data1 = pickle.load(open(deltamu_file1, "rb"))
deltamu_data100 = pickle.load(open(deltamu_file100, "rb"))


deltamu_stat50 = deltamu_data50["delta_mu_stat"]
deltamu_syst50 = deltamu_data50["delta_mu_syst"]
deltamu_tot50 = deltamu_data50["delta_mu_tot"]

deltamu_stat1 = deltamu_data1["delta_mu_stat"]
deltamu_syst1 = deltamu_data1["delta_mu_syst"]
deltamu_tot1 = deltamu_data1["delta_mu_tot"]

deltamu_stat100 = deltamu_data100["delta_mu_stat"]
deltamu_syst100 = deltamu_data100["delta_mu_syst"]
deltamu_tot100 = deltamu_data100["delta_mu_tot"]

lambda_list = [1, 50, 100]

stat_list = [deltamu_stat1, deltamu_stat50, deltamu_stat100]
syst_list = [deltamu_syst1, deltamu_syst50, deltamu_syst100]
tot_list = [deltamu_tot1, deltamu_tot50, deltamu_tot100]


fig_stat = plt.figure()
plt.plot(lambda_list, stat_list)
plt.xlabel('lambdas')
plt.ylabel('delta_mu_stat')
# plt.legend(loc = 'lower right')
# hep.atlas.text(loc=1, text = " ")


plot_file_stat = os.path.join(current_dir, "stat_lambda.png")


plt.savefig(plot_file_stat)
plt.close(fig_stat)


fig_syst = plt.figure()
plt.plot(lambda_list, syst_list)
plt.xlabel('lambdas')
plt.ylabel('delta_mu_syst')
# plt.legend(loc = 'lower right')
# hep.atlas.text(loc=1, text = " ")


plot_file_syst = os.path.join(current_dir, "syst_lambda.png")


plt.savefig(plot_file_syst)
plt.close(fig_syst)


fig_tot = plt.figure()
plt.plot(lambda_list, tot_list)
plt.xlabel('lambdas')
plt.ylabel('delta_mu_tot')
# plt.legend(loc = 'lower right')
# hep.atlas.text(loc=1, text = " ")


plot_file_tot = os.path.join(current_dir, "tot_lambda.png")


plt.savefig(plot_file_tot)
plt.close(fig_tot)