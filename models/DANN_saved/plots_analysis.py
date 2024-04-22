import os 
import matplotlib.pyplot as plt 
import pandas as pd 
import csv
import numpy as np
import time
from datetime import datetime as dt 
import mplhep as hep
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
events_dir = os.path.join(current_dir, "DANN_saved3 - 4_4_4_100_lambda50_epochs4_plots_optimization")
events_path = os.path.join(events_dir, "events.csv")

# df_events = pd.read_csv(events_path)
# print(df_events.values)


with open(events_path, "r") as f:
    data = list(csv.reader(f, delimiter="\t"))

# data = np.array(data)
# data = data[1:]

# theta_list = [data[i][0] for i in range(len(data))]
# print(theta_list)
# s_list = [data[i][1] for i in range(len(data))]
# print(theta_list)
# b_list = [data[i][2] for i in range(len(data))]
# print(theta_list)
# print(data)




