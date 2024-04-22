import os 
import matplotlib.pyplot as plt 
import pandas as pd 
import time
from datetime import datetime as dt 
import mplhep as hep
hep.set_style("ATLAS")



# Making plot analysis after having registered the results from the predictions 
# by saved models 

# 1- Events for HGBC and DANN on the same plot (for the best threshold, it's better)
# 2- Zmax (do some plots in the region close to the best threshold to be more precise)
# 3- best lambda (see later)
# 4- average of the Zmax over three TES : 0.97 1.00 and 1.03
# 5 - Hist analysis 






