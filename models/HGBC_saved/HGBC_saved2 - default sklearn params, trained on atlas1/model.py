import os
from sys import path
import numpy as np
import pandas as pd
from math import sqrt, log
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import ensemble
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as stats
import matplotlib.pyplot as plt 
import pickle 
import json 



# ------------------------------
# Absolute path to submission dir
# ------------------------------
submissions_dir = os.path.dirname(os.path.abspath(__file__))
path.append(submissions_dir)


# ------------------------------
# Constants
# ------------------------------
EPSILON = np.finfo(float).eps


hist_analysis_dir = os.path.dirname(submissions_dir)
path.append(hist_analysis_dir)

# from hist_analysis_flt import compute_result, plot_score



# ------------------------------
# Baseline Model
# ------------------------------
class Model():
    """
    This is a model class to be submitted by the participants in their submission.

    This class should consists of the following functions
    1) init : initialize a classifier
    2) fit : can be used to train a classifier
    3) predict: predict mu_hats,  delta_mu_hat and q1,q2

    Note:   Add more methods if needed e.g. save model, load pre-trained model etc.
            It is the participant's responsibility to make sure that the submission 
            class is named "Model" and that its constructor arguments remains the same.
            The ingestion program initializes the Model class and calls fit and predict methods
    """

    def __init__(
            self,
            train_set=None,
            systematics=None
    ):
        """
        Model class constructor

        Params:
            train_set:
                labelled train set
                
            systematics:
                systematics class

        Returns:
            None
        """

        # Set class variables from parameters
        del systematics
        del train_set
        # Intialize class variables


    def fit(self):
        """
        Params:
            None

        Functionality:
            this function can be used to train a model using the train set

        Returns:
            None
        """

        self._read_model()
        self.plot_count = 0

        self.plot_file = os.path.join(submissions_dir, "Plots")
        if not os.path.exists(self.plot_file):
            os.makedirs(self.plot_file)


    def _return_score(self, X):
        y_predict = self.model.predict(X,verbose=0)
        y_predict = y_predict.pop(0)
        y_predict = y_predict.ravel()
        return np.array(y_predict)

    def predict(self, test_set):
        """
        Params:
            None

        Functionality:
           to predict using the test sets

        Returns:
            dict with keys
                - mu_hat
                - delta_mu_hat
                - p16
                - p84
        """
        
        print("[*] - Testing")
        test_df = test_set['data']
        test_df = self.scaler.transform(test_df)
        Y_hat_test = self._return_score(test_df)

        print("[*] - Computing Test result")
        weights_train = self.train_set["weights"].copy()
        weights_test = test_set["weights"].copy()

        print(f"[*] --- total weight test: {weights_test.sum()}") 
        print(f"[*] --- total weight train: {weights_train.sum()}")
        print(f"[*] --- total weight mu_cals_set: {self.holdout['weights'].sum()}")

        weight_clean = weights_test[Y_hat_test > self.threshold]
        test_df = test_set['data'][Y_hat_test > self.threshold]
        
        
        # get n_roi
        n_roi = (weight_clean.sum())

        mu_hat = (n_roi - self.beta_roi)/self.gamma_roi

        sigma_mu_hat = np.sqrt(n_roi)/self.gamma_roi

        delta_mu_hat = 2*sigma_mu_hat

        mu_p16 = mu_hat - sigma_mu_hat
        mu_p84 = mu_hat + sigma_mu_hat


        print(f"[*] --- mu_hat: {mu_hat.mean()}")
        print(f"[*] --- delta_mu_hat: {delta_mu_hat}")
        print(f"[*] --- p16: {mu_p16}")
        print(f"[*] --- p84: {mu_p84}")

        return {
            "mu_hat": mu_hat.mean(),
            "delta_mu_hat": delta_mu_hat,
            "p16": mu_p16,
            "p84": mu_p84
        }
    

    def _read_model(self):  
        print("[*] - Intialize Baseline Model (HGBC)")
        

        model_file = os.path.join(submissions_dir, "model.h5")
        settings_file = os.path.join(submissions_dir, "settings.pkl")
        scaler_file = os.path.join(submissions_dir, "scaler.pkl")

        self.model = pickle.load(open(model_file, 'rb'))
        
        settings = pickle.load(open(settings_file, "rb"))

        self.threshold = settings["threshold"]
        self.gamma_roi = settings["gamma_roi"]
        self.beta_roi = settings["beta_roi"]

        self.scaler = pickle.load(open(scaler_file, 'rb'))


        # self.bin_nums = settings["bin_nums"]
        # self.bins = np.linspace(0, 1, self.bin_nums + 1)
        # self.calibration = settings["calibration"]
        # self.control_bins = settings["control_bins"]
        # self.SYST = settings["SYST"]
        # if self.SYST:
        #     self.coef_s_list = settings["coef_s_list"]
        #     self.coef_b_list = settings["coef_b_list"]
        #     fit_line_s_list = []
        #     fit_line_b_list = []

        #     for coef_s_,coef_b_ in zip(self.coef_s_list,self.coef_b_list):

        #         coef_s = np.array(coef_s_)
        #         coef_b = np.array(coef_b_)

        #         fit_line_s_list.append(np.poly1d(coef_s))
        #         fit_line_b_list.append(np.poly1d(coef_b))

        #     self.fit_function_dict = {
        #         "gamma_roi":fit_line_s_list,
        #         "beta_roi":fit_line_b_list
        #     }
           
        # else:
        #     self.fit_function_dict = {
        #         "gamma_roi": settings["gamma_roi"],
        #         "beta_roi": settings["beta_roi"]
        #     }

        # self.fit_function_dict_control = {
        #     "gamma_roi": self.fit_function_dict["gamma_roi"][-self.control_bins:],
        #     "beta_roi": self.fit_function_dict["beta_roi"][-self.control_bins:]
        # }



        