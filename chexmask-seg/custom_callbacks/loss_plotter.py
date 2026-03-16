import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import os
from os.path import join
import matplotlib.pyplot as plt

class LossPlotter(object):

    def __init__(self, log_path="./log", log_name="training.log", metric_names=["loss","mean_squared_error","mean_absolute_error"]):
        super(LossPlotter, self).__init__()
        self.log_path = log_path
        self.log_name = log_name
        self.metric_names = list(metric_names)
        os.makedirs(join(self.log_path, "plot"), exist_ok=True)

    def plotter(self):

        dataframe = pd.read_csv(join(self.log_path,self.log_name), skipinitialspace=True)

        for i in range(len(self.metric_names)):
            plt.figure(i)
            plt.plot(dataframe[self.metric_names[i]],label="train_"+self.metric_names[i])
            plt.plot(dataframe["val_"+self.metric_names[i]],label="val_"+self.metric_names[i])
            plt.grid(True)
            plt.legend()
            plt.savefig(join(self.log_path,"plot",self.metric_names[i]+".png"))
            plt.close()