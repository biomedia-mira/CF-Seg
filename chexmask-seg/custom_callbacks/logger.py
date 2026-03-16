import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn
import numpy as np
import pandas as pd
import os
from os.path import join
import matplotlib.pyplot as plt


class Logger_Single(object):

    def __init__(self, log_path="./log", log_name="training.log", metric_names=["loss","mean_squared_error", "mean_absolute_error"]):
        super(Logger_Single, self).__init__()

        self.log_path = log_path
        self.log_name = log_name
        self.metric_names = list(metric_names)

    def to_csv(self, metric_array, epoch):
        if epoch == 0:
            train_c = self.metric_names
            df = pd.DataFrame(columns=train_c)
            df.loc[0] = metric_array
        else:
            df = pd.read_csv(join(self.log_path,self.log_name), index_col=0)
            df.loc[len(df)] = metric_array
        df.to_csv(join(self.log_path,self.log_name))


########################################################################################################################################################

class Logger(object):

    def __init__(self, log_path="./log", log_name="training.log", metric_names=["loss","mean_squared_error", "mean_absolute_error"]):
        super(Logger, self).__init__()

        self.log_path = log_path
        self.log_name = log_name
        self.metric_names = list(metric_names)

    def to_csv(self, metric_array, epoch):
        if epoch == 0:
            train_c = self.metric_names
            val_c = ['val_'+t for t in train_c]
            df = pd.DataFrame(columns=train_c+val_c)
            df.loc[0] = metric_array
        else:
            df = pd.read_csv(join(self.log_path,self.log_name), index_col=0)
            df.loc[len(df)] = metric_array
        df.to_csv(join(self.log_path,self.log_name))

##############################################################################################################################################################

class Logger_test(object):

    def __init__(self, log_path="./log", log_name="training.log", metric_names=["loss","mean_squared_error", "mean_absolute_error"]):
        super(Logger_test, self).__init__()

        self.log_path = log_path
        self.log_name = log_name
        self.metric_names = list(metric_names)

    def to_csv(self, metric_array, epoch):
        if epoch == 0:
            train_c = self.metric_names
            val_c = ['val_'+t for t in train_c]
            test_c = ['test_'+t for t in train_c]
            df = pd.DataFrame(columns=train_c+val_c+test_c)
            df.loc[0] = metric_array
        else:
            df = pd.read_csv(join(self.log_path,self.log_name), index_col=0)
            df.loc[len(df)] = metric_array
        df.to_csv(join(self.log_path,self.log_name))