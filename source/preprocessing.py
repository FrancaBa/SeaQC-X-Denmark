#####################################################
## Preprocessing sea level tide gauge data by frb  ##
#####################################################
import os, sys
import pdb
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class PreProcessor():
    
    #def __init__(self):

    def read_data(self, path, file):
        self.df_meas = pd.read_csv(os.path.join(path,file), sep="\s+", header=None, names=['Timestamp', 'WaterLevel', 'Flag'])
        self.df_meas['Timestamp'] = pd.to_datetime(self.df_meas['Timestamp'], format='%Y%m%d%H%M%S')
        #WL_data.loc[WL_data['WaterLevel'] == 999.000, 'WaterLevel'] = np.nan

    def check_timestamp(self):

        #Generate a new ts in 1 min timestamp
        start_time = self.df_meas['Timestamp'].iloc[0]
        end_time = self.df_meas['Timestamp'].iloc[-1]
        ts_full = pd.date_range(start= start_time, end= end_time, freq='min').to_frame(name='Timestamp')

        #Merge df based on timestamp and plot the outcome
        df_meas_long = ts_full.merge(self.df_meas[['WaterLevel','Timestamp']], on='Timestamp', how = 'left')
        self.plot_df(df_meas_long['WaterLevel'],'Water Level','Timestamp','Measured water level')

    def remove_stat_outliers(self):
        y = 1
        return y
    
    def plot_df(self, data, y_title, x_title, title = None):

        plt.figure(figsize=(14, 7))
        plt.plot(data)
        if title != None:
            plt.title(title)
        plt.xlabel(x_title)
        plt.ylabel(y_title)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
