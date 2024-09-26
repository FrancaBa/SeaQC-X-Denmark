#####################################################
## Preprocessing sea level tide gauge data by frb  ##
#####################################################
import os, sys
import pdb
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from cotede import datasets, qctests
from scipy.signal import argrelextrema

import source.helper_methods as helper

class PreProcessor():
    
    def __init__(self):
        self.missing_meas_value = 999.000
        self.helper = helper.HelperMethods()

    def set_output_folder(self, folder_path):

        self.folder_path = folder_path 

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.helper.set_output_folder(folder_path)

    def read_data(self, path, file):
        self.df_meas = pd.read_csv(os.path.join(path,file), sep="\s+", header=None, names=['Timestamp', 'WaterLevel', 'Flag'])
        self.df_meas = self.df_meas[['Timestamp', 'WaterLevel']]
        self.df_meas['Timestamp'] = pd.to_datetime(self.df_meas['Timestamp'], format='%Y%m%d%H%M%S')
        
        #drop seconds
        self.df_meas['Timestamp'] = self.df_meas['Timestamp'].dt.round('min')

        #Initial screening of measurements:
        #1. Invalid characters: Set non-float elements to NaN
        # Count the original NaN values
        original_nan_count = self.df_meas['WaterLevel'].isna().sum()
        self.df_meas['WaterLevel'] = self.df_meas['WaterLevel'].apply(lambda x: x if isinstance(x, float) else np.nan)
        # Count the new NaN values after the transformation -> Where there any invalid characters?
        new_nan_count = self.df_meas['WaterLevel'].isna().sum() - original_nan_count
        print('The measurement series contained',new_nan_count,'invalid data entires.')
        
        #2. Replace the missing values with nan
        self.df_meas.loc[self.df_meas['WaterLevel'] == self.missing_meas_value, 'WaterLevel'] = np.nan

        self.helper.plot_df(self.df_meas['Timestamp'], self.df_meas['WaterLevel'],'Water Level','Timestamp','Measured water level')
        
        #Subsequent line makes the code slow, only enable when needed
        #self.helper.zoomable_plot_df(self.df_meas['Timestamp'], self.df_meas['WaterLevel'],'Water Level','Timestamp', 'Measured water level','measured water level')
        

    def check_timestamp(self):

        #Generate a new ts in 1 min timestamp
        start_time = self.df_meas['Timestamp'].iloc[0]
        end_time = self.df_meas['Timestamp'].iloc[-1]
        ts_full = pd.date_range(start= start_time, end= end_time, freq='min').to_frame(name='Timestamp')

        #Merge df based on timestamp and plot the outcome
        #Measurements from Greenland have no winter/ summer shift
        self.df_meas_long = pd.merge(ts_full, self.df_meas, on='Timestamp', how = 'outer')
        
        #Check specific lines
        #self.df_meas_long[self.df_meas_long['Timestamp'] == '2009-08-13 00:37:00']

        #plot with new ts
        self.df_meas_long_filled = self.df_meas_long.fillna(0)
        self.helper.plot_df(self.df_meas_long['Timestamp'], self.df_meas_long_filled['WaterLevel'],'Water Level','Timestamp','Measured water level in 1 min timestamp (nans filled with 0 for plot)')

    def remove_stat_outliers(self):

        # Quantile Detection for large range outliers
        # Calculate the interquartile range (IQR)
        Q1 = self.df_meas_long.quantile(0.25)
        Q3 = self.df_meas_long.quantile(0.75)
        IQR = Q3 - Q1

        # Define the lower and upper bounds for outlier detection
        # normally it is lower_bound = Q1 - 1.5 * IQR and upper_bound = Q3 + 1.5 * IQR
        # We can set them to even larger range to only tidal analysis, coz outlier detection is after tidal analysis 
        lower_bound = Q1 - 2.5 * IQR
        upper_bound = Q3 + 2.5 * IQR

        # Detect outliers and set them to NaN specifically for the 'WaterLevel' column
        self.df_meas_long['WaterLevel'] = self.df_meas_long['WaterLevel'].mask((self.df_meas_long['WaterLevel'] < lower_bound['WaterLevel']) | (self.df_meas_long['WaterLevel'] > upper_bound['WaterLevel'])) 
        
        #plot results without outliers
        self.df_meas_long_filled = self.df_meas_long.fillna(0)
        self.helper.plot_df(self.df_meas_long['Timestamp'], self.df_meas_long_filled['WaterLevel'],'Water Level','Timestamp','Measured water level wo outliers in 1 min timestamp (nans filled with 0 for plot)')

    def remove_spikes(self):
        # The spike check is a quite traditional one and is based on the principle of comparing one measurement with the tendency observed from the neighbor values.
        # This is already implemented in CoTeDe as qctests.spike

        # If no threshold is required, you can do like this:
        sea_level_spike = qctests.spike(self.df_meas_long["WaterLevel"])  
        print("The largest spike observed was: {:.3f}".format(np.nanmax(np.abs(sea_level_spike))))
        print("Value at detected position", sea_level_spike[np.nanargmax(np.abs(sea_level_spike))])

        #Assessment of max spike via plots
        min_value = np.nanargmax(np.abs(sea_level_spike)) - 2000
        max_value = np.nanargmax(np.abs(sea_level_spike)) + 2000
        self.helper.plot_df(self.df_meas_long['Timestamp'][min_value:max_value], self.df_meas_long['WaterLevel'][min_value:max_value],'Water Level','Timestamp','Measured water level wo outliers in 1 min timestamp (zoomed to max spike)')
        self.helper.plot_df(self.df_meas_long['Timestamp'], sea_level_spike,'Detected spike [m]','Timestamp','WL spikes in measured ts')

        # If the spike is less than 1 cm (absolut value), it is not a spike -> Do not remove this small noise
        #sea_level_spike[abs(sea_level_spike) < 0.01] = 0.0
        sea_level_spike[abs(sea_level_spike) < 0.1] = 0.0
        #Plot together
        self.helper.plot_two_df(self.df_meas_long['Timestamp'][min_value:max_value], sea_level_spike[min_value:max_value],'Detected spikes', self.df_meas_long['WaterLevel'][min_value:max_value], 'Water Level', 'Timestamp','Spikes and measured WL')

        #balance out spike
        for index, value in enumerate(sea_level_spike):
            if abs(value) > 0:
                spike = self.df_meas_long['WaterLevel'][index]
                spike_bound1 = self.df_meas_long['WaterLevel'][index+20]
                spike_bound2 = self.df_meas_long['WaterLevel'][index-20]
                bound = (spike_bound1+spike_bound2)/2
                if spike > bound:
                    self.df_meas_long['WaterLevel'][index] = self.df_meas_long['WaterLevel'][index] - abs(value)
                else:
                    self.df_meas_long['WaterLevel'][index] = self.df_meas_long['WaterLevel'][index] + abs(value)
        
        self.helper.plot_df(self.df_meas_long['Timestamp'][min_value:max_value], self.df_meas_long['WaterLevel'][min_value:max_value],'Water Level','Timestamp','Measured water level wo outliers and spikes in 1 min timestamp (zoomed to max spike) - Step 20')
        #Subsequent line makes the code slow, only enable when needed
        #self.helper.zoomable_plot_df(self.df_meas_long['Timestamp'], self.df_meas_long['WaterLevel'],'Water Level','Timestamp','Spike analysis', 'measured water level',self.df_meas_long['Timestamp'], sea_level_spike,'Detected spikes')