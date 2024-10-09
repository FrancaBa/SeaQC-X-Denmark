#####################################################
## Preprocessing sea level tide gauge data by frb  ##
#####################################################
import os, sys
import pdb
import datetime
import numpy as np
from numpy import ma
import matplotlib.pyplot as plt
import pandas as pd
import json

from cotede import datasets, qctests
from scipy.signal import argrelextrema

import source.helper_methods as helper

class QualityFlagger():
    
    def __init__(self):
        self.missing_meas_value = 999.000
        self.window_constant_value = 10
        self.helper = helper.HelperMethods()

    def load_qf_classification(self, json_path):

        # Open and load JSON file containing the quality flag classification
        with open(json_path, 'r') as file:
            config_data = json.load(file)

        self.qf_classes = config_data['qc_flag_classification']


    def set_output_folder(self, folder_path):

        self.folder_path = folder_path 

        #generate output folder for graphs and other docs
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.helper.set_output_folder(folder_path)

    def read_data(self, path, file):
        self.df_meas = pd.read_csv(os.path.join(path,file), sep="\s+", header=None, names=['Timestamp', 'WaterLevel', 'Flag'])
        self.df_meas = self.df_meas[['Timestamp', 'WaterLevel']]
        self.df_meas['Timestamp'] = pd.to_datetime(self.df_meas['Timestamp'], format='%Y%m%d%H%M%S')
        
        #drop seconds
        self.df_meas['Timestamp'] = self.df_meas['Timestamp'].dt.round('min')

        # Extract the resolution in seconds, minutes, etc.
        # Here, we extract the time difference in minutes
        self.df_meas['time_diff'] = self.df_meas['Timestamp'].diff()
        self.df_meas['resolution'] = self.df_meas['time_diff'].dt.total_seconds()/60
        self.df_meas.loc[self.df_meas['resolution'] > 3600, 'resolution'] = np.nan
        # Save the DataFrame to a text file (tab-separated)
        self.df_meas.to_csv(os.path.join(self.folder_path,'time_series_with_resolution.txt'), sep='\t', index=False)
        self.helper.plot_df(self.df_meas['Timestamp'][-2000:-1000], self.df_meas['resolution'][-2000:-1000],'step size','Timestamp','Time resolution (zoomed)')
        self.helper.plot_df(self.df_meas['Timestamp'][33300:33320], self.df_meas['resolution'][33300:33320],'step size','Timestamp','Time resolution (zoomed 2)')
        self.helper.plot_df(self.df_meas['Timestamp'], self.df_meas['resolution'],'step size','Timestamp','Time resolution')

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
        #print('length of ts:', len(self.df_meas))
        #self.helper.zoomable_plot_df(self.df_meas['Timestamp'], self.df_meas['WaterLevel'],'Water Level','Timestamp', 'Measured water level','measured water level')
        
    def check_timestamp(self):

        #Generate a new ts in 1 min timestamp
        start_time = self.df_meas['Timestamp'].iloc[0]
        end_time = self.df_meas['Timestamp'].iloc[-1]
        ts_full = pd.date_range(start= start_time, end= end_time, freq='min').to_frame(name='Timestamp').reset_index(drop=True)

        #Merge df based on timestamp and plot the outcome
        self.df_meas_long = pd.merge(ts_full, self.df_meas, on='Timestamp', how = 'outer')
        self.df_meas_long['resolution'] = self.df_meas_long['resolution'].bfill()

        #Check specific lines
        print('The new ts is',len(self.df_meas_long),'entries long.')
        print(self.df_meas_long[self.df_meas_long['Timestamp'] == '2005-10-24 03:00:00'])
        #print(self.df_meas_long.index[self.df_meas_long['Timestamp'] == '2005-10-24 03:00:00'].to_list())
        print(self.df_meas_long[33300:33320])
        print(self.df_meas_long[:20])

        #plot with new 1-min ts
        self.helper.plot_df(self.df_meas_long['Timestamp'], self.df_meas_long['WaterLevel'],'Water Level','Timestamp','Measured water level in 1 min timestamp')
        self.helper.plot_df(self.df_meas_long['Timestamp'][33300:33400], self.df_meas_long['WaterLevel'][33300:33400],'Water Level','Timestamp','Measured water level in 1 min timestamp (zoom)')
        #self.helper.zoomable_plot_df(self.df_meas_long['Timestamp'][:33600], self.df_meas_long_filled['WaterLevel'][:33600],'Water Level','Timestamp', 'Measured water level time','measured water level time')
    
    def detect_constant_value(self):

        # Check if the value is constant over a window of 'self.window_constant_value' min (counts only non-nan)
        constant_mask = self.df_meas['WaterLevel'].rolling(window=self.window_constant_value).apply(lambda x: x.nunique() == 1, raw=True)
        # Mask the constant values
        self.df_meas_long['altered'] = np.where(constant_mask, np.nan, self.df_meas_long['Timestamp'])
        self.df_meas['masked_constant'] = np.where(constant_mask, self.qf_classes['bad_data'], self.qf_classes['good_data'])
        self.helper.plot_df(self.df_meas['masked_constant'], self.df_meas['WaterLevel'],'Water Level','Timestamp','Measured water level')
        print(self.df_meas['masked'])
    
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
        self.helper.plot_df(self.df_meas_long['Timestamp'], self.df_meas_long['WaterLevel'],'Water Level','Timestamp','Measured water level wo outliers in 1 min timestamp')
        self.helper.plot_df(self.df_meas_long['Timestamp'][33300:33400], self.df_meas_long['WaterLevel'][33300:33400],'Water Level','Timestamp','Measured water level in 1 min timestamp wo outliers(zoom)')
        
        #Subsequent line makes the code slow, only enable when needed
        #self.helper.zoomable_plot_df(self.df_meas_long['Timestamp'][:1000000], self.df_meas_long['WaterLevel'][:1000000],'Water Level','Timestamp', 'Measured water level wo outliers','measured water level')
        
    def remove_spikes_cotede(self):
        # The spike check is a quite traditional one and is based on the principle of comparing one measurement with the tendency observed from the neighbor values.
        # This is already implemented in CoTeDe as qctests.spike
        # Revelant weaknesses:
        # 1. Cannot detect spikes when NaN in the neighbourhood
        # 2. Cannot detect smaller, continious spiking behaviour

        # If no threshold is required, you can do like this:
        sea_level_spike = qctests.spike(self.df_meas_long["WaterLevel"])  
        print("The largest spike observed was: {:.3f}".format(np.nanmax(np.abs(sea_level_spike))))
        print("Value at detected position", sea_level_spike[np.nanargmax(np.abs(sea_level_spike))])

        #Assessment of max spike via plots
        min_value = np.nanargmax(np.abs(sea_level_spike)) - 500
        max_value = np.nanargmax(np.abs(sea_level_spike)) + 1500
        self.helper.plot_df(self.df_meas_long['Timestamp'][min_value:max_value], self.df_meas_long['WaterLevel'][min_value:max_value],'Water Level','Timestamp','Measured water level wo outliers in 1 min timestamp (zoomed to max spike)')
        self.helper.plot_df(self.df_meas_long['Timestamp'], sea_level_spike,'Detected spike [m]','Timestamp','WL spikes in measured ts')

        # If the spike is less than 1 cm (absolut value), it is not a spike -> Do not remove this small noise
        sea_level_spike[abs(sea_level_spike) < 0.01] = 0.0
        #Plot together
        self.helper.plot_two_df(self.df_meas_long['Timestamp'][min_value:max_value], sea_level_spike[min_value:max_value],'Detected spikes', self.df_meas_long['WaterLevel'][min_value:max_value], 'Water Level', 'Timestamp','Spikes and measured WL')

        #balance out spike
        for index, value in enumerate(sea_level_spike):
            if abs(value) > 0:
                spike = self.df_meas_long['WaterLevel'][index]
                spike_bound1 = self.df_meas_long['WaterLevel'][index+1]
                spike_bound2 = self.df_meas_long['WaterLevel'][index-1]
                bound = (spike_bound1+spike_bound2)/2
                if spike > bound:
                    self.df_meas_long['WaterLevel'][index] = self.df_meas_long['WaterLevel'][index] - abs(value)
                else:
                   self.df_meas_long['WaterLevel'][index] = self.df_meas_long['WaterLevel'][index] + abs(value)
            
        self.helper.plot_df(self.df_meas_long['Timestamp'][min_value:max_value], self.df_meas_long['WaterLevel'][min_value:max_value],'Water Level','Timestamp','Measured water level wo outliers and spikes in 1 min timestamp (zoomed to max spike)')
        self.helper.plot_df(self.df_meas_long['Timestamp'], self.df_meas_long['WaterLevel'],'Water Level','Timestamp','Measured water level wo outliers and spikes in 1 min timestamp')
        #Some more plots for assessment
        self.helper.plot_df(self.df_meas_long['Timestamp'][:1000000], self.df_meas_long['WaterLevel'][:1000000],'Water Level','Timestamp','Measured water level wo outliers and spikes in 1 min timestamp 1')
        self.helper.plot_df(self.df_meas_long['Timestamp'][1000000:2000000], self.df_meas_long['WaterLevel'][1000000:2000000],'Water Level','Timestamp','Measured water level wo outliers and spikes in 1 min timestamp2')
        self.helper.plot_df(self.df_meas_long['Timestamp'][2000000:3000000], self.df_meas_long['WaterLevel'][2000000:3000000],'Water Level','Timestamp','Measured water level wo outliers and spikes in 1 min timestamp3')
        self.helper.plot_df(self.df_meas_long['Timestamp'][3000000:4000000], self.df_meas_long['WaterLevel'][3000000:4000000],'Water Level','Timestamp','Measured water level wo outliers and spikes in 1 min timestamp4')
        #ubsequent line makes the code slow, only enable when need
        #self.helper.zoomable_plot_df(self.df_meas_long['Timestamp'][:1000000], self.df_meas_long['WaterLevel'][:1000000],'Water Level','Timestamp','Measured water level wo outliers and spikes in 1 min timestamp', 'measured water level')
        #self.helper.zoomable_plot_df(self.df_meas_long['Timestamp'][:1000000], self.df_meas_long['WaterLevel'][:1000000],'Water Level','Timestamp','Measured water level wo outliers and spikes in 1 min timestamp', 'measured water level')
        #self.helper.zoomable_plot_df(self.df_meas_long['Timestamp'][:1000000], self.df_meas_long['WaterLevel'][:1000000],'Water Level','Timestamp','Measured water level wo outliers and spikes in 1 min timestamp', 'measured water level')
        #self.helper.zoomable_plot_df(self.df_meas_long['Timestamp'][:1000000], self.df_meas_long['WaterLevel'][:1000000],'Water Level','Timestamp','Measured water level wo outliers and spikes in 1 min timestamp', 'measured water level')
        #self.helper.zoomable_plot_df(self.df_meas_long['Timestamp'][:1000000], self.df_meas_long['WaterLevel'][:1000000],'Water Level','Timestamp','Measured water level wo outliers and spikes in 1 min timestamp', 'measured water level')
        #self.helper.zoomable_plot_df(self.df_meas_long['Timestamp'][:1000000], self.df_meas_long['WaterLevel'][:1000000],'Water Level','Timestamp','Measured water level wo outliers and spikes in 1 min timestamp', 'measured water level')
    
    #Improve the cotede script by looking at the next existing neighbours and not just the next nieghbours
    #Now: Neighbours within 2 hours of the center point are allowed
    #This allows to work with different time scales and detect spikes even there are NaN values
    def remove_spikes_cotede_improved(self):
        print(datetime.datetime.now())
        self.get_valid_neighbours(self.df_meas_long[6000000:6500000],'WaterLevel', max_distance=60)
        print(datetime.datetime.now())
        spikes = np.abs(self.df_meas_long['WaterLevel'] - (self.df_meas_long['prev_neighbor'] + self.df_meas_long['next_neighbor']) / 2.0) - np.abs((self.df_meas_long['next_neighbor'] - self.df_meas_long['prev_neighbor']) / 2.0)
        self.helper.plot_df(self.df_meas_long['Timestamp'], spikes,'Detected spike [m]','Timestamp','WL spikes in measured ts')

    def get_valid_neighbours(self, df, column, max_distance=60):

        df['next_neighbor'] = np.nan
        df['prev_neighbor'] = np.nan

        # Loop through the DataFrame
        for i in range(len(df)):
            if not pd.isna(df[column].iloc[i]):
                # Find the index of the next non-NaN value for every row
                # Limit search to the next 60 neighbors or the end of the DataFrame, whichever is smaller
                for j in range(i+1, min(i+max_distance+1, len(df))):
                    if not pd.isna(df[column].iloc[j]):
                        df['next_neighbor'].iloc[i] = df[column].iloc[j]
                        break  # Stop after finding the next valid value
                        
                # Find the previous non-NaN neighbor within the last 60 rows
                max_search_range_prev = max(0, i - max_distance)  # Ensure we don't go out of bounds
                for k in range(i - 1, max_search_range_prev - 1, -1):
                    if not pd.isna(df[column].iloc[k]):
                        df['prev_neighbor'].iloc[i] = df[column].iloc[k]
                        break  # Stop after finding the last valid value

        print('hey')

