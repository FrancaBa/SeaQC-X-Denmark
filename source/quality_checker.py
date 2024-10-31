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
import ruptures as rpt
import random

from cotede import datasets, qctests
from scipy.signal import argrelextrema
from ruptures.detection import Pelt

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
        self.df_meas_long['quality_flag'] = np.where(np.isnan(self.df_meas_long['WaterLevel']), self.qf_classes['missing_data'], self.qf_classes['good_data'])

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
        constant_mask = self.df_meas_long['WaterLevel'].rolling(window=self.window_constant_value, min_periods=1).apply(lambda x: (x == x[0]).all(), raw=True)

        # Mask the constant values
        self.df_meas_long['altered'] = np.where(constant_mask, np.nan, self.df_meas_long['WaterLevel'])
        self.df_meas_long['quality_flag'] = np.where(self.df_meas_long['quality_flag'] == self.qf_classes['missing_data'], self.qf_classes['missing_data'],  # Keep it as missing_data
                                        np.where(constant_mask, self.qf_classes['bad_data'], self.qf_classes['good_data']))  # Else set to good or bad based on constant_mask
        self.helper.plot_df(self.df_meas_long['Timestamp'], self.df_meas_long['quality_flag'],'Quality Flags','Timestamp','Applied quality flags')

        if self.qf_classes['bad_data'] in self.df_meas_long['quality_flag'].unique():
            ratio = (len(self.df_meas_long[self.df_meas_long['quality_flag'] == self.qf_classes['bad_data']])/len(self.df_meas_long))*100
            print(f"There are {len(self.df_meas_long[self.df_meas_long['quality_flag'] == self.qf_classes['bad_data']])} constant value in this timeseries. This is {ratio}% of the overall dataset.")

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
        outlier_mask = (self.df_meas_long['WaterLevel'] < lower_bound['WaterLevel']) | (self.df_meas_long['WaterLevel'] > upper_bound['WaterLevel'])
        # Mask the outliers
        self.df_meas_long['altered'] = np.where(np.isnan(outlier_mask), np.nan, self.df_meas_long['altered'])
        #Flag & remove the outlier
        self.df_meas_long.loc[(self.df_meas_long['quality_flag'] == self.qf_classes['good_data']) & (outlier_mask), 'quality_flag'] = self.qf_classes['outliers']
        self.df_meas_long['altered'] = np.where(outlier_mask, np.nan, self.df_meas_long['WaterLevel'])

        #plot results without outliers
        self.helper.plot_df(self.df_meas_long['Timestamp'], self.df_meas_long['altered'],'Water Level','Timestamp','Measured water level wo outliers in 1 min timestamp')
        self.helper.plot_df(self.df_meas_long['Timestamp'][33300:33400], self.df_meas_long['altered'][33300:33400],'Water Level','Timestamp','Measured water level in 1 min timestamp wo outliers(zoom)')
        
        if self.qf_classes['outliers'] in self.df_meas_long['quality_flag'].unique():
            ratio = (len(self.df_meas_long[self.df_meas_long['quality_flag'] == self.qf_classes['outliers']])/len(self.df_meas_long))*100
            print(f"There are {len(self.df_meas_long[self.df_meas_long['quality_flag'] == self.qf_classes['outliers']])} outliers in this timeseries. This is {ratio}% of the overall dataset.")
    
        #Subsequent line makes the code slow, only enable when needed
        #self.helper.zoomable_plot_df(self.df_meas_long['Timestamp'][:1000000], self.df_meas_long['WaterLevel'][:1000000],'Water Level','Timestamp', 'Measured water level wo outliers','measured water level')
        
    def remove_spikes_cotede(self):
        # used 'altered WL' series to detect spike as it is already cleaned (NOT raw measurement series)
        # The spike check is a quite traditional one and is based on the principle of comparing one measurement with the tendency observed from the neighbor values.
        # This is already implemented in CoTeDe as qctests.spike
        # Revelant weaknesses:
        # 1. Cannot detect spikes when NaN in the neighbourhood
        # 2. Cannot detect smaller, continious spiking behaviour

        # If no threshold is required, you can do like this:
        sea_level_spike = qctests.spike(self.df_meas_long["altered"])  
        print("The largest spike observed was: {:.3f}".format(np.nanmax(np.abs(sea_level_spike))))
        print("Value at detected position", sea_level_spike[np.nanargmax(np.abs(sea_level_spike))])

        #Assessment of max spike via plots
        min_value = np.nanargmax(np.abs(sea_level_spike)) - 500
        max_value = np.nanargmax(np.abs(sea_level_spike)) + 1500
        self.helper.plot_df(self.df_meas_long['Timestamp'][min_value:max_value], self.df_meas_long['altered'][min_value:max_value],'Water Level','Timestamp','Measured water level wo outliers in 1 min timestamp (zoomed to max spike)')
        self.helper.plot_df(self.df_meas_long['Timestamp'], sea_level_spike,'Detected spike [m]','Timestamp','WL spikes in measured ts')

        # If the spike is less than 1 cm (absolut value), it is not a spike -> Do not remove this small noise
        sea_level_spike[abs(sea_level_spike) < 0.01] = 0.0
        #Plot together
        self.helper.plot_two_df(self.df_meas_long['Timestamp'][min_value:max_value], sea_level_spike[min_value:max_value],'Detected spikes', self.df_meas_long['WaterLevel'][min_value:max_value], 'Water Level', 'Timestamp','Spikes and measured WL')

        #balance out spike
        self.df_meas_long['WaterLevel_shiftedpast'] = self.df_meas_long['altered'].shift(50)
        self.df_meas_long['WaterLevel_shiftedfuture'] = self.df_meas_long['altered'].shift(-50)
        self.df_meas_long['bound'] = (self.df_meas_long['WaterLevel_shiftedfuture']+self.df_meas_long['WaterLevel_shiftedpast'])/2
        self.df_meas_long['altered_2'] = np.where(
            self.df_meas_long['bound'] <= self.df_meas_long['altered'],  # Condition 1: bound <= WaterLevel
            self.df_meas_long['altered'] - abs(sea_level_spike),         # Action 1: WaterLevel - abs(sea_level_spike)
            np.where(
                self.df_meas_long['bound'] > self.df_meas_long['altered'],  # Condition 2: bound > WaterLevel
                self.df_meas_long['altered'] + abs(sea_level_spike),         # Action 2: WaterLevel + abs(sea_level_spike)
                self.df_meas_long['altered']                                    # Else: keep original value from 'altered'
            )
        )

        #Mask spikes
        sea_level_spike_bool = (~np.isnan(sea_level_spike)) & (sea_level_spike != 0)
        self.df_meas_long.loc[(self.df_meas_long['quality_flag'] == self.qf_classes['good_data']) & (sea_level_spike_bool), 'quality_flag'] = self.qf_classes['bad_data_correctable']
        if self.qf_classes['bad_data_correctable'] in self.df_meas_long['quality_flag'].unique():
            ratio = (len(self.df_meas_long[self.df_meas_long['quality_flag'] == self.qf_classes['bad_data_correctable']])/len(self.df_meas_long))*100
            print(f"There are {len(self.df_meas_long[self.df_meas_long['quality_flag'] == self.qf_classes['bad_data_correctable']])} spikes in this timeseries. This is {ratio}% of the overall dataset.")

        #More plots
        for i in range(1, 41):
            min = random.randint(np.where(~np.isnan(sea_level_spike))[0][0], len(self.df_meas_long)-4000)
            max = min + 3000
            self.helper.plot_two_df_same_axis(self.df_meas_long['Timestamp'][min:max], self.df_meas_long['altered_2'][min:max],'Water Level', 'Water Level (corrected)', self.df_meas_long['WaterLevel'][min:max], 'Timestamp', 'Water Level (measured)',f'Graph-Cotede{i}- Measured water level wo outliers and spikes in 1 min timestamp vs orig (zoomed to max spike) (improved)')


        #delete helper columns
        self.df_meas_long = self.df_meas_long.drop(columns=['WaterLevel_shiftedpast', 'WaterLevel_shiftedfuture', 'bound'])
        
        #Analyse spike detection
        self.helper.plot_df(self.df_meas_long['Timestamp'], self.df_meas_long['altered_2'],'Water Level','Timestamp','Measured water level wo outliers and spikes in 1 min timestamp (all)')
        self.helper.plot_two_df(self.df_meas_long['Timestamp'][min_value:max_value], self.df_meas_long['altered_2'][min_value:max_value],'Water Level', self.df_meas_long['quality_flag'][min_value:max_value], 'Quality flag', 'Timestamp','Measured water level wo outliers and spikes in 1 min timestamp (zoomed to max spike)')
        self.helper.plot_two_df(self.df_meas_long['Timestamp'], self.df_meas_long['altered_2'],'Water Level', self.df_meas_long['quality_flag'], 'Quality flag', 'Timestamp','Measured water level wo outliers and spikes in 1 min timestamp')
        self.helper.plot_two_df(self.df_meas_long['Timestamp'], self.df_meas_long['WaterLevel'],'Water Level', self.df_meas_long['quality_flag'], 'Quality flag', 'Timestamp','Measured water level in 1 min timestamp incl. flags')
        self.helper.plot_two_df_same_axis(self.df_meas_long['Timestamp'][min_value:max_value], self.df_meas_long['altered_2'][min_value:max_value],'Water Level', 'Water Level (corrected)', self.df_meas_long['WaterLevel'][min_value:max_value], 'Timestamp', 'Water Level (measured)', 'Measured water level wo outliers and spikes in 1 min timestamp vs orig (zoomed to max spike)')
        #Some more plots for assessment
        self.max_value_plotting = max_value
        self.min_value_plotting = min_value
        min_value = np.nanargmax(np.abs(self.df_meas_long['altered_2'])) - 500
        max_value = np.nanargmax(np.abs(self.df_meas_long['altered_2'])) + 500
        self.helper.plot_two_df(self.df_meas_long['Timestamp'][min_value:max_value], self.df_meas_long['altered_2'][min_value:max_value],'Water Level', self.df_meas_long['quality_flag'][min_value:max_value], 'Quality flag', 'Timestamp','Measured water level wo outliers and wo spike in 1 min timestamp (zoomed to max spike after removing spike)')
        #is deshifting of spike working?
        self.helper.plot_df(self.df_meas_long['Timestamp'][:1000000], self.df_meas_long['altered_2'][:1000000],'Water Level','Timestamp','Measured water level wo outliers and spikes in 1 min timestamp 1')
        self.helper.plot_df(self.df_meas_long['Timestamp'][1000000:2000000], self.df_meas_long['altered_2'][1000000:2000000],'Water Level','Timestamp','Measured water level wo outliers and spikes in 1 min timestamp2')
        self.helper.plot_df(self.df_meas_long['Timestamp'][2000000:3000000], self.df_meas_long['altered_2'][2000000:3000000],'Water Level','Timestamp','Measured water level wo outliers and spikes in 1 min timestamp3')
        self.helper.plot_df(self.df_meas_long['Timestamp'][3000000:4000000], self.df_meas_long['altered_2'][3000000:4000000],'Water Level','Timestamp','Measured water level wo outliers and spikes in 1 min timestamp4')
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
    # used 'altered WL' series to detect spike as it is already cleaned (NOT raw measurement series)
    def remove_spikes_cotede_improved(self):
        self.get_valid_neighbours(self.df_meas_long, 'altered', 'next_neighbour', True, max_distance=60)
        self.get_valid_neighbours(self.df_meas_long, 'altered', 'past_neighbour', False, max_distance=60)
        sea_level_spikes = np.abs(self.df_meas_long['altered'] - (self.df_meas_long['past_neighbour'] + self.df_meas_long['next_neighbour']) / 2.0) - np.abs((self.df_meas_long['next_neighbour'] - self.df_meas_long['past_neighbour']) / 2.0)
        sea_level_spikes[abs(sea_level_spikes) < 0.02] = 0.0
        self.helper.plot_df(self.df_meas_long['Timestamp'], sea_level_spikes,'Detected spike [m]','Timestamp','WL spikes in measured ts (improved)')

        #Remove spike
        self.df_meas_long['bound'] = (self.df_meas_long['next_neighbour']+self.df_meas_long['past_neighbour'])/2
        self.df_meas_long['altered'] = np.where(
            self.df_meas_long['bound'] <= self.df_meas_long['altered'],  # Condition 1: bound <= WaterLevel
            self.df_meas_long['altered'] - abs(sea_level_spikes),         # Action 1: WaterLevel - abs(sea_level_spike)
            np.where(
                self.df_meas_long['bound'] > self.df_meas_long['altered'],  # Condition 2: bound > WaterLevel
                self.df_meas_long['altered'] + abs(sea_level_spikes),         # Action 2: WaterLevel + abs(sea_level_spike)
                self.df_meas_long['altered']                                    # Else: keep original value from 'altered'
            )
        )

        #Mask spikes
        sea_level_spike_bool = (~np.isnan(sea_level_spikes)) & (sea_level_spikes != 0)
        self.df_meas_long.loc[(self.df_meas_long['quality_flag'] == self.qf_classes['good_data']) & (sea_level_spike_bool), 'quality_flag'] = self.qf_classes['bad_data_correctable']
        if self.qf_classes['bad_data_correctable'] in self.df_meas_long['quality_flag'].unique():
            ratio = (len(self.df_meas_long[self.df_meas_long['quality_flag'] == self.qf_classes['bad_data_correctable']])/len(self.df_meas_long))*100
            print(f"There are {len(self.df_meas_long[self.df_meas_long['quality_flag'] == self.qf_classes['bad_data_correctable']])} spikes in this timeseries. This is {ratio}% of the overall dataset.")

        #Analyse spike detection
        self.helper.plot_df(self.df_meas_long['Timestamp'], self.df_meas_long['altered'],'Water Level','Timestamp','Measured water level wo outliers and spikes in 1 min timestamp (all) (improved)')
        self.helper.plot_two_df(self.df_meas_long['Timestamp'][self.min_value_plotting:self.max_value_plotting], self.df_meas_long['altered'][self.min_value_plotting:self.max_value_plotting],'Water Level', self.df_meas_long['quality_flag'][self.min_value_plotting:self.max_value_plotting], 'Quality flag', 'Timestamp','Measured water level wo outliers and spikes in 1 min timestamp (zoomed to max spike) (improved)')
        self.helper.plot_two_df_same_axis(self.df_meas_long['Timestamp'][self.min_value_plotting:self.max_value_plotting], self.df_meas_long['altered'][self.min_value_plotting:self.max_value_plotting],'Water Level', 'Water Level (corrected)', self.df_meas_long['WaterLevel'][self.min_value_plotting:self.max_value_plotting], 'Timestamp', 'Water Level (measured)', 'Measured water level wo outliers and spikes in 1 min timestamp vs orig (zoomed to max spike) (improved)')
        self.helper.plot_two_df(self.df_meas_long['Timestamp'], self.df_meas_long['altered'],'Water Level', self.df_meas_long['quality_flag'], 'Quality flag', 'Timestamp','Measured water level wo outliers and spikes in 1 min timestamp (improved)')
        self.helper.plot_two_df(self.df_meas_long['Timestamp'], self.df_meas_long['WaterLevel'],'Water Level', self.df_meas_long['quality_flag'], 'Quality flag', 'Timestamp','Measured water level in 1 min timestamp incl. flags (improved)')
        
        #More plots
        for i in range(1, 41):
            min = random.randint(1, len(self.df_meas_long)-4000)
            max = min + 4000
            self.helper.plot_two_df_same_axis(self.df_meas_long['Timestamp'][min:max], self.df_meas_long['altered'][min:max],'Water Level', 'Water Level (corrected)', self.df_meas_long['WaterLevel'][min:max], 'Timestamp', 'Water Level (measured)', f'Graph{i}- Measured water level wo outliers and spikes in 1 min timestamp vs orig (zoomed to max spike) (improved)')
    
    def get_valid_neighbours(self, df, column, column_name, shift_future, max_distance=60):

        #Create a list of shifted columns for 60 min shifted values (depends on max_distance)
        # Create shifted columns using numpy (pandas memory issue)
        if shift_future:
            for i in range(1, max_distance + 1):
                df[f'shift_{i}'] = np.roll(df[column], -i)
                df.iloc[-i:, df.columns.get_loc(f'shift_{i}')] = np.nan
        else:
            for i in range(1, max_distance + 1):
                df[f'shift_{i}'] = np.roll(df[column], i)
                df.loc[:i, f'shift_{i}'] = np.nan

        #combine shifted cells to find next closest neighbour value
        df[column_name] = df['shift_1'].fillna(df['shift_2'])
        del df['shift_1']
        del df['shift_2']
        for i in range(3, max_distance+1):
            df[column_name] = df[column_name].fillna(df[f'shift_{i}'])
            del df[f'shift_{i}']

    def detect_shifts(self):
        # Detect shifts in mean using Pruned Exact Linear Time approach
        algo = Pelt(model="l2").fit(self.df_meas_long['altered'].values)
        result = algo.predict(pen=1000)

        # Detect shifts in variance using Pruned Exact Linear Time approach
        #algo = Pelt(model="l2").fit(self.df_meas_long['altered'])
        #result_2 = algo.predict(pen=50)

        # Plot the results
        rpt.display(self.df_meas_long['altered'], result)  # Display the signal with detected change points
        plt.show()
        #rpt.display(self.df_meas_long['altered'], result_2)  # Display the signal with detected change points
        #plt.show()