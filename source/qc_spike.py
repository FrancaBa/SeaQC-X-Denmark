################################################################################
## Written by frb for GronSL project (2024-2025)                              ##
## To detect single spikes in measured series.                                ##
## Approaches:                                                                ##
## 0. Statistical changepoint analysis                                        ##
## 1. Cotede & improved cotede                                                ##
## 2. Selene (adapted)                                                        ##
## 3. ML deviation (train on poly. fitted series) and compare to measurements ##
## 4. Deviation from polynomial fitted series                                 ##
################################################################################

import os
import pandas as pd
import numpy as np
import numpy.ma as ma
import random
import builtins

from cotede import qctests

from xgboost import XGBRegressor

import source.helper_methods as helper

class SpikeDetector():

    def __init__(self):
        self.helper = helper.HelperMethods()

        #Cotede
        self.cotede_threshold = 0.01
        self.improved_cotede_threshold = 0.05 #0.02
        self.max_window_neighbours = 60 #under the assumption that timestamp is in 1 min

        # Parameters for ml (to split into training and testing set)
        self.test_size = 0.85    # Size of the test set in percentage

        #Selene spline approach needed constants
        self.nsigma = 3

    def set_output_folder(self, folder_path):
        folder_path = os.path.join(folder_path,'spike detection')

        #generate output folder for graphs and other docs
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.helper.set_output_folder(folder_path)

        # Impute missing values
    def filled_ts_tight(self, filled_ts):
        self.filled_measured_ts = filled_ts
    
    def detect_spikes_statistical(self, df, data_column_name, time_column, measurement_column):
        """
        In measurement segments, find all stronger changepoints that fit to a real measurement data points. Make a condition to only mark changepoints
        and not other other shifts (here: too_close_indices)
        
        Input:
        -Main dataframe [df]
        -Name of colum where measurements have been interpolated to fill NaNs [str]
        -Column name for timestamp [str]
        -Column name with measurements to analyse [str]
        """
        
        df['test'] = df[measurement_column].copy()
        df['spike_value_statistical'] = False
        shift_points = (df['segments'] != df['segments'].shift())

        #For Qaqortoq: relevant period to test the code
        #test_df = df[(df[time_column].dt.year == 2014) & (df[time_column].dt.month == 1) & (df[time_column].dt.day == 24)]

        #For measurement segment: polynomial fit (6th degree) to the polynomial interpolated measurement series
        for i in range(0,len(df['segments'][shift_points]), 1):
            start_index = df['segments'][shift_points].index[i]
            if i == len(df['segments'][shift_points])-1:
                end_index = len(df)
            else:
                end_index = df['segments'][shift_points].index[i+1]
            if df['segments'][start_index] == 0:
                relev_df = df[start_index:end_index]
                #Get shifted points based on strong gradient in interpolated series (hard to define a gradient if a lot of NaN values.)
                relev_df['change'] = np.abs(np.diff(relev_df[data_column_name], append=np.nan))
                change_points= relev_df[relev_df['change'] > 0.05].index
                if change_points.any():
                    #per change area only 1 changepoint
                    mask = np.diff(change_points, append=np.inf) > 1
                    filtered_changepoints =  np.array(change_points[mask])

                    # Get indices of non-NaN values in measurement column
                    non_nan_indices = relev_df[~relev_df[measurement_column].isna()].index.to_numpy()

                    chunk_size = 100  # Adjust this based on memory capacity
                    closest_indices = []

                    for i in range(0, len(filtered_changepoints), chunk_size): #This loop is only needed because of memory issues.
                        chunk = filtered_changepoints[i:i + chunk_size]
                        distances = np.abs(chunk[:, None] - non_nan_indices)
                        min_indices = np.argmin(distances, axis=1)
                        closest_indices.extend(non_nan_indices[min_indices])

                    closest_indices = np.unique(closest_indices)

                    #Detect only changepoints and not any other shifts
                    too_close_indices = np.where((np.diff(closest_indices) < 20))[0]
                    remove_indices = set(too_close_indices).union(set(too_close_indices + 1))
                    mask = np.array([i not in remove_indices for i in range(len(closest_indices))])
                    result = closest_indices[mask]

                    if result.any():
                        df.loc[result, 'spike_value_statistical'] = True
                        df.loc[result, 'test'] = np.nan

        true_indices = df['spike_value_statistical'][df['spike_value_statistical']].index
        for i in range(0,41):
            min = builtins.max(0,(random.choice(true_indices))-1000)
            max = builtins.min(min + 2000, len(df))
            self.helper.plot_two_df_same_axis(df[time_column][min:max], df['test'][min:max],'Water Level', 'Water Level (corrected)', df[measurement_column][min:max], 'Timestamp', 'Water Level (measured)',f'Statistical spike Graph-local spike detected via statistics {min}')

        #print details on the statistical spike check
        if df['spike_value_statistical'].any():
            ratio = (df['spike_value_statistical'].sum()/len(df))*100
            print(f"There are {df['spike_value_statistical'].sum()} spikes in periods based on a changepoint detection. This is {ratio}% of the overall dataset.")

        del df['test']

        return df
    
    def remove_spikes_cotede(self, df_meas_long, adapted_meas_col_name, time_column):
        """
        Uses CoTeDe package to find spikes (qctests.spike). It decides based on a threshold in regards to the neighboring values, if the checked value is a spike. 
        Thus, this only works if a neighboring value exists (big limitation for incontinuous timeseries).
        
        Input:
        -Main dataframe [df]
        -Column name with measurements to analyse [str]
        -Column name for timestamp [str]
        """
        #Used dummy column to remove spikes to visual analyse the outcome
        df_meas_long['test'] = df_meas_long[adapted_meas_col_name].copy()

        #Call Cotede package
        sea_level_spike = qctests.spike(df_meas_long[adapted_meas_col_name])  
        print("The largest spike observed was: {:.3f}".format(np.nanmax(np.abs(sea_level_spike))))
        print("Value at detected position", sea_level_spike[np.nanargmax(np.abs(sea_level_spike))])

        #Assessment of max spike via plots
        min_value = np.nanargmax(np.abs(sea_level_spike)) - 500
        max_value = np.nanargmax(np.abs(sea_level_spike)) + 1500
        min_value_2 = np.nanargmax(np.abs(sea_level_spike)) - 50
        max_value_2 = np.nanargmax(np.abs(sea_level_spike)) + 50
        self.helper.plot_df(df_meas_long[time_column][min_value:max_value], df_meas_long[adapted_meas_col_name][min_value:max_value],'Water Level','Timestamp','Cotede Graph: Measured water level wo outliers in 1 min timestamp (zoomed to max spike)')
        self.helper.plot_df(df_meas_long[time_column][min_value_2:max_value_2], df_meas_long[adapted_meas_col_name][min_value_2:max_value_2],'Water Level','Timestamp','Cotede Graph: Measured water level wo outliers in 1 min timestamp (very zoomed to max spike)')
        self.helper.plot_df(df_meas_long[time_column], sea_level_spike,'Detected spike [m]','Timestamp','Cotede Graph: WL spikes in measured ts')

        # If the spike is less than 1 cm (absolut value), it is not a spike -> Do not remove the small noise
        sea_level_spike[abs(sea_level_spike) < self.cotede_threshold] = 0.0
        #Plot together
        self.helper.plot_two_df(df_meas_long[time_column][min_value:max_value], sea_level_spike[min_value:max_value],'Detected spikes', df_meas_long[adapted_meas_col_name][min_value:max_value], 'Water Level', 'Timestamp','Cotede Graph: Spikes and measured WL')

        """
        This code balances out spike (not working well!) based on neighbors. In the calculated offset by Cotede, the sign does not fit the sign of the spike.
        Thus, it cannot be easily added or substracted. Code below is an idea ob how to handle it.
        Better to use Cotede only for detection
        #df_meas_long['WaterLevel_shiftedpast'] = df_meas_long[adapted_meas_col_name].shift(50)
        #df_meas_long['WaterLevel_shiftedfuture'] = df_meas_long[adapted_meas_col_name].shift(-50)
        #df_meas_long['bound'] = (df_meas_long['WaterLevel_shiftedfuture']+df_meas_long['WaterLevel_shiftedpast'])/2
        #df_meas_long[adapted_meas_col_name] = np.where(
        #    df_meas_long['bound'] <= df_meas_long[adapted_meas_col_name],  # Condition 1: bound <= WaterLevel
        #    df_meas_long[adapted_meas_col_name] - abs(sea_level_spike),         # Action 1: WaterLevel - abs(sea_level_spike)
        #    np.where(
        #        df_meas_long['bound'] > df_meas_long[adapted_meas_col_name],  # Condition 2: bound > WaterLevel
        #        df_meas_long[adapted_meas_col_name] + abs(sea_level_spike),         # Action 2: WaterLevel + abs(sea_level_spike)
        #        df_meas_long[adapted_meas_col_name]                                    # Else: keep original value from 'altered'
        #    )
        #)

        #delete helper columns
        #df_meas_long = df_meas_long.drop(columns=['WaterLevel_shiftedpast', 'WaterLevel_shiftedfuture', 'bound'])
        """

        #Mask spikes
        sea_level_spike_bool = (~np.isnan(sea_level_spike)) & (sea_level_spike != 0)
        df_meas_long['test'] = np.where(sea_level_spike_bool, np.nan, df_meas_long['test'])
        df_meas_long['cotede_spikes'] = sea_level_spike_bool
        if df_meas_long['cotede_spikes'].any():
            ratio = (df_meas_long['cotede_spikes'].sum()/len(df_meas_long))*100
            print(f"There are {df_meas_long['cotede_spikes'].sum()} spikes in this timeseries according to cotede. This is {ratio}% of the overall dataset.")

        #Plots to analyse spikes
        for i in range(1, 41):
            min = random.randint(np.where(~np.isnan(sea_level_spike))[0][0], len(df_meas_long)-1000)
            max = min + 2000
            self.helper.plot_two_df_same_axis(df_meas_long[time_column][min:max], df_meas_long['test'][min:max],'Water Level', 'Water Level (corrected)', df_meas_long[adapted_meas_col_name][min:max], 'Timestamp', 'Water Level (measured)',f'Cotede Graph {i}- Measured water level wo outliers and spikes in 1 min timestamp vs orig')

        #Analyse spike detection
        self.helper.plot_df(df_meas_long[time_column], df_meas_long['test'],'Water Level','Timestamp','Cotede - Measured water level wo outliers and spikes in 1 min timestamp (all)')
        self.helper.plot_two_df(df_meas_long[time_column][min_value:max_value], df_meas_long[adapted_meas_col_name][min_value:max_value],'Water Level', df_meas_long['test'][min_value:max_value], 'Measured WaterLevel', 'Timestamp','Cotede - Measured water level wo outliers and spikes in 1 min timestamp (zoomed to max spike)')
        #self.helper.plot_two_df(df_meas_long[time_column], df_meas_long[adapted_meas_col_name],'Water Level', df_meas_long[quality_column_name], 'Quality flag', 'Timestamp','Measured water level wo outliers and spikes in 1 min timestamp incl. flags')
        #self.helper.plot_two_df(df_meas_long[time_column], df_meas_long[measurement_column],'Water Level', df_meas_long[quality_column_name], 'Quality flag', 'Timestamp','Measured water level in 1 min timestamp incl. flags')
        self.helper.plot_two_df_same_axis(df_meas_long[time_column][min_value:max_value], df_meas_long[adapted_meas_col_name][min_value:max_value],'Water Level', 'Water Level (corrected)', df_meas_long['test'][min_value:max_value], 'Timestamp', 'Water Level (measured)', 'Cotede - Measured water level wo outliers and spikes in 1 min timestamp vs orig (zoomed to max spike)')
        #Some more plots for assessment
        self.max_value_plotting = max_value
        self.min_value_plotting = min_value
        min_value = np.nanargmax(np.abs(df_meas_long[adapted_meas_col_name])) - 500
        max_value = np.nanargmax(np.abs(df_meas_long[adapted_meas_col_name])) + 500
        self.helper.plot_two_df(df_meas_long[time_column][min_value:max_value], df_meas_long[adapted_meas_col_name][min_value:max_value],'Water Level', df_meas_long['test'][min_value:max_value], 'Quality flag', 'Timestamp','Cotede - Measured water level wo outliers and wo spike in 1 min timestamp (zoomed to max spike after removing spike)')

        del df_meas_long['test']

        return df_meas_long
    
    """  
    Improve the cotede script by looking at the next existing neighbours and not just the next nieghbours.
    Now: It looks for closest neighbouring values within 1 hour of the center point and calculates spikes based on a threshold and these values. 
    This allows to work with different time scales and detect spikes even there are NaN values.

    However, it is hard to choose a correct threshold as the time difference between central point and neighboring value is changing which makes different scales possible.
    """  

    def remove_spikes_cotede_improved(self, df_meas_long, adapted_meas_col_name, time_column):
        """
        Input:
        -Main dataframe [df]
        -Column name with measurements to analyse [str]
        -Column name for timestamp [str]
        """
        
        #Used dummy column to remove spikes to visual analyse the outcome
        df_meas_long['test'] = df_meas_long[adapted_meas_col_name].copy()

        #Find closest neighbour to each point (past and ahead)
        self.get_valid_neighbours(df_meas_long, adapted_meas_col_name, 'next_neighbour', True, max_distance=self.max_window_neighbours)
        self.get_valid_neighbours(df_meas_long, adapted_meas_col_name, 'past_neighbour', False, max_distance=self.max_window_neighbours)
        #Calcualte offset in regards to neighbours and ingnore small spikes
        sea_level_spikes = np.abs(df_meas_long[adapted_meas_col_name] - (df_meas_long['past_neighbour'] + df_meas_long['next_neighbour']) / 2.0) - np.abs((df_meas_long['next_neighbour'] - df_meas_long['past_neighbour']) / 2.0)
        sea_level_spikes[abs(sea_level_spikes) < self.improved_cotede_threshold] = 0.0
        self.helper.plot_df(df_meas_long[time_column], sea_level_spikes,'Detected spike [m]','Timestamp ','Improved Cotede Graph: WL spikes in measured ts (improved)')

        """
        Subsequent lines are needed to remove spike and adapt the measured value (not done at the moment & not working well).
        See method above for more details.
        #df_meas_long['bound'] = (df_meas_long['next_neighbour']+df_meas_long['past_neighbour'])/2
        #df_meas_long[adapted_meas_col_name] = np.where(
        #    df_meas_long['bound'] <= df_meas_long[adapted_meas_col_name],  # Condition 1: bound <= WaterLevel
        #    df_meas_long[adapted_meas_col_name] - abs(sea_level_spikes),         # Action 1: WaterLevel - abs(sea_level_spike)
        #    np.where(
        #        df_meas_long['bound'] > df_meas_long[adapted_meas_col_name],  # Condition 2: bound > WaterLevel
        #        df_meas_long[adapted_meas_col_name] + abs(sea_level_spikes),         # Action 2: WaterLevel + abs(sea_level_spike)
        #        df_meas_long[adapted_meas_col_name]                                    # Else: keep original value from 'altered'
        #    )
        #)

        #del df_meas_long['bound']
        """

        #Mask spikes to flags
        sea_level_spike_bool = (~np.isnan(sea_level_spikes)) & (sea_level_spikes != 0)
        df_meas_long['test'] = np.where(sea_level_spike_bool, np.nan, df_meas_long['test'])
        df_meas_long['cotede_improved_spikes'] = sea_level_spike_bool
        if df_meas_long['cotede_improved_spikes'].any():
            ratio = (df_meas_long['cotede_improved_spikes'].sum()/len(df_meas_long))*100
            print(f"There are {df_meas_long['cotede_improved_spikes'].sum()} spikes in this timeseries according to improved cotede. This is {ratio}% of the overall dataset.")

        #Analyse spike detection
        self.helper.plot_df(df_meas_long[time_column], df_meas_long['test'],'Water Level','Timestamp ','Improved Cotede - Measured water level wo outliers and spikes in 1 min timestamp')
        #self.helper.plot_two_df(df_meas_long[time_column][self.min_value_plotting:self.max_value_plotting], df_meas_long[adapted_meas_col_name][self.min_value_plotting:self.max_value_plotting],'Water Level', df_meas_long[quality_column_name][self.min_value_plotting:self.max_value_plotting], 'Quality flag', 'Timestamp ','Measured water level wo outliers and spikes in 1 min timestamp (zoomed to max spike) (improved)')
        self.helper.plot_two_df_same_axis(df_meas_long[time_column][self.min_value_plotting:self.max_value_plotting], df_meas_long[adapted_meas_col_name][self.min_value_plotting:self.max_value_plotting],'Water Level', 'Water Level (corrected)', df_meas_long['test'][self.min_value_plotting:self.max_value_plotting], 'Timestamp ', 'Water Level (measured)', 'Improved Cotede - Measured water level wo outliers and spikes in 1 min timestamp vs orig (zoomed to max spike) (improved)')
        #self.helper.plot_two_df(df_meas_long[time_column], df_meas_long[adapted_meas_col_name],'Water Level', df_meas_long[quality_column_name], 'Quality flag', 'Timestamp ','Corrected water level wo outliers and spikes in 1 min timestamp (improved)')
        #self.helper.plot_two_df(df_meas_long[time_column], df_meas_long['test'],'Water Level', df_meas_long[quality_column_name], 'Quality flag', 'Timestamp ','Measured water level in 1 min timestamp incl. flags (improved)')
        
        true_indices = df_meas_long['cotede_improved_spikes'][df_meas_long['cotede_improved_spikes']].index
        #More plots
        for i in range(1, 41):
            min = builtins.max(0,(random.choice(true_indices))-2000)
            max = min + 2000
            self.helper.plot_two_df_same_axis(df_meas_long[time_column][min:max], df_meas_long['test'][min:max],'Water Level', 'Water Level (corrected)', df_meas_long[adapted_meas_col_name][min:max], 'Timestamp ', 'Water Level (measured)', f'Improved Cotede Graph{i}- Measured water level wo outliers and spikes in 1 min timestamp vs orig')
          
        del df_meas_long['next_neighbour']
        del df_meas_long['past_neighbour']
        del df_meas_long['test']
        
        return df_meas_long
    
    def get_valid_neighbours(self, df, column, column_name, shift_future, max_distance=60):
        """
        'filled_meas_col_name' is a column generated during preprocessing to have splines fitted to measurements. A spline is fitted to a 14-16 hours window.
        Afterwards, the RMSE is calculated between spline and value and spike assessed based on it.

        Input:
        -Main dataframe [df]
        -Column name for column containing the relevant data to shift [str]
        -Name for the new shifted column [str]
        -Defines the direction of the shift (True = step ahead, False = step back) [boolean]
        -Maximum timesteps shifted in this direction [int]
        """

        #Create a list of shifted columns for 60 min shifted values (depends on max_distance)
        #Create shifted columns using numpy (pandas memory issue)
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
            
    """
    Using Selene package approach for spike detection - code rewritten, but still spline idea. Parameter values are based on Selene.
    This tool can only detect spikes and not directly correct them.
    """   

    def selene_spike_detection(self, df_meas_long, adapted_meas_col_name, time_column, filled_meas_col_name):
        """
        'filled_meas_col_name' is a column generated during preprocessing to have splines fitted to measurements. A spline is fitted to a 14-16 hours window.
        Afterwards, the RMSE is calculated between spline and value and spike assessed based on it.

        Input:
        -Main dataframe [df]
        -Column name with measurements to analyse [str]
        -Column name for timestamp [str]
        -Column name for spline fitted column [str]
        """
        
        outlier_mask =  np.full(len(df_meas_long), False, dtype = bool)
        shift_points = (df_meas_long['segments'] != df_meas_long['segments'].shift())
        df_meas_long['test'] = df_meas_long[adapted_meas_col_name].copy()

        #For Qaqortoq: relevant period to test the code
        #test_df = df[(df[time_column].dt.year == 2014) & (df[time_column].dt.month == 1) & (df[time_column].dt.day == 24)]

        for i in range(0,len(df_meas_long['segments'][shift_points]), 1):
            start_index = df_meas_long['segments'][shift_points].index[i]
            if i == len(df_meas_long['segments'][shift_points])-1:
                end_index = len(df_meas_long)
            else:
                end_index = df_meas_long['segments'][shift_points].index[i+1]
            if df_meas_long['segments'][start_index] == 0:
                relev_df = df_meas_long[start_index:end_index]

                #calculate RMSE between measurements and spline to detect and assess outliers
                sea_level_rmse = self.rmse(relev_df[filled_meas_col_name], relev_df[adapted_meas_col_name])
                outlier_mask[start_index:end_index] = abs(relev_df[filled_meas_col_name]-relev_df[adapted_meas_col_name]) >= self.nsigma*sea_level_rmse
                

        #Mark & remove outliers
        df_meas_long['test'] = np.where(outlier_mask, np.nan, df_meas_long['test'])
        df_meas_long['selene_improved_spikes'] = outlier_mask
        if df_meas_long['selene_improved_spikes'].any():
            ratio = (df_meas_long['selene_improved_spikes'].sum()/len(df_meas_long))*100
            print(f"There are {df_meas_long['selene_improved_spikes'].sum()} spikes in this timeseries according to Selene. This is {ratio}% of the overall dataset.")

        #Analyse spike detection
        self.helper.plot_df(df_meas_long[time_column], df_meas_long['test'],'Water Level','Timestamp ','SELENE: Measured water level wo outliers and spikes in 1 min timestamp (all) (improved)')
        true_indices = df_meas_long['selene_improved_spikes'][df_meas_long['selene_improved_spikes']].index
        #More plots
        for i in range(1, 41):
            min = builtins.max(0,(random.choice(true_indices))-2000)
            max = min + 2000
            self.helper.plot_two_df_same_axis(df_meas_long[time_column][min:max], df_meas_long['test'][min:max],'Water Level', 'Water Level (corrected)', df_meas_long[adapted_meas_col_name][min:max], 'Timestamp ', 'Water Level (measured)', f'SELENE Graph{i}- Corredted spikesS')
            self.helper.plot_two_df_same_axis(df_meas_long[time_column][min:max], df_meas_long['test'][min:max],'Water Level', 'Water Level (corrected)', df_meas_long[filled_meas_col_name][min:max], 'Timestamp ', 'Modelled Water Level', f'SELENE Graph{i}- Spline vs measurement)')
        
        del df_meas_long['test']

        return df_meas_long
    
    def rmse(self, predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean()) 
        
    """
    Using a polynomial fitted series over the measurements to detect spikes diverging from the fitted line (using a threshold approach to detect outliers).
    During preprocessing, the polynomial fitted series has been generated over 7 hour windows of measurements like the spline fitting.
    """    

    def remove_spikes_harmonic(self, data, filled_data_column, adapted_meas_col_name, time_column):
        """
        Input:
        -Main dataframe [df]
        -Column name for polynomial fitted column [str]
        -Column name with measurements to analyse [str]
        -Column name for timestamp [str]
        """

        outlier_mask =  np.full(len(data), False, dtype = bool)
        shift_points = (data['segments'] != data['segments'].shift())
        data['test'] = data[adapted_meas_col_name].copy()

        for i in range(0,len(data['segments'][shift_points]), 1):
            start_index = data['segments'][shift_points].index[i]
            if i == len(data['segments'][shift_points])-1:
                end_index = len(data)
            else:
                end_index = data['segments'][shift_points].index[i+1]
            if data['segments'][start_index] == 0:
                relev_df = data[start_index:end_index]
                residuals = np.abs(relev_df[adapted_meas_col_name] - relev_df[filled_data_column])

                # Threshold for anomaly detection
                threshold = np.mean(residuals) + self.nsigma * np.std(residuals)
                outlier_mask[start_index:end_index] = residuals > threshold

        #Mark & remove outliers
        data['test'] = np.where(outlier_mask, np.nan, data['test'])
        data['ml_detected_spikes'] = outlier_mask
        if data['ml_detected_spikes'].any():
            ratio = (data['ml_detected_spikes'].sum()/len(data))*100
            print(f"There are {data['ml_detected_spikes'].sum()} spikes in this timeseries according to harmonic series. This is {ratio}% of the overall dataset.")

        #Analyse spike detection
        true_indices = data['ml_detected_spikes'][data['ml_detected_spikes']].index
        #More plots
        for i in range(1, 41):
            min = builtins.max(0,(random.choice(true_indices))-2000)
            max = min + 2000
            self.helper.plot_two_df_same_axis(data[time_column][min:max], data['test'][min:max],'Water Level', 'Water Level (corrected)', data[adapted_meas_col_name][min:max], 'Timestamp ', 'Water Level (measured)', f'Harmonic spike Graph{i}- Corrected spikes')
            self.helper.plot_two_df_same_axis(data[time_column][min:max], data['test'][min:max],'Water Level', 'Water Level (corrected)', data[filled_data_column][min:max], 'Timestamp ', 'Modelled Water Level', f'Harmonic Spike Graph{i}- Spline vs measurement)')
            
        del data['test']

        return data
    
    """
    Using semisupervised ML for spike detection (just a simple and quick test, but doesn't improve anything).

    Use the polynomial fitted series over measurements as 'good' training data as this doesn't have spikes including some features (f.e. lags). 
    During preprocessing, the polynomial fitted series has been generated over 7 hour windows of measurements like the spline fitting.

    This approach has, in the end, barely any difference to a direct comparison to the polynomial fitted series.
    """    

    def remove_spikes_ml(self, data, filled_data_column, adapted_meas_col_name, time_column):
        """
        Input:
        -Main dataframe [df]
        -Column name for polynomial fitted column [str]
        -Column name with measurements to analyse [str]
        -Column name for timestamp [str]
        """

        outlier_mask =  np.full(len(data), False, dtype = bool)
        shift_points = (data['segments'] != data['segments'].shift())
        data['test'] = data[adapted_meas_col_name].copy()

        for i in range(0,len(data['segments'][shift_points]), 1):
            start_index = data['segments'][shift_points].index[i]
            if i == len(data['segments'][shift_points])-1:
                end_index = len(data)
            else:
                end_index = data['segments'][shift_points].index[i+1]
            if data['segments'][start_index] == 0:
                relev_df = data[start_index:end_index]

                #Add relevant features
                relev_df['lag_05tide'] = relev_df[filled_data_column].shift(372)  #min max amplitude
                relev_df['lag_1tide'] = relev_df[filled_data_column].shift(745)  #min min cycle development
                relev_df['lag_2tide'] = relev_df[filled_data_column].shift(1490)  # 2 * min min cycle development
                relev_df['rolling_mean'] = relev_df[filled_data_column].rolling(window=10430).mean() # 7-day mean
                relev_df['rolling_std'] = relev_df[filled_data_column].rolling(window=10430).std() # 7-day std

                #Extract relevant output and input features
                X = relev_df[[filled_data_column, 'lag_1tide', 'lag_05tide', 'lag_2tide', 'rolling_mean', 'rolling_std']]
                y = relev_df[filled_data_column]

                # Split into train and test
                X_train = X[:-int((self.test_size*len(relev_df)))]
                y_train = y[:-int((self.test_size*len(relev_df)))]

                # Initialize XGBoost model
                model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
                # Final training on full train set
                model.fit(X_train, y_train)

                # Test predictions
                y_pred_test = model.predict(X)
                residuals = np.abs(relev_df[adapted_meas_col_name] - y_pred_test)

                # Threshold for anomaly detection
                threshold = np.mean(residuals) + self.nsigma * np.std(residuals)
                outlier_mask[start_index:end_index] = residuals > threshold

        #Mark & remove outliers
        data['test'] = np.where(outlier_mask, np.nan, data['test'])
        data['ml_detected_spikes'] = outlier_mask
        if data['ml_detected_spikes'].any():
            ratio = (data['ml_detected_spikes'].sum()/len(data))*100
            print(f"There are {data['ml_detected_spikes'].sum()} spikes in this timeseries according to the ML analysis. This is {ratio}% of the overall dataset.")

        #Analyse spike detection
        true_indices = data['ml_detected_spikes'][data['ml_detected_spikes']].index
        #More plots
        for i in range(1, 41):
            min = builtins.max(0,(random.choice(true_indices))-2000)
            max = min + 2000
            self.helper.plot_two_df_same_axis(data[time_column][min:max], data['test'][min:max],'Water Level', 'Water Level (corrected)', data[adapted_meas_col_name][min:max], 'Timestamp ', 'Water Level (measured)', f'ML spike Graph{i}- Corrected spikes')
            self.helper.plot_two_df_same_axis(data[time_column][min:max], data['test'][min:max],'Water Level', 'Water Level (corrected)', data[filled_data_column][min:max], 'Timestamp ', 'Modelled Water Level', f'ML Spike Graph{i}- Spline vs measurement)')
        
        del data['test']

        return data