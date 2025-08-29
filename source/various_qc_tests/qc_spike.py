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
import numpy as np
import random
import builtins

from cotede import qctests
from datetime import datetime
from xgboost import XGBRegressor

import source.helper_methods as helper

class SpikeDetector():

    def __init__(self):
        self.helper = helper.HelperMethods()

        #Statistical spike detection
        self.change_threshold = None
        #Cotede
        self.cotede_threshold = None
        self.improved_cotede_threshold = None
        self.max_window_neighbours = None #under the assumption that timestamp is in 1 min
        #set bounds as back-up for plotting (overwritten later)
        self.min_value_plotting = 0
        self.max_value_plotting = 10000
        #Selene spline approach needed constants
        self.nsigma = 3
        self.splinelength = None #min
        self.splinedegree = 2 #typically 2 or 3

    def set_output_folder(self, folder_path):
        folder_path = os.path.join(folder_path,'spike detection')

        #generate output folder for graphs and other docs
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.helper.set_output_folder(folder_path)

    #Load relevant parameters for this QC test from conig.json
    def set_parameters(self, params):

        #Cotede
        self.cotede_threshold = params['spike_detection']['cotede_threshold']
        self.improved_cotede_threshold = params['spike_detection']['improved_cotede_threshold']
        self.max_window_neighbours = params['spike_detection']['max_window_neighbours'] #under the assumption that timestamp is in 1 min

        # Parameters for ml (to split into training and testing set)
        self.test_size = params['spike_detection']['test_size'] # Size of the test set in percentage

        #Selene spline approach needed constants
        self.nsigma = params['spike_detection']['nsigma']
        self.splinelength = params['spike_detection']['splinelength'] #min
        self.splinedegree = params['spike_detection']['splinedegree']
    
    def remove_spikes_cotede(self, df_meas_long, data_col_name, time_column, information, original_length, suffix):
        """
        Uses CoTeDe package to find spikes (qctests.spike). It decides based on a threshold in regards to the neighboring values, if the checked value is a spike. 
        Thus, this only works if a neighboring value exists (big limitation for incontinuous timeseries).
        
        Input:
        -Main dataframe [df]
        -Column name with measurements to analyse [str]
        -Column name for timestamp [str]
        """
        start_time = datetime.now()

        #Used dummy column to remove spikes to visual analyse the outcome
        df_meas_long['test'] = df_meas_long[data_col_name].copy()

        #Call Cotede package
        sea_level_spike = qctests.spike(df_meas_long[data_col_name])  

        # If the spike is less than 1 cm (absolut value), it is not a spike -> Do not remove the small noise
        sea_level_spike[abs(sea_level_spike) < self.cotede_threshold] = np.nan

        #Only execute if spike is detected
        if np.any(~np.isnan(sea_level_spike)):
            print("The largest spike observed was: {:.3f}".format(np.nanmax(np.abs(sea_level_spike))))
            print("Value at detected position", sea_level_spike[np.nanargmax(np.abs(sea_level_spike))])

            #Assessment of max spike via plots
            min_value = np.nanargmax(np.abs(sea_level_spike)) - 500
            max_value = np.nanargmax(np.abs(sea_level_spike)) + 1500
            min_value_2 = np.nanargmax(np.abs(sea_level_spike)) - 50
            max_value_2 = np.nanargmax(np.abs(sea_level_spike)) + 50
            self.helper.plot_df(df_meas_long[time_column][min_value:max_value], df_meas_long[data_col_name][min_value:max_value],'Water Level [m]','Timestamp',f'Cotede Graph: Measured water level wo outliers in 1 min timestamp (zoomed to max spike) -{suffix}')
            self.helper.plot_df(df_meas_long[time_column][min_value_2:max_value_2], df_meas_long[data_col_name][min_value_2:max_value_2],'Water Level [m]','Timestamp',f'Cotede Graph: Measured water level wo outliers in 1 min timestamp (very zoomed to max spike) -{suffix}')
            self.helper.plot_df(df_meas_long[time_column], sea_level_spike,'Detected spike [m]','Timestamp',f'Cotede Graph: WL spikes in measured ts -{suffix}')

            #Plot together
            self.helper.plot_two_df(df_meas_long[time_column][min_value:max_value], sea_level_spike[min_value:max_value],'Detected spikes', df_meas_long[data_col_name][min_value:max_value], 'Water Level [m]', 'Timestamp',f'Cotede Graph: Spikes and measured WL -{suffix}')

            #Mask spikes
            sea_level_spike_bool = (~np.isnan(sea_level_spike)) & (sea_level_spike != 0)
            df_meas_long['test'] = np.where(sea_level_spike_bool, df_meas_long['test'], np.nan)
            df_meas_long[f'cotede_spikes{suffix}'] = sea_level_spike_bool
            
            ratio = (df_meas_long[f'cotede_spikes{suffix}'].sum()/original_length)*100
            print(f"There are {df_meas_long[f'cotede_spikes{suffix}'].sum()} spikes in this timeseries according to cotede. This is {ratio}% of the overall dataset.")
            information.append([f"There are {df_meas_long[f'cotede_spikes{suffix}'].sum()} spikes in this timeseries according to cotede. This is {ratio}% of the overall dataset."])

            #Plots to analyse spikes
            true_indices = df_meas_long[f'cotede_spikes{suffix}'][df_meas_long[f'cotede_spikes{suffix}']].index
            max_range = builtins.min(31, len(true_indices))
            for i in range(0,max_range):
                x = random.choice(range(0,len(true_indices)))
                min = builtins.max(0,(true_indices[x]-500))
                max = builtins.min(min + 1000, len(df_meas_long))
                self.helper.plot_two_df_same_axis(df_meas_long[time_column][min:max], df_meas_long['test'][min:max],'Water Level [m]', 'Detected Spike', df_meas_long[data_col_name][min:max], 'Timestamp', 'Measured Water Level',f'Cotede Graph {i}- Measured water level wo outliers and spikes in 1 min timestamp vs orig -{suffix}')

            #Analyse spike detection
            self.helper.plot_df(df_meas_long[time_column], df_meas_long['test'],'Water Level [m]','Timestamp','Cotede - Measured water level wo outliers and spikes in 1 min timestamp (all)')
            self.helper.plot_two_df(df_meas_long[time_column][min_value:max_value], df_meas_long['test'][min_value:max_value],'Water Level [m]', df_meas_long[data_col_name][min_value:max_value], 'Measured WaterLevel', 'Timestamp',f'Cotede - Measured water level wo outliers and spikes in 1 min timestamp (zoomed to max spike) -{suffix}')
            self.helper.plot_two_df_same_axis(df_meas_long[time_column][min_value:max_value], df_meas_long['test'][min_value:max_value],'Water Level [m]', 'Detected Spike', df_meas_long[data_col_name][min_value:max_value], 'Timestamp', 'Measured Water Level',f'Cotede - Measured water level wo outliers and spikes in 1 min timestamp vs orig (zoomed to max spike) -{suffix}')
            #Some more plots for assessment
            self.max_value_plotting = max_value
            self.min_value_plotting = min_value
            min_value = np.nanargmax(np.abs(df_meas_long[data_col_name])) - 500
            max_value = np.nanargmax(np.abs(df_meas_long[data_col_name])) + 500
            self.helper.plot_two_df_same_axis(df_meas_long[time_column][min_value:max_value], df_meas_long[data_col_name][min_value:max_value],'Water Level [m]', 'Detected Spike', df_meas_long['test'][min_value:max_value], 'Timestamp','Measured Water Level', f'Cotede - Zoomed{suffix}')
            self.helper.plot_two_df(df_meas_long[time_column][min_value:max_value], df_meas_long[data_col_name][min_value:max_value],'Water Level [m]', df_meas_long['test'][min_value:max_value], 'Water Level (clean)', 'Timestamp',f'Cotede - Measured water level wo outliers and wo spike in 1 min timestamp (zoomed to max spike after removing spike) -{suffix}')
        else:
            self.min_value_plotting = 0
            self.max_value_plotting = 10000
            print(f"There are 0 spikes in this timeseries according to cotede. This is 0% of the overall dataset.")
            information.append([f"There are 0 spikes in this timeseries according to cotede. This is 0% of the overall dataset."])

        end_time = datetime.now()
        elapsed_time = end_time - start_time
        information.append([f"This algorithm needed {elapsed_time.total_seconds():.6f} seconds to complete."])

        del df_meas_long['test']

        return df_meas_long
    
    """  
    Improve the cotede script by looking at the next existing neighbours and not just the next neighbours.
    Now: It looks for closest neighbouring values within 1 hour of the center point and calculates spikes based on a threshold and these values. 
    This allows to work with different time scales and detect spikes even there are NaN values.

    However, it is hard to choose a correct threshold as the time difference between central point and neighboring value is changing which makes different scales possible.
    """  

    def remove_spikes_cotede_improved(self, df_meas_long, data_column, time_column, information, original_length, suffix):
        """
        Input:
        -Main dataframe [df]
        -Column name with measurements to analyse [str]
        -Column name for timestamp [str]
        """
        start_time = datetime.now()

        #Used dummy column to remove spikes to visual analyse the outcome
        df_meas_long['test'] = df_meas_long[data_column].copy()

        #Find closest neighbour to each point (past and ahead)
        self.get_valid_neighbours(df_meas_long, data_column, 'next_neighbour', True, max_distance=self.max_window_neighbours)
        self.get_valid_neighbours(df_meas_long, data_column, 'past_neighbour', False, max_distance=self.max_window_neighbours)
        #Calcualte offset in regards to neighbours and ingnore small spikes
        sea_level_spikes = np.abs(df_meas_long[data_column] - (df_meas_long['past_neighbour'] + df_meas_long['next_neighbour']) / 2.0) - np.abs((df_meas_long['next_neighbour'] - df_meas_long['past_neighbour']) / 2.0)
        sea_level_spikes[abs(sea_level_spikes) < self.improved_cotede_threshold] = 0.0
        self.helper.plot_df(df_meas_long[time_column], sea_level_spikes,'Detected spike [m]','Timestamp ',f'Improved Cotede Graph: WL spikes in measured ts (improved) -{suffix}')

        #Mask spikes to flags
        sea_level_spike_bool = (~np.isnan(sea_level_spikes)) & (sea_level_spikes != 0)
        #df_meas_long['test'] = np.where(sea_level_spike_bool, np.nan, df_meas_long['test'])
        df_meas_long['test'] = np.where(sea_level_spike_bool, df_meas_long['test'], np.nan)
        df_meas_long[f'cotede_improved_spikes{suffix}'] = sea_level_spike_bool
       
        ratio = (df_meas_long[f'cotede_improved_spikes{suffix}'].sum()/original_length)*100
        print(f"There are {df_meas_long[f'cotede_improved_spikes{suffix}'].sum()} spikes in this timeseries according to improved cotede. This is {ratio}% of the overall dataset.")
        information.append([f"There are {df_meas_long[f'cotede_improved_spikes{suffix}'].sum()} spikes in this timeseries according to improved cotede. This is {ratio}% of the overall dataset."])

        #Analyse spike detection
        self.helper.plot_df(df_meas_long[time_column], df_meas_long['test'],'Water Level [m]','Timestamp ', f'Improved Cotede - Measured water level wo outliers and spikes in 1 min timestamp-{suffix}')
        self.helper.plot_two_df_same_axis(df_meas_long[time_column][self.min_value_plotting:self.max_value_plotting], df_meas_long[data_column][self.min_value_plotting:self.max_value_plotting],'Water Level [m]', 'Detected Spike', df_meas_long['test'][self.min_value_plotting:self.max_value_plotting], 'Timestamp ', 'Measured Water Level', f'Improved Cotede - Measured water level wo outliers and spikes in 1 min timestamp vs orig (zoomed to max spike) (improved)-{suffix}')

        true_indices = df_meas_long[f'cotede_improved_spikes{suffix}'][df_meas_long[f'cotede_improved_spikes{suffix}']].index
        #More plots
        max_range = builtins.min(31, len(true_indices))
        for i in range(0,max_range):
            x = random.choice(range(0,len(true_indices)))
            min = builtins.max(0,(true_indices[x]-500))
            max = builtins.min(min + 1000, len(df_meas_long))
            self.helper.plot_two_df_same_axis(df_meas_long[time_column][min:max], df_meas_long['test'][min:max],'Water Level [m]', 'Detected Spikes', df_meas_long[data_column][min:max], 'Timestamp ', 'Measured Water Level', f'Improved Cotede Graph{i}- Measured water level wo outliers and spikes in 1 min timestamp vs orig-{suffix}')

        end_time = datetime.now()
        elapsed_time = end_time - start_time
        information.append([f"This algorithm needed {elapsed_time.total_seconds():.6f} seconds to complete."])

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
    def selene_spike(self, data, data_column, time_column, segment_column, filled_data, information, original_length, suffix):
        """
        In measurement segments, fit a polynomial curve over existing measurements. This genereates a constinuous curve without sudden jumps or missing values. A continuous ts is needed for some QCs later on.
        
        Input:
        -Main dataframe [df]
        -Column name with measurements to analyse [str]
        -Column name for timestamp [str]
        -Column name for segments [str]
        -Column name for polynomial interpolated series (in pervious method) [str]
        """
        start_time = datetime.now()

        #Used dummy column to remove spikes to visual analyse the outcome
        data['test'] = np.nan

        #Get start and end point of each segment
        shift_points = (data[segment_column] != data[segment_column].shift())
        #Assuming a 1-min time step with spline length defined in hours
        winsize = self.splinelength
        data[f'selene_spikes{suffix}'] = False

        #For measurement segment: polynomial fit (6th degree) to the polynomial interpolated measurement series
        for i in range(0,len(data[segment_column][shift_points]), 1):  
            start_index = data[segment_column][shift_points].index[i]
            if i == len(data[segment_column][shift_points])-1:
                end_index = len(data)
            else:
                end_index = data[segment_column][shift_points].index[i+1]
            if data[segment_column][start_index] == 0:
                #Fit a serie to each segment of x hours (= self.splinelength). Iterate the following steps:
                #1. Get start and end index
                #2. Extract connected x- and y-series
                #3. Fit polynomial series to x- and y-series pair
                #4. Overwrite beginning and end of segment with next segment for smoother transition
                for start in range(start_index, end_index):
                    if (start < ((winsize/2)+start_index)):
                        ini=0+start_index
                        end=start_index + winsize-1
                        winix = start-start_index
                    elif (start > len(data.index) - winsize/2):
                        ini=len(data.index)-winsize
                        end=len(data.index)-1
                        winix = (winsize-1)-(end-start+1)
                    else:
                        ini = int(start - winsize/2)
                        end = int(start + winsize/2)
                        winix = int(winsize/2)
                    x_window = data.loc[ini:end,time_column]
                    x_numeric = (x_window - x_window.min()).dt.total_seconds() / 60  # Convert to minutes
                    y_window = data.loc[ini:end, filled_data].ffill()

                    coefficients = np.polyfit(x_numeric, y_window, self.splinedegree)

                    # Create a polynomial series based on the coefficients and timeseries
                    y_fit = np.polyval(coefficients, x_numeric)
                    rmse = self.rmse(y_fit, data.loc[ini:end, data_column])
                    if (abs(y_fit[winix]-data[data_column][start]) >= self.nsigma*rmse):
                        data.loc[start, f'selene_spikes{suffix}'] = True
                        data.loc[start, 'test'] = data.loc[start, data_column]

        #Plot for visual analysis
        ratio = (data[f'selene_spikes{suffix}'].sum()/original_length)*100
        print(f"There are {data[f'selene_spikes{suffix}'].sum()} spikes in this timeseries according to original SELENE. This is {ratio}% of the overall dataset.")
        information.append([f"There are {data[f'selene_spikes{suffix}'].sum()} spikes in this timeseries according to original SELENE. This is {ratio}% of the overall dataset."])

        true_indices = data[f'selene_spikes{suffix}'][data[f'selene_spikes{suffix}']].index
        #More plots
        max_range = builtins.min(31, len(true_indices))
        for i in range(0,max_range):
            x = random.choice(range(0,len(true_indices)))
            min = builtins.max(0,(true_indices[x]-500))
            max = builtins.min(min + 1000, len(data))
            self.helper.plot_two_df_same_axis(data[time_column][min:max], data['test'][min:max],'Water Level [m]', 'Detected Spike', data[data_column][min:max], 'Timestamp ', 'Measured Water Level', f'SELENE{i}- Measured water level wo outliers and spikes in 1 min timestamp vs orig -{suffix}')
        
        end_time = datetime.now()
        elapsed_time = end_time - start_time
        information.append([f"This algorithm needed {elapsed_time.total_seconds():.6f} seconds to complete."])

        del data['test']

        return data
    
    def selene_spike_detection(self, df_meas_long, data_column, time_column, filled_meas_col_name, information, original_length, suffix):
        """
        'filled_meas_col_name' is a column generated during preprocessing to have splines fitted to measurements. A spline is fitted to a 7 hour window.
        Afterwards, the RMSE is calculated between spline and value and spike assessed based on it.

        Input:
        -Main dataframe [df]
        -Column name with measurements to analyse [str]
        -Column name for timestamp [str]
        -Column name for spline fitted column [str]
        """
        start_time = datetime.now()

        outlier_mask =  np.full(len(df_meas_long), False, dtype = bool)
        shift_points = (df_meas_long['segments'] != df_meas_long['segments'].shift())

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
                sea_level_rmse = self.rmse(relev_df[filled_meas_col_name], relev_df[data_column])
                outlier_mask[start_index:end_index] = abs(relev_df[filled_meas_col_name]-relev_df[data_column]) >= self.nsigma*sea_level_rmse

        #Mark & remove outliers
        df_meas_long['test'] = np.where(outlier_mask, df_meas_long[data_column], np.nan)
        df_meas_long[f'selene_improved_spikes{suffix}'] = outlier_mask
        
        ratio = (outlier_mask.sum()/original_length)*100
        print(f"There are {outlier_mask.sum()} spikes in this timeseries according to improved Selene. This is {ratio}% of the overall dataset.")
        information.append([f"There are {outlier_mask.sum()} spikes in this timeseries according to improved Selene. This is {ratio}% of the overall dataset."])

        #Analyse spike detection
        self.helper.plot_df(df_meas_long[time_column], df_meas_long['test'],'Water Level [m]','Timestamp ','Improved SELENE: Measured water level wo outliers and spikes in 1 min timestamp (all) (improved)')
        true_indices = df_meas_long[f'selene_improved_spikes{suffix}'][df_meas_long[f'selene_improved_spikes{suffix}']].index
        #More plots
        max_range = builtins.min(31, len(true_indices))
        for i in range(0,max_range):
            x = random.choice(range(0,len(true_indices)))
            min = builtins.max(0,(true_indices[x]-500))
            max = builtins.min(min + 1000, len(df_meas_long))
            self.helper.plot_two_df_same_axis(df_meas_long[time_column][min:max], df_meas_long['test'][min:max],'Water Level [m]', 'Detected Spike', df_meas_long[data_column][min:max], 'Timestamp ', 'Measured Water Level', f'Improved SELENE Graph{i}- Corrected spikes -{suffix}')
            self.helper.plot_two_df_same_axis(df_meas_long[time_column][min:max], df_meas_long['test'][min:max],'Water Level [m]', 'Detected Spike', df_meas_long[filled_meas_col_name][min:max], 'Timestamp ', 'Modelled Water Level', f'Improved SELENE Graph{i}- Spline vs measurement)-{suffix}')
        
        end_time = datetime.now()
        elapsed_time = end_time - start_time
        information.append([f"This algorithm needed {elapsed_time.total_seconds():.6f} seconds to complete."])

        del df_meas_long['test']

        return df_meas_long
    
    def rmse(self, predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean()) 