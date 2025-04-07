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
        # Parameters for ml (to split into training and testing set)
        self.test_size = 0.85 # Size of the test set in percentage
        self.lag_05tide = None #min max amplitude
        self.lag_1tide = None #min min cycle development
        self.lag_2tide = None # 2 * min min cycle development
        #self.seven_days = None
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
        self.lag_05tide = params['spike_detection']['lag_05tide'] #min max amplitude
        self.lag_1tide = params['spike_detection']['lag_1tide'] #min min cycle development
        self.lag_2tide = params['spike_detection']['lag_2tide'] # 2 * min min cycle development
        self.seven_days = params['spike_detection']['seven_days']

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

            """
            This code balances out spike (not working well!) based on neighbors. In the calculated offset by Cotede, the sign does not fit the sign of the spike.
            Thus, it cannot be easily added or substracted. Code below is an idea ob how to handle it.
            Better to use Cotede only for detection
            #df_meas_long['WaterLevel_shiftedpast'] = df_meas_long[data_col_name].shift(50)
            #df_meas_long['WaterLevel_shiftedfuture'] = df_meas_long[data_col_name].shift(-50)
            #df_meas_long['bound'] = (df_meas_long['WaterLevel_shiftedfuture']+df_meas_long['WaterLevel_shiftedpast'])/2
            #df_meas_long[data_col_name] = np.where(
            #    df_meas_long['bound'] <= df_meas_long[data_col_name],  # Condition 1: bound <= WaterLevel
            #    df_meas_long[data_col_name] - abs(sea_level_spike),         # Action 1: WaterLevel - abs(sea_level_spike)
            #    np.where(
            #        df_meas_long['bound'] > df_meas_long[data_col_name],  # Condition 2: bound > WaterLevel
            #        df_meas_long[data_col_name] + abs(sea_level_spike),         # Action 2: WaterLevel + abs(sea_level_spike)
            #        df_meas_long[data_col_name]                                    # Else: keep original value from 'altered'
            #    )
            #)

            #delete helper columns
            #df_meas_long = df_meas_long.drop(columns=['WaterLevel_shiftedpast', 'WaterLevel_shiftedfuture', 'bound'])
            """

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
            #self.helper.plot_two_df(df_meas_long[time_column], df_meas_long[data_col_name],'Water Level [m]', df_meas_long[quality_column_name], 'Quality flag', 'Timestamp',f'Measured water level wo outliers and spikes in 1 min timestamp incl. flags -{suffix}')
            #self.helper.plot_two_df(df_meas_long[time_column], df_meas_long[data_col_name],'Water Level [m]', df_meas_long[quality_column_name], 'Quality flag', 'Timestamp',f'Measured water level in 1 min timestamp incl. flags -{suffix}')
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
    Improve the cotede script by looking at the next existing neighbours and not just the next nieghbours.
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

        """
        Subsequent lines are needed to remove spike and adapt the measured value (not done at the moment & not working well).
        See method above for more details.
        #df_meas_long['bound'] = (df_meas_long['next_neighbour']+df_meas_long['past_neighbour'])/2
        #df_meas_long[data_column] = np.where(
        #    df_meas_long['bound'] <= df_meas_long[data_column],  # Condition 1: bound <= WaterLevel
        #    df_meas_long[data_column] - abs(sea_level_spikes),         # Action 1: WaterLevel - abs(sea_level_spike)
        #    np.where(
        #        df_meas_long['bound'] > df_meas_long[data_column],  # Condition 2: bound > WaterLevel
        #        df_meas_long[data_column] + abs(sea_level_spikes),         # Action 2: WaterLevel + abs(sea_level_spike)
        #        df_meas_long[data_column]                                    # Else: keep original value from 'altered'
        #    )
        #)

        #del df_meas_long['bound']
        """

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
        #self.helper.plot_two_df(df_meas_long[time_column][self.min_value_plotting:self.max_value_plotting], df_meas_long[data_column][self.min_value_plotting:self.max_value_plotting],'Water Level [m]', df_meas_long[quality_column_name][self.min_value_plotting:self.max_value_plotting], 'Quality flag', 'Timestamp ', f'Measured water level wo outliers and spikes in 1 min timestamp (zoomed to max spike) (improved)-{suffix}')
        self.helper.plot_two_df_same_axis(df_meas_long[time_column][self.min_value_plotting:self.max_value_plotting], df_meas_long[data_column][self.min_value_plotting:self.max_value_plotting],'Water Level [m]', 'Detected Spike', df_meas_long['test'][self.min_value_plotting:self.max_value_plotting], 'Timestamp ', 'Measured Water Level', f'Improved Cotede - Measured water level wo outliers and spikes in 1 min timestamp vs orig (zoomed to max spike) (improved)-{suffix}')
        #self.helper.plot_two_df(df_meas_long[time_column], df_meas_long[data_column],'Water Level [m]', df_meas_long[quality_column_name], 'Quality flag', 'Timestamp ',f'Corrected water level wo outliers and spikes in 1 min timestamp (improved)-{suffix}')
        #self.helper.plot_two_df(df_meas_long[time_column], df_meas_long['test'],'Water Level [m]', df_meas_long[quality_column_name], 'Quality flag', 'Timestamp ',f'Measured water level in 1 min timestamp incl. flags (improved)-{suffix}')
        
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
        
    """
    Using a polynomial fitted series over the measurements to detect spikes diverging from the fitted line (using a threshold approach to detect outliers).
    During preprocessing, the polynomial fitted series has been generated over 7 hour windows of measurements like the spline fitting.
    """    

    def remove_spikes_harmonic(self, data, filled_data_column, data_column, time_column, information, original_length, suffix):
        """
        Input:
        -Main dataframe [df]
        -Column name for polynomial fitted column [str]
        -Column name with measurements to analyse [str]
        -Column name for timestamp [str]
        """
        start_time = datetime.now()

        outlier_mask =  np.full(len(data), False, dtype = bool)
        shift_points = (data['segments'] != data['segments'].shift())

        for i in range(0,len(data['segments'][shift_points]), 1):
            start_index = data['segments'][shift_points].index[i]
            if i == len(data['segments'][shift_points])-1:
                end_index = len(data)
            else:
                end_index = data['segments'][shift_points].index[i+1]
            if data['segments'][start_index] == 0:
                relev_df = data[start_index:end_index]
                residuals = np.abs(relev_df[data_column] - relev_df[filled_data_column])

                # Threshold for anomaly detection
                threshold = np.mean(residuals) + self.nsigma * np.std(residuals)
                outlier_mask[start_index:end_index] = residuals > threshold

        #Mark & remove outliers
        data['test'] = np.where(outlier_mask, data[data_column], np.nan)
        data[f'harmonic_detected_spikes{suffix}'] = outlier_mask
        
        ratio = (data[f'harmonic_detected_spikes{suffix}'].sum()/original_length)*100
        print(f"There are {data[f'harmonic_detected_spikes{suffix}'].sum()} spikes in this timeseries according to harmonic series. This is {ratio}% of the overall dataset.")
        information.append([f"There are {data[f'harmonic_detected_spikes{suffix}'].sum()} spikes in this timeseries according to harmonic series. This is {ratio}% of the overall dataset."])

        #Analyse spike detection
        true_indices = data[f'harmonic_detected_spikes{suffix}'][data[f'harmonic_detected_spikes{suffix}']].index
        
        #More plots
        if true_indices.any():
            max_range = builtins.min(31, len(true_indices))
            for i in range(0,max_range):
                x = random.choice(range(0,len(true_indices)))
                min = builtins.max(0,(true_indices[x]-500))
                max = builtins.min(min + 1000, len(data))
                self.helper.plot_two_df_same_axis(data[time_column][min:max], data['test'][min:max],'Water Level [m]', 'Detected Spike', data[data_column][min:max], 'Timestamp ', 'Measured Water Level', f'Harmonic spike Graph{i}- Corrected spikes -{suffix}')
                self.helper.plot_two_df_same_axis(data[time_column][min:max], data['test'][min:max],'Water Level [m]', 'Detected Spike', data[filled_data_column][min:max], 'Timestamp ', 'Modelled Water Level', f'Harmonic Spike Graph{i}- Spline vs measurement) -{suffix}')

        end_time = datetime.now()
        elapsed_time = end_time - start_time
        information.append([f"This algorithm needed {elapsed_time.total_seconds():.6f} seconds to complete."])

        del data['test']

        return data
    
    """
    Using semisupervised ML for spike detection (just a simple and quick test, but doesn't improve anything).

    Use the polynomial fitted series over measurements as 'good' training data as this doesn't have spikes including some features (f.e. lags). 
    During preprocessing, the polynomial fitted series has been generated over 7 hour windows of measurements like the spline fitting.

    This approach has, in the end, barely any difference to a direct comparison to the polynomial fitted series.
    """    

    def remove_spikes_ml(self, data, filled_data_column, data_column, time_column, information, original_length, suffix):
        """
        Input:
        -Main dataframe [df]
        -Column name for polynomial fitted column [str]
        -Column name with measurements to analyse [str]
        -Column name for timestamp [str]
        """
        start_time = datetime.now()
        
        outlier_mask =  np.full(len(data), False, dtype = bool)
        shift_points = (data['segments'] != data['segments'].shift())

        for i in range(0,len(data['segments'][shift_points]), 1):
            start_index = data['segments'][shift_points].index[i]
            if i == len(data['segments'][shift_points])-1:
                end_index = len(data)
            else:
                end_index = data['segments'][shift_points].index[i+1]
            if data['segments'][start_index] == 0:
                relev_df = data[start_index:end_index].copy()
                if len(relev_df) >= self.lag_2tide:
                    #Add relevant features
                    relev_df.loc[:,'lag_05tide'] = relev_df.loc[:,filled_data_column].shift(self.lag_05tide).bfill()  #min max amplitude
                    relev_df.loc[:,'lag_1tide'] = relev_df.loc[:,filled_data_column].shift(self.lag_1tide).bfill()   #min min cycle development
                    relev_df.loc[:,'lag_2tide'] = relev_df.loc[:,filled_data_column].shift(self.lag_2tide).bfill()   # 2 * min min cycle development

                    #Extract relevant output and input features
                    X = relev_df[[filled_data_column, 'lag_1tide', 'lag_05tide', 'lag_2tide']]

                else:
                    relev_df.loc[:,'lag_05tide'] = relev_df.loc[:,filled_data_column].shift(self.lag_05tide).bfill()  #min max amplitude
                    #Extract relevant output and input features
                    X = relev_df[[filled_data_column, 'lag_05tide']]
                y = relev_df[filled_data_column].values

                # Split into train and test
                X_train = X[:-int((self.test_size*len(relev_df)))]
                y_train = y[:-int((self.test_size*len(relev_df)))]

                # Initialize XGBoost model
                model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)

                print(y_train)
                print(X_train)

                # Final training on full train set
                model.fit(X_train, y_train)

                # Test predictions
                y_pred_test = model.predict(X)
                residuals = np.abs(relev_df[data_column] - y_pred_test)

                # Threshold for anomaly detection
                threshold = np.mean(residuals) + self.nsigma * np.std(residuals)
                outlier_mask[start_index:end_index] = residuals > threshold

        #Mark & remove outliers
        data['test'] = np.where(outlier_mask, data[data_column], np.nan)
        data[f'ml_detected_spikes{suffix}'] = outlier_mask
        
        ratio = (data[f'ml_detected_spikes{suffix}'].sum()/original_length)*100
        print(f"There are {data[f'ml_detected_spikes{suffix}'].sum()} spikes in this timeseries according to the ML analysis. This is {ratio}% of the overall dataset.")
        information.append([f"There are {data[f'ml_detected_spikes{suffix}'].sum()} spikes in this timeseries according to the ML analysis. This is {ratio}% of the overall dataset."])

        #Analyse spike detection
        true_indices = data[f'ml_detected_spikes{suffix}'][data[f'ml_detected_spikes{suffix}']].index
        #More plots
        if true_indices.any():
            max_range = builtins.min(31, len(true_indices))
            for i in range(0,max_range):
                x = random.choice(range(0,len(true_indices)))
                min = builtins.max(0,(true_indices[x]-500))
                max = builtins.min(min + 1000, len(data))
                self.helper.plot_two_df_same_axis(data[time_column][min:max], data['test'][min:max],'Water Level [m]', 'Detected Spike', data[data_column][min:max], 'Timestamp ', 'Measured Water Level', f'ML spike Graph{i}- Corrected spikes -{suffix}')
                self.helper.plot_two_df_same_axis(data[time_column][min:max], data['test'][min:max],'Water Level [m]', 'Detected Spike', data[filled_data_column][min:max], 'Timestamp ', 'Modelled Water Level', f'ML Spike Graph{i}- Fitted vs measurement -{suffix}')
        
        end_time = datetime.now()
        elapsed_time = end_time - start_time
        information.append([f"This algorithm needed {elapsed_time.total_seconds():.6f} seconds to complete."])

        del data['test']

        return data