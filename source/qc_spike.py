import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
import random
import builtins

from cotede import qctests

from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


import source.helper_methods as helper

class SpikeDetector():

    def __init__(self):
        self.helper = helper.HelperMethods()

        #Cotede
        self.cotede_threshold = 0.01
        self.improved_cotede_threshold = 0.1 #0.02
        self.max_window_neighbours = 60 #under the assumption that timestamp is in 1 min

        # Parameters for ml
        self.test_size = 0.85    # Size of the test set in percentage

        #Spline approach needed constants
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
        
        df['test'] = df[measurement_column].copy()
        df['spike_value_statistical'] = False
        shift_points = (df['segments'] != df['segments'].shift())

        #test_df = df[(df[time_column].dt.year == 2014) & (df[time_column].dt.month == 1) & (df[time_column].dt.day == 24)]

        for i in range(0,len(df['segments'][shift_points]), 1):
            start_index = df['segments'][shift_points].index[i]
            if i == len(df['segments'][shift_points])-1:
                end_index = len(df)
            else:
                end_index = df['segments'][shift_points].index[i+1]
            if df['segments'][start_index] == 0:
                relev_df = df[start_index:end_index]
                #Get shifted points based on strong gradient
                relev_df['change'] = np.abs(np.diff(relev_df[data_column_name], append=np.nan))
                change_points= relev_df[relev_df['change'] > 0.05].index
                if change_points.any():
                    mask = np.diff(change_points, append=np.inf) > 1
                    filtered_changepoints =  np.array(change_points[mask])

                    # Get indices of non-NaN values in measurement column
                    non_nan_indices = relev_df[~relev_df[measurement_column].isna()].index.to_numpy()

                    chunk_size = 100  # Adjust this based on memory capacity
                    closest_indices = []

                    for i in range(0, len(filtered_changepoints), chunk_size):
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
                    print(result)

                    if result.any():
                        df.loc[result, 'spike_value_statistical'] = True
                        df.loc[result, measurement_column] = np.nan
                        #min = builtins.max(0,(random.choice(result))-1000)
                        #max = builtins.min(min + 2000, len(df))
                        #self.helper.plot_two_df_same_axis(df[time_column][min:max], df[measurement_column][min:max],'Water Level', 'Water Level (corrected)', df['test'][min:max], 'Timestamp', 'Water Level (measured)',f'Statistical spike Graph - local spike detected via statistics {min}')

        true_indices = df['spike_value_statistical'][df['spike_value_statistical']].index
        for i in range(0,41):
            min = builtins.max(0,(random.choice(true_indices))-1000)
            max = builtins.min(min + 2000, len(df))
            self.helper.plot_two_df_same_axis(df[time_column][min:max], df[measurement_column][min:max],'Water Level', 'Water Level (corrected)', df['test'][min:max], 'Timestamp', 'Water Level (measured)',f'Statistical spike Graph-local spike detected via statistics {min}')

        #print details on the small distribution check
        if df['spike_value_statistical'].any():
            ratio = (df['spike_value_statistical'].sum()/len(df))*100
            print(f"There are {df['spike_value_statistical'].sum()} spikes in periods based on a changepoint detection. This is {ratio}% of the overall dataset.")

        del df['test']

        return df
    
    def remove_spikes_cotede(self, df_meas_long, adapted_meas_col_name, time_column):

        df_meas_long['test'] = df_meas_long[adapted_meas_col_name].copy()
        # used 'altered WL' series to detect spike as it is already cleaned (NOT raw measurement series)
        # The spike check is a quite traditional one and is based on the principle of comparing one measurement with the tendency observed from the neighbor values.
        # This is already implemented in CoTeDe as qctests.spike
        # Revelant weaknesses:
        # 1. Cannot detect spikes when NaN in the neighbourhood
        # 2. Cannot detect smaller, continious spiking behaviour

        # If no threshold is required, you can do like this:{"IsDistinguishedFolder":true,"FolderId":{"Id":"AAMkADJkMWFjYmQ0LWFjMzQtNDhlMy1iMzdkLWExN2Q4MjJkMDA3ZQAuAAAAAABcbTRROxEgQZ3JJ2pBWrh8AQBb+VROiwnuTqOVc4B9qTLcAAAAAAEMAAA=","ChangeKey":"AQAAABYAAABb+VROiwnuTqOVc4B9qTLcAAA+Lmf7"},"DragItemType":3}
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

        # If the spike is less than 1 cm (absolut value), it is not a spike -> Do not remove this small noise
        sea_level_spike[abs(sea_level_spike) < self.cotede_threshold] = 0.0
        #Plot together
        self.helper.plot_two_df(df_meas_long[time_column][min_value:max_value], sea_level_spike[min_value:max_value],'Detected spikes', df_meas_long[adapted_meas_col_name][min_value:max_value], 'Water Level', 'Timestamp','Cotede Graph: Spikes and measured WL')

        #balance out spike (not working well!!) - Only use it as detection
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

        #Mask spikes
        sea_level_spike_bool = (~np.isnan(sea_level_spike)) & (sea_level_spike != 0)
        df_meas_long[adapted_meas_col_name] = np.where(sea_level_spike_bool, np.nan, df_meas_long[adapted_meas_col_name])
        df_meas_long['cotede_spikes'] = sea_level_spike_bool
        if df_meas_long['cotede_spikes'].any():
            ratio = (df_meas_long['cotede_spikes'].sum()/len(df_meas_long))*100
            print(f"There are {df_meas_long['cotede_spikes'].sum()} spikes in this timeseries according to cotede. This is {ratio}% of the overall dataset.")

        #More plots
        for i in range(1, 41):
            min = random.randint(np.where(~np.isnan(sea_level_spike))[0][0], len(df_meas_long)-1000)
            max = min + 2000
            self.helper.plot_two_df_same_axis(df_meas_long[time_column][min:max], df_meas_long[adapted_meas_col_name][min:max],'Water Level', 'Water Level (corrected)', df_meas_long['test'][min:max], 'Timestamp', 'Water Level (measured)',f'Cotede Graph {i}- Measured water level wo outliers and spikes in 1 min timestamp vs orig')


        #delete helper columns
        #df_meas_long = df_meas_long.drop(columns=['WaterLevel_shiftedpast', 'WaterLevel_shiftedfuture', 'bound'])
        
        #Analyse spike detection
        self.helper.plot_df(df_meas_long[time_column], df_meas_long[adapted_meas_col_name],'Water Level','Timestamp','Cotede - Measured water level wo outliers and spikes in 1 min timestamp (all)')
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
    Improve the cotede script by looking at the next existing neighbours and not just the next nieghbours
    Now: Neighbours within 2 hours of the center point are allowed
    This allows to work with different time scales and detect spikes even there are NaN values
    """  

    def remove_spikes_cotede_improved(self, df_meas_long, adapted_meas_col_name, time_column):

        df_meas_long['test'] = df_meas_long[adapted_meas_col_name].copy()

        self.get_valid_neighbours(df_meas_long, adapted_meas_col_name, 'next_neighbour', True, max_distance=self.max_window_neighbours)
        self.get_valid_neighbours(df_meas_long, adapted_meas_col_name, 'past_neighbour', False, max_distance=self.max_window_neighbours)
        sea_level_spikes = np.abs(df_meas_long[adapted_meas_col_name] - (df_meas_long['past_neighbour'] + df_meas_long['next_neighbour']) / 2.0) - np.abs((df_meas_long['next_neighbour'] - df_meas_long['past_neighbour']) / 2.0)
        sea_level_spikes[abs(sea_level_spikes) < self.improved_cotede_threshold] = 0.0
        self.helper.plot_df(df_meas_long[time_column], sea_level_spikes,'Detected spike [m]','Timestamp ','Improved Cotede Graph: WL spikes in measured ts (improved)')

        #Remove spike
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

        #Mask spikes to flags
        sea_level_spike_bool = (~np.isnan(sea_level_spikes)) & (sea_level_spikes != 0)
        df_meas_long[adapted_meas_col_name] = np.where(sea_level_spike_bool, np.nan, df_meas_long[adapted_meas_col_name])
        df_meas_long['cotede_improved_spikes'] = sea_level_spike_bool
        if df_meas_long['cotede_improved_spikes'].any():
            ratio = (df_meas_long['cotede_improved_spikes'].sum()/len(df_meas_long))*100
            print(f"There are {df_meas_long['cotede_improved_spikes'].sum()} spikes in this timeseries according to improved cotede. This is {ratio}% of the overall dataset.")

        #Analyse spike detection
        self.helper.plot_df(df_meas_long[time_column], df_meas_long[adapted_meas_col_name],'Water Level','Timestamp ','Improved Cotede - Measured water level wo outliers and spikes in 1 min timestamp')
        #self.helper.plot_two_df(df_meas_long[time_column][self.min_value_plotting:self.max_value_plotting], df_meas_long[adapted_meas_col_name][self.min_value_plotting:self.max_value_plotting],'Water Level', df_meas_long[quality_column_name][self.min_value_plotting:self.max_value_plotting], 'Quality flag', 'Timestamp ','Measured water level wo outliers and spikes in 1 min timestamp (zoomed to max spike) (improved)')
        self.helper.plot_two_df_same_axis(df_meas_long[time_column][self.min_value_plotting:self.max_value_plotting], df_meas_long[adapted_meas_col_name][self.min_value_plotting:self.max_value_plotting],'Water Level', 'Water Level (corrected)', df_meas_long['test'][self.min_value_plotting:self.max_value_plotting], 'Timestamp ', 'Water Level (measured)', 'Improved Cotede - Measured water level wo outliers and spikes in 1 min timestamp vs orig (zoomed to max spike) (improved)')
        #self.helper.plot_two_df(df_meas_long[time_column], df_meas_long[adapted_meas_col_name],'Water Level', df_meas_long[quality_column_name], 'Quality flag', 'Timestamp ','Corrected water level wo outliers and spikes in 1 min timestamp (improved)')
        #self.helper.plot_two_df(df_meas_long[time_column], df_meas_long['test'],'Water Level', df_meas_long[quality_column_name], 'Quality flag', 'Timestamp ','Measured water level in 1 min timestamp incl. flags (improved)')
        
        true_indices = df_meas_long['cotede_improved_spikes'][df_meas_long['cotede_improved_spikes']].index
        #More plots
        for i in range(1, 41):
            min = builtins.max(0,(random.choice(true_indices))-2000)
            max = min + 2000
            self.helper.plot_two_df_same_axis(df_meas_long[time_column][min:max], df_meas_long[adapted_meas_col_name][min:max],'Water Level', 'Water Level (corrected)', df_meas_long['test'][min:max], 'Timestamp ', 'Water Level (measured)', f'Improved Cotede Graph{i}- Measured water level wo outliers and spikes in 1 min timestamp vs orig')
            #self.helper.plot_two_df(df_meas_long[time_column][min:max], df_meas_long['test'][min:max],'Water Level', df_meas_long[quality_column_name][min:max], 'Quality flag', 'Timestamp ',f'Graph{i}- Measured water level incl. flags')
        
        del df_meas_long['next_neighbour']
        del df_meas_long['past_neighbour']
        del df_meas_long['test']
        #del df_meas_long['bound']

        return df_meas_long
    
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
            
    """
    Using Selene package for spike detection - code rewritten, but still spline idea (do not improve data, but delete them)
    Parameter values are based on Selene
    """   

    def selene_spike_detection(self, df_meas_long, adapted_meas_col_name, time_column, filled_meas_col_name):
        
        outlier_mask =  np.full(len(df_meas_long), False, dtype = bool)
        shift_points = (df_meas_long['segments'] != df_meas_long['segments'].shift())
        df_meas_long['test'] = df_meas_long[adapted_meas_col_name].copy()

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
        df_meas_long[adapted_meas_col_name] = np.where(outlier_mask, np.nan, df_meas_long[adapted_meas_col_name])
        df_meas_long['selene_improved_spikes'] = outlier_mask
        if df_meas_long['selene_improved_spikes'].any():
            ratio = (df_meas_long['selene_improved_spikes'].sum()/len(df_meas_long))*100
            print(f"There are {df_meas_long['selene_improved_spikes'].sum()} spikes in this timeseries according to Selene. This is {ratio}% of the overall dataset.")

        #make Plots
        #Analyse spike detection
        self.helper.plot_df(df_meas_long[time_column], df_meas_long[adapted_meas_col_name],'Water Level','Timestamp ','SELENE: Measured water level wo outliers and spikes in 1 min timestamp (all) (improved)')
        #self.helper.plot_two_df(df_meas_long[time_column][self.min_value_plotting:self.max_value_plotting], df_meas_long[adapted_meas_col_name][self.min_value_plotting:self.max_value_plotting],'Water Level', df_meas_long[quality_column_name][self.min_value_plotting:self.max_value_plotting], 'Quality flag', 'Timestamp ','SELENE: Measured water level wo outliers and spikes in 1 min timestamp (zoomed to max spike')
        #self.helper.plot_two_df_same_axis(df_meas_long[time_column][self.min_value_plotting:self.max_value_plotting], df_meas_long[adapted_meas_col_name][self.min_value_plotting:self.max_value_plotting],'Water Level', 'Water Level (corrected)', df_meas_long['test'][self.min_value_plotting:self.max_value_plotting], 'Timestamp ', 'Water Level (measured)', 'SELENE: Measured water level wo outliers and spikes in 1 min timestamp vs orig (zoomed to max spike) (improved)')
        #self.helper.plot_two_df(df_meas_long[time_column], df_meas_long['test'],'Water Level', df_meas_long[quality_column_name], 'Quality flag', 'Timestamp ','SELENE: Measured water level in 1 min timestamp incl. flags')
        
        true_indices = df_meas_long['selene_improved_spikes'][df_meas_long['selene_improved_spikes']].index
        #More plots
        for i in range(1, 41):
            min = builtins.max(0,(random.choice(true_indices))-2000)
            max = min + 2000
            self.helper.plot_two_df_same_axis(df_meas_long[time_column][min:max], df_meas_long[adapted_meas_col_name][min:max],'Water Level', 'Water Level (corrected)', df_meas_long['test'][min:max], 'Timestamp ', 'Water Level (measured)', f'SELENE Graph{i}- Corredted spikesS')
            self.helper.plot_two_df_same_axis(df_meas_long[time_column][min:max], df_meas_long[adapted_meas_col_name][min:max],'Water Level', 'Water Level (corrected)', df_meas_long[filled_meas_col_name][min:max], 'Timestamp ', 'Modelled Water Level', f'SELENE Graph{i}- Spline vs measurement)')
        
        del df_meas_long['test']

        return df_meas_long
    
    def rmse(self, predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean()) 
        
    """
    Using ML for spike detection
    """    

    def remove_spikes_ml(self, data, filled_data_column, adapted_meas_col_name, time_column):

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
                relev_df['day_of_week'] = relev_df[time_column].dayofweek
                relev_df['month'] = relev_df[time_column].month
                relev_df['hour'] = relev_df[time_column].hour
                relev_df['lag_05tide'] = relev_df[filled_data_column].shift(372)  #min max amplitude
                relev_df['lag_1tide'] = relev_df[filled_data_column].shift(745)  # min min cycle development
                relev_df['rolling_mean'] = relev_df[adapted_meas_col_name].rolling(window=745).mean()  # 7-day mean
                relev_df['rolling_std'] = relev_df[adapted_meas_col_name].rolling(window=745).std()
                print('here1')
                #Extract relevant output and input features
                X = data[['lag_1tide', 'lag_05tide', 'rolling_mean', 'rolling_std', 'hour', 'month', 'month']]
                y = data[filled_data_column]

                # Split into train and test
                X_train, X_test = X[:-(self.test_size*len(relev_df))], X[-(self.test_size*len(relev_df)):]
                y_train, y_test = y[:-(self.test_size*len(relev_df))], y[-(self.test_size)*len(relev_df):]

                # Initialize XGBoost model
                model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
                # Final training on full train set
                model.fit(X_train, y_train)
                print('here2')

                # Test predictions
                y_pred_test = model.predict(X_test)
                residuals = np.abs(data[-(self.test_size)*len(relev_df):, adapted_meas_col_name] - y_pred_test)

                # Threshold for anomaly detection
                threshold = np.mean(residuals) + 2 * np.std(residuals)
                outlier_mask[start_index:end_index] = np.where(residuals > threshold)[0]

        #Mark & remove outliers
        data[adapted_meas_col_name] = np.where(outlier_mask, np.nan, data[adapted_meas_col_name])
        data['ml_detected_spikes'] = outlier_mask
        if data['ml_detected_spikes'].any():
            ratio = (data['ml_detected_spikes'].sum()/len(data))*100
            print(f"There are {data['ml_detected_spikes'].sum()} spikes in this timeseries according to Selene. This is {ratio}% of the overall dataset.")

        #Analyse spike detection
        self.helper.plot_df(data[time_column], data[adapted_meas_col_name],'Water Level','Timestamp ','SELENE: Measured water level wo outliers and spikes in 1 min timestamp (all) (improved)')

        true_indices = data['ml_detected_spikes'][data['ml_detected_spikes']].index
        #More plots
        for i in range(1, 41):
            min = builtins.max(0,(random.choice(true_indices))-2000)
            max = min + 2000
            self.helper.plot_two_df_same_axis(data[time_column][min:max], data[adapted_meas_col_name][min:max],'Water Level', 'Water Level (corrected)', data['test'][min:max], 'Timestamp ', 'Water Level (measured)', f'ML spike Graph{i}- Corredted spikesS')
            self.helper.plot_two_df_same_axis(data[time_column][min:max], data[adapted_meas_col_name][min:max],'Water Level', 'Water Level (corrected)', data[filled_data_column][min:max], 'Timestamp ', 'Modelled Water Level', f'ML Spike Graph{i}- Spline vs measurement)')
        
        del data['test']

        return data