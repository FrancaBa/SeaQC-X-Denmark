##########################################################
## Written by frb for GronSL project (2024-2025)        ##
## To detect constant slope/ linear interpolated values ##
##########################################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

import source.helper_methods as helper

"""
Code parts from hydrology unit: Floods Monitoring Systems/OVE_hydro_code/kvalitetssikering/qa_interpolation.py
"""

class Interpolation_Detector():
        
        
    def __init__(self):
        # Define the window size to check for a constant slope (= linear interpolation)
        self.window_size_const_gradient = 7

        self.helper = helper.HelperMethods()


    def set_output_folder(self, folder_path):
        folder_path = os.path.join(folder_path,'interpolated periods')

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.helper.set_output_folder(folder_path)


    def run_interpolation_detection(self, data, value_column, column_time, information):
        """
        Mark all periods which are probably linear interpolated. This means periods that have a constant slope over self.window_size_const_gradient (now 7 timesteps).

        Input:
        -Main dataframe [df]
        -Column name with measurements to analyse [str]
        -Column name of timestamp column [str]
        """
        gradient_mask = self.find_constant_slope(data, value_column, column_time)

        #Remove interpolated periods from copied ts for visual analysis
        data.loc[gradient_mask, value_column] = np.nan

        #Flag the interpolated periods in boolean mask
        data['interpolated_value'] = gradient_mask

        if gradient_mask.any():
            ratio = (gradient_mask.sum()/len(data))*100
            print(f"There are {gradient_mask.sum()} interpolated values in this timeseries. This is {ratio}% of the overall dataset.")
            information.append([f"There are {gradient_mask.sum()} interpolated values in this timeseries. This is {ratio}% of the overall dataset."])

        return data
    

    def find_constant_slope(self, data, data_column_name, column_time):
        """
        Function that returns the points that might be interpolated based on a constant gradient between points.

        Input:
        -Main dataframe [df]
        -Column name with measurements to analyse [str]
        -Column name of timestamp column [str]
        """

        # Compute the differential of the values and time (in min).
        value_diffs = data[data_column_name].diff()
        time_diffs = data[column_time].diff().dt.total_seconds()/60

        # Compute the values of the slopes and sets the ones with a slope of 0 equal to nan, as we find them as constant values instead.
        data['slope'] = np.divide(value_diffs, time_diffs)
        data.loc[data['slope'] == 0, 'slope'] = np.nan

        # Check if the value is constant over a window of 'self.window_constant_value' min (counts only non-nan)
        # Step 1: Identify where the values change, ignoring NaNs
        # Step 2: Assign a unique group ID for each sequence of the same value
        # Step 3: Count the size of each group, and check if each run is at least 'window_size_const_gradient' entries long
        # Step 4: Create the mask based on the length of consecutive identical values
        is_new_value = (data['slope'] != data['slope'].shift()) | data['slope'].isna()
        groups = is_new_value.cumsum()
        group_sizes = data.groupby(groups)['slope'].transform('size')
        gradient_mask = (group_sizes >= self.window_size_const_gradient) & data['slope'].notna()

        #print details on the constant gradient check and plts for visual analysis
        if gradient_mask.any():
            print(f"There are {gradient_mask.sum()} values with constant gradient over a period of {self.window_size_const_gradient} minutes in this timeseries.")
            true_indices = gradient_mask[gradient_mask].index
            data['y_nan'] = data[data_column_name]
            data.loc[gradient_mask, 'y_nan'] = np.nan
            self.helper.plot_two_df_same_axis(data[column_time][true_indices[0]-60:true_indices[0]+60], data[data_column_name][true_indices[0]-60:true_indices[0]+60],'Water Level', 'Water Level', data['y_nan'][true_indices[0]-60:true_indices[0]+60], 'Timestamp', 'Interpolated WL','Constant gradient period in TS')
            self.helper.plot_two_df_same_axis(data[column_time][true_indices[-1]-60:true_indices[-1]+60], data[data_column_name][true_indices[-1]-60:true_indices[-1]+60],'Water Level', 'Water Level', data['y_nan'][true_indices[-1]-60:true_indices[-1]+60], 'Timestamp', 'Interpolated WL', 'Constant gradient period 2.0 in TS')
            del data['y_nan']

        del data['slope']
        
        return gradient_mask
