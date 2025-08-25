##########################################################
## Written by frb for GronSL project (2024-2025)        ##
## This script markes struck values (=constant values). ##
##########################################################

import os
import numpy as np
import builtins
import random
import source.helper_methods as helper

class StuckValuesDetector(): 

    def __init__(self):
        self.helper = helper.HelperMethods()

        #Number of constant entries needed to mark a period as constant
        self.window_constant_value = None

    def set_output_folder(self, folder_path):
        self.folder_path = os.path.join(folder_path,'Stuck values')

        #generate output folder for graphs and other docs
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

        self.helper.set_output_folder(self.folder_path)

    #Load relevant parameters for this QC test from conig.json
    def set_parameters(self, params):
        #Number of constant entries needed to mark a period as constant
        self.window_constant_value = params['stuck_value']

    def run(self, df_meas_long, time_column, adapted_meas_col_name, information, original_length, suffix):
        """
        Detect stuck values based on constant measurement over time.

        Input:
        -Main dataframe [pandas df]
        -Column name of time & date information [str]
        -Column name of measurement series of interest [str]
        -Information list where QC report is collected [lst]
        -Length of original measurement series [int]
        -suffix: ending for columns and graphs in order to run in different modes [str]
        """
        
        df_meas_long['test'] = df_meas_long[adapted_meas_col_name].copy()
        df_meas_long['helper'] = df_meas_long['test'].copy().ffill()

        # Check if the value is constant over a window of 'self.window_constant_value' min (counts only non-nan)
        # Step 1: Identify where the values change, ignoring NaNs
        # Step 2: Assign a unique group ID for each sequence of the same value
        # Step 3: Count the size of each group, and check if each run is at least 'self.window_constant_value' entries long
        # Step 4: Create the mask based on the length of consecutive identical values
        #is_new_value = (df_meas_long[adapted_meas_col_name] != df_meas_long[adapted_meas_col_name].shift()) | df_meas_long[adapted_meas_col_name].isna()
        is_new_value = (df_meas_long['helper'] != df_meas_long['helper'].shift()) & df_meas_long['test'].notna()
        groups = is_new_value.cumsum()
        group_sizes = df_meas_long.groupby(groups)['test'].transform('size')
        constant_mask = (group_sizes >= self.window_constant_value) & df_meas_long[adapted_meas_col_name].notna()

        #Mask the constant values and add it as a column
        df_meas_long['test'] = np.where(constant_mask, df_meas_long[adapted_meas_col_name], np.nan)
        df_meas_long[f'stuck_value{suffix}'] = constant_mask

        # Get indices where the mask is True (as check that approach works)
        if constant_mask.any():
            true_indices = constant_mask[constant_mask].index
            max_range = builtins.min(31, len(true_indices))
            for i in range(1, max_range):
                x = random.choice(range(0,len(true_indices)))
                min = builtins.max(0,(true_indices[x]-100))
                max = builtins.min(len(df_meas_long), min+200)
                self.helper.plot_two_df_same_axis(df_meas_long[time_column][min:max], df_meas_long['test'][min:max], 'Water Level [m]', 'Stuck values', df_meas_long[adapted_meas_col_name][min:max], 'Timestamp', 'Measured Water Level', f'Constant period in TS{suffix}-{i}')

        #print details on the constant value check
        ratio = (constant_mask.sum()/original_length)*100
        print(f"There are {constant_mask.sum()} constant values in this time series. This is {ratio}% of the overall dataset.")
        information.append([f"There are {constant_mask.sum()} constant values in this time series. This is {ratio}% of the overall dataset."])
        
        #Remove constant periods, so they don't impact the assessment later on
        df_meas_long[adapted_meas_col_name] = np.where(constant_mask, np.nan, df_meas_long[adapted_meas_col_name])
        
        del df_meas_long['test']
        del df_meas_long['helper']

        return df_meas_long