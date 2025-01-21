##########################################################
## Written by frb for GronSL project (2024-2025)        ##
## This script markes struck values (=constant values). ##
##########################################################

import os
import numpy as np

import source.helper_methods as helper

class StuckValuesDetector(): 

    def __init__(self):
        self.helper = helper.HelperMethods()

        #Number of constant entries needed to mark a period as constant
        self.window_constant_value = 3

    def set_output_folder(self, folder_path):
        self.folder_path = os.path.join(folder_path,'Stuck values')

        #generate output folder for graphs and other docs
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

        self.helper.set_output_folder(self.folder_path)


    def run(self, df_meas_long, time_column, adapted_meas_col_name, information):

        df_meas_long['test'] = df_meas_long[adapted_meas_col_name].copy()

        # Check if the value is constant over a window of 'self.window_constant_value' min (counts only non-nan)
        # Step 1: Identify where the values change, ignoring NaNs
        # Step 2: Assign a unique group ID for each sequence of the same value
        # Step 3: Count the size of each group, and check if each run is at least 'self.window_constant_value' entries long
        # Step 4: Create the mask based on the length of consecutive identical values
        is_new_value = (df_meas_long[adapted_meas_col_name] != df_meas_long[adapted_meas_col_name].shift()) | df_meas_long[adapted_meas_col_name].isna()
        groups = is_new_value.cumsum()
        group_sizes = df_meas_long.groupby(groups)[adapted_meas_col_name].transform('size')
        constant_mask = (group_sizes >= self.window_constant_value) & df_meas_long[adapted_meas_col_name].notna()

        # Mask the constant values and add it as a column
        df_meas_long[adapted_meas_col_name] = np.where(constant_mask, np.nan, df_meas_long[adapted_meas_col_name])
        df_meas_long['stuck_value'] = constant_mask

        # Get indices where the mask is True (as check that approach works)
        if constant_mask.any():
            true_indices = constant_mask[constant_mask].index
            self.helper.plot_two_df_same_axis(df_meas_long[time_column][true_indices[0]-30:true_indices[0]+50], df_meas_long[adapted_meas_col_name][true_indices[0]-30:true_indices[0]+50],'Water Level', 'Water Level',  df_meas_long['test'][true_indices[0]-30:true_indices[0]+50], 'Timestamp', 'WL removed','Constant period in TS')
            self.helper.plot_two_df_same_axis(df_meas_long[time_column][true_indices[-1]-30:true_indices[-1]+50], df_meas_long[adapted_meas_col_name][true_indices[-1]-30:true_indices[-1]+50],'Water Level', 'Water Level', df_meas_long['test'][true_indices[-1]-30:true_indices[-1]+50], 'Timestamp', 'WL removed','Constant period in TS (2)')

        #print details on the constant value check
        ratio = (constant_mask.sum()/len(df_meas_long))*100
        print(f"There are {constant_mask.sum()} constant values in this timeseries. This is {ratio}% of the overall dataset.")
        information.append([f"There are {constant_mask.sum()} constant values in this timeseries. This is {ratio}% of the overall dataset."])
            
        del df_meas_long['test']

        return df_meas_long