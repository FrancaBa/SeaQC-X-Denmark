#########################################################################################
## Written by frb for GronSL project (2024-2025)                                       ##
## This script markes outliers and noisy periods based on their change in cm per time. ##
#########################################################################################

import os
import numpy as np
import builtins
import random

import source.helper_methods as helper
import source.qc_spike as qc_spike

class ImplausibleChangeDetector(): 

    def __init__(self):
        self.helper = helper.HelperMethods()

        #Parameters needed to define an impossible change rate (maximum possible change in 30cm in 10 min)
        self.threshold_max_change = None
        self.range_spike_pair = None
        #Parameters needed to define a noisy periods
        self.shifting_periods = None
        self.outliers_per_hour = None

    def set_output_folder(self, folder_path):
        self.folder_path = os.path.join(folder_path,'implausible change')

        #generate output folder for graphs and other docs
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

        self.helper.set_output_folder(self.folder_path)

    #Load relevant parameters for this QC test from conig.json
    def set_parameters(self, params):
        #Range to find the next neighbour
        self.search_step_neighbours = params['implausible_change']['search_step_neighbours']
        

        #Parameters needed to define an impossible change rate (maximum possible change in 30cm in 10 min)
        self.threshold_max_change = params['implausible_change']['threshold_max_change']
        self.range_spike_pair = params['implausible_change']['range_spike_pair']

        #Parameters needed to define a noisy periods
        self.shifting_periods = params['implausible_change']['shifting_periods']
        self.outliers_per_hour = params['implausible_change']['outliers_per_hour']

    def run(self, df, adapted_meas_col_name, time_column, information, original_length, suffix):
        """
        Based on neighboring values, detect the change in cm per time. If single big change over a longer period, mark it as spike. 
        If several smaller changes in a period, mark it as noisy period.
      
        Input: 
        -data: Main dataframe [df]
        -adapted_meas_col_name: Column name for measurement series [str]
        -time_column: Column name for timestamp [str]
        -Information list where QC report is collected [lst]
        -Length of original measurement series [int]
        -suffix: ending for columns and graphs in order to run in different modes [str]
        """
        self.original_length = original_length
        #Get the difference between measurement and flag all measurement wth change more than x cm in 1 min
        # If there are more then 2 outliers in 1 hour keep them, as this could indicate a more systemetic error in the measurement

        #Call method from detect spike values
        spike_detection = qc_spike.SpikeDetector()
        #spike_detection.set_output_folder(self.folder_path)
        spike_detection.get_valid_neighbours(df, adapted_meas_col_name, 'next_neighbour', True, max_distance=self.search_step_neighbours)
    
        #not really correct, but good enough for now!
        df['change'] = (df[adapted_meas_col_name]-df['next_neighbour']).abs()
        outlier_change_rate= df['change'] > self.threshold_max_change
        distributed_periods = df['change'] > self.shifting_periods

        self.run_single_spikes(df, outlier_change_rate, adapted_meas_col_name, time_column, information, suffix)
        self.run_noisy_periods(df, distributed_periods, adapted_meas_col_name, time_column, information, suffix)

        del df['change']
        del df['next_neighbour']

        return df

    def extract_outlier(self, outlier_change_rate):
        # When shifting back from outliers, there is a large jump again. Make sure that those jumps are not marked as outlier! This is not really working
        true_indices = outlier_change_rate[outlier_change_rate == True]
        sorted_values = np.sort(true_indices.index)

        # Step 2: Compute the forward differences between consecutive values
        differences = np.diff(sorted_values)

        # Step 3: Identify second values in pairs
        # A second value in a pair is where the difference between it and the previous value <= tolerance
        is_second_in_pair = np.zeros_like(sorted_values, dtype=bool)
        is_second_in_pair[1:] = differences <= self.range_spike_pair  # Mark second values of pairs

        # Step 4: Identify single values
        # A value is single if it is not part of any pair (both forward and backward checks fail)
        is_single = np.ones_like(sorted_values, dtype=bool)
        is_single[:-1] &= differences > self.range_spike_pair  # Check the forward difference
        is_single[1:] &= differences > self.range_spike_pair   # Check the backward difference

        # Step 5: Combine single values and second values of pairs
        result = np.concatenate([sorted_values[is_single], sorted_values[is_second_in_pair]])

        # Step 6: Output the result (sorted and unique)
        result = np.unique(result)
        outlier_change_rate[:] = False
        outlier_change_rate.loc[result] = True

        return outlier_change_rate
    
    def run_single_spikes(self, df, outlier_change_rate, adapted_meas_col_name, time_column, information, suffix):
        df['test'] = df[adapted_meas_col_name].copy()
        # When shifting back from outliers, there is a large jump again. Make sure that those jumps are not marked as outlier! This is not working 100%
        outlier_change_rate = self.extract_outlier(outlier_change_rate)

        #For windows exceeding the threshold, check proximity condition
        #To keep only single peaks periods
        for idx in outlier_change_rate[outlier_change_rate].index:
            # Extract indices of True values in the current window of 2 hours around the selected index
            window_indices = range(builtins.max(0, idx - 59), idx + 60)
            true_indices = [i for i in window_indices if outlier_change_rate[i]]

            if len(true_indices) > 2:
                # If not, set all `True` values in this window to `False`
                outlier_change_rate[window_indices] = False

        df[f'outlier_change_rate{suffix}'] = outlier_change_rate
        df['test'] = np.where(df[f'outlier_change_rate{suffix}'], np.nan, df['test'])

         # Get indices where the mask is True (as check that approach works)
        if df[f'outlier_change_rate{suffix}'].any():
            true_indices = df[f'outlier_change_rate{suffix}'][df[f'outlier_change_rate{suffix}']].index
            self.helper.plot_df(df[time_column][true_indices[0]-100:true_indices[0]+100], df[adapted_meas_col_name][true_indices[0]-100:true_indices[0]+100],'Water Level',f'Timestamp ','Max plausible change period in TS -{suffix}')
            self.helper.plot_df(df[time_column][true_indices[0]-100:true_indices[0]+100], df['test'][true_indices[0]-100:true_indices[0]+100],'Water Level','Timestamp ',f'Max plausible change period in TS (corrected) -{suffix}')
            self.helper.plot_df(df[time_column][true_indices[-1]-100:true_indices[-1]+100], df[adapted_meas_col_name][true_indices[-1]-100:true_indices[-1]+100],'Water Level','Timestamp ',f'Max plausible change period in TS (2) -{suffix}')
            self.helper.plot_df(df[time_column][true_indices[-1]-100:true_indices[-1]+100], df['test'][true_indices[-1]-100:true_indices[-1]+100],'Water Level','Timestamp ',f'Max plausible change period in TS (corrected) (2) -{suffix}')
            #More plots
            for i in range(1, 21):
                min = (random.choice(true_indices))-200
                max = min + 400
                self.helper.plot_two_df_same_axis(df[time_column][min:max], df['test'][min:max],'Water Level', 'Water Level (corrected)', df[adapted_meas_col_name][min:max], 'Timestamp', 'Water Level (measured)',f'Graph-Implausible change rate detected {i}-{suffix}')

        ratio = (df[f'outlier_change_rate{suffix}'].sum()/self.original_length)*100
        print(f"There are {df[f'outlier_change_rate{suffix}'].sum()} outliers in this timeseries which change their level within 15 min too much. This is {ratio}% of the overall dataset.")
        information.append([f"There are {df[f'outlier_change_rate{suffix}'].sum()} outliers in this timeseries which change their level within 15 min too much. This is {ratio}% of the overall dataset."])

        del df['test']


    def run_noisy_periods(self, df, distributed_periods, adapted_meas_col_name, time_column, information, suffix):

        df[f'noisy_period{suffix}'] = False
        df['test'] = df[adapted_meas_col_name].copy()

        # When shifting back from outliers, there is a large jump again. Make sure that those jumps are not marked as outlier! This is not working 100%
        distributed_periods = self.extract_outlier(distributed_periods)

        #For values exceeding the threshold, check proximity condition
        #To keep weird measurement periods, filter if other outliers are close by
        for idx in distributed_periods[distributed_periods].index:
            # Extract indices of True values in the current window of 4 hours around the selected index
            window_indices = range(builtins.max(df.index[0], idx - 119), builtins.min(idx + 120, df.index[-1]))
            true_indices = [i for i in window_indices if distributed_periods[i]]

            if (len(true_indices)/4) >= (self.outliers_per_hour):
                df.loc[window_indices,f'noisy_period{suffix}'] = True

        #Check if selection is a noisy period and yes, remove value
        df['test'] = np.where(df[f'noisy_period{suffix}'], np.nan, df['test'])

        ratio = (df[f'noisy_period{suffix}'].sum()/self.original_length)*100
        print(f"There are {df[f'noisy_period{suffix}'].sum()} noisy periods in this timeseries which change their level within a short timeframe a lot. This is {ratio}% of the overall dataset.")
        information.append([f"There are {df[f'noisy_period{suffix}'].sum()} noisy periods in this timeseries which change their level within a short timeframe a lot. This is {ratio}% of the overall dataset."])

        #Plot marked periods to check
        if df[f'noisy_period{suffix}'].any():
            true_indices = df[f'noisy_period{suffix}'][df[f'noisy_period{suffix}']].index
            for i in range(1, 21):
                min = (random.choice(true_indices))-2000
                max = min + 4000
                self.helper.plot_two_df_same_axis(df[time_column][min:max], df['test'][min:max],'Water Level', 'Water Level (corrected)', df[adapted_meas_col_name][min:max], 'Timestamp', 'Water Level (measured)',f'Graph-noisy period detected {i} -{suffix}')
        
        del df['test']