#################################################################################
## Written by frb for GronSL project (2024-2025)                               ##
## This script markes short good periods between bad periods as probably good. ##
#################################################################################

import os
import numpy as np
import random
import builtins

import source.helper_methods as helper

class ProbablyGoodDataFlagger():

    def __init__(self):
        self.helper = helper.HelperMethods()

        #Define periods with less then 10 min of good measurements between bad measurements as probably good 
        self.probably_good_threshold = 10

    def set_output_folder(self, folder_path):
        folder_path = os.path.join(folder_path,'probably good periods')

        #generate output folder for graphs and other docs
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.helper.set_output_folder(folder_path)


    def run(self, data, adapted_meas_col_name, time_column, measurement_column, information):
        """
        Based on a mask counting the consecutive false (good data as tests have been negativ), define probably good periods. Data is good, 
        but surrounded by a lot of bad data which makes it a little bit less important.
      
        Input: 
        -data: Main dataframe [df]
        -adapted_meas_col_name: Column name for measurement series [str]
        -time_column: Column name for timestamp [str]
        -measurement_column: Column name for raw measurement [str]
        """

        boolean_columns = data.select_dtypes(include='bool')
        #statement for analysis
        all_combined = boolean_columns.any(axis=1)
        ratio = (all_combined.sum()/len(data))*100
        print(f"There are {all_combined.sum()} elements in this timeseries which have been flagged by different QC tests. This is {ratio}% of the overall dataset.")
        information.append([f"There are {all_combined.sum()} elements in this timeseries which have been flagged by different QC tests. This is {ratio}% of the overall dataset."])
        
        del boolean_columns['missing_values']
        combined = boolean_columns.any(axis=1)
        data['combined_mask'] = combined

        # Apply the function to the column
        data['probably_good_mask'] = self.mask_fewer_than_x_consecutive_false(data['combined_mask'])

        #Check if selection is a probably good period and yes, remove value
        data[adapted_meas_col_name] = np.where(data['probably_good_mask'], np.nan, data[adapted_meas_col_name])

        ratio = (data['probably_good_mask'].sum()/len(data))*100
        print(f"There are {data['probably_good_mask'].sum()} elements in this timeseries which have failed subtests around them and split the segements to not by default trustworthy. This is {ratio}% of the overall dataset.")
        information.append([f"There are {data['probably_good_mask'].sum()} elements in this timeseries which have failed subtests around them and split the segements to not by default trustworthy. This is {ratio}% of the overall dataset."])
        ratio = ((data['probably_good_mask'].sum()+all_combined.sum())/len(data))*100
        print(f"In total, there are {(data['probably_good_mask'].sum()+all_combined.sum())} flagged elements in this timeseries. This is {ratio}% of the overall dataset.")
        information.append([f"In total, there are {(data['probably_good_mask'].sum()+all_combined.sum())} flagged elements in this timeseries. This is {ratio}% of the overall dataset."])
        
        #Plot marked periods to check
        if data['probably_good_mask'].any():
            true_indices = data['probably_good_mask'][data['probably_good_mask']].index
            for i in range(1, 41):
                min = builtins.max(0,(random.choice(true_indices))-4250)
                max = builtins.min(min + 4250, len(data))
                self.helper.plot_two_df_same_axis(data[time_column][min:max], data[adapted_meas_col_name][min:max],'Water Level', 'Water Level (corrected)', data[measurement_column][min:max], 'Timestamp', 'Water Level (measured)',f'Graph-probably good period detected {i}')
        
        return data
    
        
    def mask_fewer_than_x_consecutive_false(self, series):
        """
        Vectorized solution to create a mask counting the consecutive false (= good data as tests have been negativ). If more then self.probably_good_threshold false in a row, mark periods as test fail.

        Input: pandas series
        """
        # Convert to integers (True -> 1, False -> 0)
        arr = series.to_numpy(dtype=int)
        
        # Find where the values are False
        is_false = arr == 0
        
        # Identify boundaries of False segments
        boundaries = np.diff(np.concatenate(([0], is_false, [0])))
        starts = np.where(boundaries == 1)[0]  # Start of False segments
        ends = np.where(boundaries == -1)[0]  # End of False segments
        
        # Create a mask (default is True for all)
        mask = np.ones_like(arr, dtype=bool)
        
        # Determine where False segments >= x and mask those regions
        for start, end in zip(starts, ends):
            if end - start >= self.probably_good_threshold:
                mask[start:end] = False

        #If value already true in previous test and not only now because of neighbours, set them to False
        mask_corrected = np.where((mask == True) & (is_false == False), False, mask)

        return mask_corrected