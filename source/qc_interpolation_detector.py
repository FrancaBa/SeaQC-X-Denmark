import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import source.helper_methods as helper

"""
Code parts from hydrology unit: Floods Monitoring Systems/OVE_hydro_code/kvalitetssikering/qa_interpolation.py
"""

class Interpolation_Detector():
        
    def __init__(self):
        # Define the window size
        self.window_size_const_gradient = 7
        self.rolling_window = 60
        self.min_periods = 20
        self.threshold = 1.1

        self.helper = helper.HelperMethods()

    def set_output_folder(self, folder_path):
        self.helper.set_output_folder(folder_path)

    def run_interpolation_detection(self, data, value_column, column_time, qf_classes, quality_column_name):
        """
        Mark all periods which are probably interpolated. However, don't change those values for now.
        """
        gradient_mask = self.find_constant_slope(data, value_column, column_time)
        distribution_mask = self.find_small_distribution(data, value_column, column_time)

        #Flag the interpolated periods
        data.loc[(data[quality_column_name] == qf_classes['good_data']) & (distribution_mask), quality_column_name] = qf_classes['interpolated_value']
        data.loc[(data[quality_column_name] == qf_classes['good_data']) & (gradient_mask), quality_column_name] = qf_classes['interpolated_value']

        if qf_classes['interpolated_value'] in data[quality_column_name].unique():
            ratio = (len(data[data[quality_column_name] == qf_classes['interpolated_value']])/len(data))*100
            print(f"There are {len(data[data[quality_column_name] == qf_classes['interpolated_value']])} interpolated values in this timeseries. This is {ratio}% of the overall dataset.")
        
        return data
    
    def find_constant_slope(self, data, data_column_name, column_time):
        """
        Function that returns the points that might be interpolated based on a constant gradient between points.
        """

        # Compute the differential of the values and time (in min).
        value_diffs = data[data_column_name].diff()
        time_diffs = data[column_time].diff().dt.total_seconds()/60

        # Compute the values of the slopes and sets the ones with a slope of 0 equal to nan, as we find them as
        # flatlines instead.
        data['slope'] = np.divide(value_diffs, time_diffs)
        data['slope'][data['slope'] == 0] = np.nan

        # Check if the value is constant over a window of 'self.window_constant_value' min (counts only non-nan)
        # Step 1: Identify where the values change, ignoring NaNs
        # Step 2: Assign a unique group ID for each sequence of the same value
        # Step 3: Count the size of each group, and check if each run is at least 'window_size_const_gradient' entries long
        # Step 4: Create the mask based on the length of consecutive identical values
        is_new_value = (data['slope'] != data['slope'].shift()) | data['slope'].isna()
        groups = is_new_value.cumsum()
        group_sizes = data.groupby(groups)['slope'].transform('size')
        gradient_mask = (group_sizes >= self.window_size_const_gradient) & data['slope'].notna()

        #print details on the constant gradient check
        if gradient_mask.any():
            print(f"There are {gradient_mask.sum()} values with constant gradient over a period of {self.window_size_const_gradient} minutes in this timeseries.")
            true_indices = gradient_mask[gradient_mask].index
            self.helper.plot_df(data[column_time][true_indices[0]-100:true_indices[0]+100], data[data_column_name][true_indices[0]-100:true_indices[0]+100],'Water Level','Timestamp ','Constant gradient period in TS')
            self.helper.plot_df(data[column_time][true_indices[7]-100:true_indices[7]+100], data[data_column_name][true_indices[7]-100:true_indices[7]+100],'Water Level','Timestamp ','Constant gradient period 2.0 in TS')
            
        return gradient_mask

    def find_small_distribution(self, data, column, column_time):
        """ 
        Real data often has more natural variability, while interpolated sections are typically smoother. Thus, check for section with a smaller standard deviation.  
        """
        # Calculate the rolling standard deviation and rolling variance
        data['rolling_std']= data[column].rolling(window=self.rolling_window, min_periods=self.min_periods).std()
        #df['rolling_var'] = df['values'].rolling(window=window_size).var()

        # Calculate the median of the rolling standard deviation and rolling variance
        rolling_std_median = data['rolling_std'].median()
        #rolling_var_median = df['rolling_var'].median()

        self.helper.plot_df(data[column_time], data['rolling_std'],'Stand. Deviation WL','Timestamp ','Small distribution values')

        # Identify interpolated values
        distribution_mask = (data['rolling_std'] < data['rolling_std'].min()*self.threshold)

        # Output results
        print("Median:", rolling_std_median)

        #print details on the small distribution check
        if distribution_mask.any():
            print(f"There are {distribution_mask.sum()} values with a small standard deviation in this timeseries.")
            true_indices = distribution_mask[distribution_mask].index
            self.helper.plot_df(data[column_time][true_indices[0]-100:true_indices[0]+100], data[column][true_indices[0]-100:true_indices[0]+100],'Water Level','Timestamp ','Small distribution period in TS')
            self.helper.plot_df(data[column_time][true_indices[-1]-100:true_indices[-1]+100], data[column][true_indices[-1]-100:true_indices[-1]+100],'Water Level','Timestamp ','Small distribution period 2.0 in TS')
            
        return distribution_mask
