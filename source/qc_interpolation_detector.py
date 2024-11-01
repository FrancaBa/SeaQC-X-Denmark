import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

"""
Code parts from hydrology unit: Floods Monitoring Systems/OVE_hydro_code/kvalitetssikering/qa_interpolation.py
"""

class Interpolation_Detector():
        
    def __init__(self):
        # Define the window size
        self.window_size_const_gradient = 10
        self.rolling_window = 100
        self.min_periods = 20

    def run_interpolation_detection(self, data, value_column, column_time, qf_classes, quality_column_name):
        """
        Mark all periods which are probably interpolated. However, don't change those values for now.
        """
        interpolated_mask = self.find_constant_gradient(data, value_column, column_time)
        gradient_mask = self.find_constant_gradient(data, value_column)

        #Flag the interpolated periods
        data.loc[(data[quality_column_name] == qf_classes['good_data']) & (interpolated_mask), quality_column_name] = qf_classes['interpolated_value']
        data.loc[(data[quality_column_name] == qf_classes['good_data']) & (gradient_mask), quality_column_name] = qf_classes['interpolated_value']

        if qf_classes['interpolated_value'] in data[quality_column_name].unique():
            ratio = (len(data[data[quality_column_name] == qf_classes['interpolated_value']])/len(data))*100
            print(f"There are {len(data[data[quality_column_name] == qf_classes['interpolated_value']])} interpolated values in this timeseries. This is {ratio}% of the overall dataset.")
        
        return data
    
    def find_constant_gradient(self, data, data_column_name, column_time):
        """
        Function that returns the points that might be interpolated based on a constant gradient between points.
        """

        # Compute the differential of the values and time (in min).
        value_diffs = data[data_column_name][1:] - data[data_column_name][:-1]
        time_diffs = ((data[column_time][1:] - data[column_time][:-1])).total_seconds() / 60

        # Compute the values of the slopes and sets the ones with a slope of 0 equal to nan, as we find them as
        # flatlines instead.
        slope_values = np.divide(value_diffs, time_diffs)
        slope_values[slope_values == 0] = np.nan

        # Compute the differential of the slopes.
        slope_value_diffs = slope_values[1:] - slope_values[:-1]

        # Create a boolean mask where the differences are the same for the window
        interpolated_mask = (slope_value_diffs[:, None] == slope_value_diffs).sum(axis=1) >= self.window_size_const_gradient

        return interpolated_mask

    def find_constant_gradient(self, data, column):
        """ 
        Real data often has more natural variability, while interpolated sections are typically smoother. Thus, check for section with a smaller standard deviation.  
        """
        # Calculate the rolling standard deviation and rolling variance
        data['rolling_std']= data[column].rolling(window=self.rolling_window, min_periods=self.min_periods).std()
        #df['rolling_var'] = df['values'].rolling(window=window_size).var()

        # Calculate the median of the rolling standard deviation and rolling variance
        rolling_std_median = data['rolling_std'].median()
        #rolling_var_median = df['rolling_var'].median()

        # Plotting the box plot using Matplotlib and Seaborn
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=data['rolling_std'], color='lightblue')
        plt.title("Distribution of Rolling Standard Deviation")
        plt.xlabel("Rolling Std Dev")
        plt.show()

        # Create a boolean mask where small std are flagged
        gradient_mask = 1 #TBD

        return gradient_mask
