import numpy as np
import pandas as pandas
import matplotlib.pyplot as plt
import datetime
import os
import random 

from scipy.interpolate import UnivariateSpline

"""
Contains different approaches to fill NaNs in a timeseries
"""

import source.helper_methods as helper

class MissingDataFiller():

    def __init__(self):

        self.nan_threshold = 375

        #to fit a spline
        self.splinelength = 14 #hours
        self.splinedegree = 3 #2

        #to fit polynomial
        self.order_ploynomial_fit = 2

        self.helper = helper.HelperMethods()

    def set_output_folder(self, folder_path):
        folder_path = os.path.join(folder_path,'segmentation and filling')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.helper.set_output_folder(folder_path)

    def segmentation_ts(self, data, data_column, time_column, segment_column):

        # Define the maximum allowed consecutive NaN length
        data[segment_column] = 0

        # Identify NaN regions
        is_nan = data[data_column].isna()

        # Group consecutive entries and count sizes
        group = is_nan.ne(is_nan.shift()).cumsum()
        group_sizes = group.map(group.value_counts())

        # Set 1 for rows where NaN group size is greater than chosen threshold
        data.loc[(is_nan) & (group_sizes > self.nan_threshold), segment_column] = 1

        #Check change points
        data['segments_filled'] = data[segment_column].ne(data[segment_column].shift()).cumsum()

        #Check a change period
        #may_2006_df = data[(data[time_column].dt.year == 2006) & (data[time_column].dt.month == 5)]
        #if len(may_2006_df)!= 0:
        #    self.helper.plot_two_df_same_axis(may_2006_df[time_column], may_2006_df[data_column],'Water Level', 'Water Level', may_2006_df[segment_column], 'Timestamp', 'Segment',f'Segment Graph 0')
        #    print(may_2006_df[1050:1100])

        #for segment
        shift_points = (data[segment_column] != data[segment_column].shift())
        print(f'This measurement series contains {len(data[segment_column][shift_points])/2} segments with measurements.')

        return data
    
    def polynomial_fill_data_column(self, data, data_column, time_column, segment_column):
        
        #for segment
        shift_points = (data[segment_column] != data[segment_column].shift())
        data['poly_interpolated_data'] = np.nan

        for i in range(0,len(data[segment_column][shift_points]), 1):  
            start_index = data[segment_column][shift_points].index[i]
            if i == len(data[segment_column][shift_points])-1:
                end_index = len(data)
            else:
                end_index = data[segment_column][shift_points].index[i+1]
            if data[segment_column][start_index] == 0:
                y_interpolated = data.loc[start_index:end_index, data_column].interpolate(method='polynomial', order= self.order_ploynomial_fit)
                data.loc[start_index:end_index,'poly_interpolated_data'] = y_interpolated
                
                if end_index - start_index > 1000:
                    self.helper.plot_two_df_same_axis(data[time_column][start_index:start_index+1000], data[data_column][start_index:start_index+1000],'Water Level', 'Water Level (corrected)', data['poly_interpolated_data'][start_index:start_index+1000], 'Timestamp ', 'Interpolated Water Level', f'Interpolated data Graph{i}- Measured water level vs Interpolated values (start)')
                    self.helper.plot_two_df_same_axis(data[time_column][end_index-500:end_index], data[data_column][end_index-500:end_index],'Water Level', 'Water Level (corrected)', data['poly_interpolated_data'][end_index-500:end_index], 'Timestamp ', 'Interpolated Water Level', f'Interpolated data Graph{i}- Measured water level vs Interpolated values (end)')

                self.helper.plot_two_df_same_axis(data[time_column][start_index:end_index], data[data_column][start_index:end_index],'Water Level', 'Water Level (corrected)', data['poly_interpolated_data'][start_index:end_index], 'Timestamp ', 'Interpolated Water Level', f'Interpolated data Graph{i}- Measured water level vs Interpolated values')
        
        return data
    
    def polynomial_fitted_data_column(self, data, data_column, time_column, segment_column, fitted_data):
        
        #for segment
        shift_points = (data[segment_column] != data[segment_column].shift())
        #assuming a 1-min time step
        winsize = (self.splinelength*60)
        data['poly_fitted_data'] = np.nan

        for i in range(0,len(data[segment_column][shift_points]), 1):  
            start_index = data[segment_column][shift_points].index[i]
            if i == len(data[segment_column][shift_points])-1:
                end_index = len(data)
            else:
                end_index = data[segment_column][shift_points].index[i+1]
            if data[segment_column][start_index] == 0:
                end_loop = False
                for start in range(start_index, end_index, winsize):
                    if (start + winsize*2) > end_index:
                        end = end_index-1
                        end_loop = True
                    else:
                        end = start + winsize
                    x_window = data.loc[start:end,time_column]
                    x_numeric = (x_window - x_window.min()).dt.total_seconds() / 60  # Convert to minutes
                    y_window = data.loc[start:end, fitted_data]

                    coefficients = np.polyfit(x_numeric, y_window, 6)

                    # Create a polynomial function from the coefficients
                    poly = np.poly1d(coefficients)
                    # Generate fitted values
                    y_fit = poly(x_numeric)
                    data.loc[start:end,'poly_fitted_data'] = y_fit
                    if end_loop:
                        break
               
                self.helper.plot_two_df_same_axis(data[time_column][start_index:end_index], data[data_column][start_index:end_index],'Water Level', 'Water Level (corrected)', data['poly_fitted_data'][start_index:end_index], 'Timestamp ', 'Polynomial fitted data Water Level', f'Polynomial fitted graph{start_index}')
            
        return data
    

    def spline_fill_measurement_column(self, data, data_column, time_column, segment_column):
        """
        Make a spline to measurements over 14 hours
        """
        interp_values = np.zeros_like(data[data_column])  # Initialize array for results
        
        #for segment
        shift_points = (data[segment_column] != data[segment_column].shift())
        #assuming a 1-min time step
        winsize = (self.splinelength*60)

        for i in range(0,len(data[segment_column][shift_points]), 1):  
            start_index = data[segment_column][shift_points].index[i]
            if i == len(data[segment_column][shift_points])-1:
                end_index = len(data)
            else:
                end_index = data[segment_column][shift_points].index[i+1]
            #Fit only segments with data
            if data[segment_column][start_index] == 0:
                # Loop over the series in windows
                for start in range(start_index, end_index, winsize):
                    if (start_index + (2*winsize)) > end_index:
                        end = end_index
                        x_window = data.index[start:end]
                        y_window = data[data_column][start:end]

                        # Fit a spline to the masked array, automatically skipping NaNs
                        # Mask out the NaNs
                        x_known = x_window[~np.isnan(y_window)]  # x values where y is not NaN
                        y_known = y_window[~np.isnan(y_window)]  # Corresponding y values (non-NaN)

                        if np.sum(~np.isnan(y_window)) < (self.splinedegree+1):
                            print('hey')
                            break

                        spline = UnivariateSpline(x_known, y_known, s=1, k = self.splinedegree)
                                    
                        # Evaluate the spline at all original `x` points
                        fitted_y = spline(x_window)
                                    
                        #Set interpolated values on comparison ts
                        interp_values[start:end] = fitted_y
                        break
                    else:
                        end =start + winsize
                    x_window = data.index[start:end]
                    y_window = data[data_column][start:end]

                    # Fit a spline to the masked array, automatically skipping NaNs
                    # Mask out the NaNs
                    x_known = x_window[~np.isnan(y_window)]  # x values where y is not NaN
                    y_known = y_window[~np.isnan(y_window)]  # Corresponding y values (non-NaN)

                    if np.sum(~np.isnan(y_window)) < (self.splinedegree+1):
                        print('hey 2')
                        pass
                    else:
                        spline = UnivariateSpline(x_known, y_known, s=1, k = self.splinedegree)
                                    
                        # Evaluate the spline at all original `x` points
                        fitted_y = spline(x_window)
                                    
                        #Set interpolated values on comparison ts
                        interp_values[start:end] = fitted_y

                self.helper.plot_two_df_same_axis(data[time_column][start_index:end_index], data[data_column][start_index:end_index],'Water Level', 'Water Level (corrected)', interp_values[start_index:end_index], 'Timestamp ', 'Interpolated Water Level', f'Spline fitted Graph{start}- Measured water level vs spline values')
                
                """
                    if np.sum(~np.isnan(y_window)) > winsize/4:
                        # Fit a spline to the masked array, automatically skipping NaNs
                        # Mask out the NaNs
                        x_known = x_window[~np.isnan(y_window)]  # x values where y is not NaN
                        y_known = y_window[~np.isnan(y_window)]  # Corresponding y values (non-NaN)

                        spline = UnivariateSpline(x_known, y_known, s=1, k = self.splinedegree)
                        
                        # Evaluate the spline at all original `x` points
                        fitted_y = spline(x_window)
                        
                        #Set interpolated values on comparison ts
                        interp_values[start:end] = fitted_y
                    else:
                        interp_values[start:end] = np.nan
                """
                
        data['spline_filled_data'] = interp_values

        return data
    
    def compare_filled_measurements(self, data, data_column, time_column, segment_column):

        shift_points = (data[segment_column] != data[segment_column].shift())

        for i in range(0,len(data[segment_column][shift_points]), 1):  
            start_index = data[segment_column][shift_points].index[i]
            if i == len(data[segment_column][shift_points])-1:
                end_index = len(data)
            else:
                end_index = data[segment_column][shift_points].index[i+1]
            if data[segment_column][start_index] == 0:
                
                if end_index - start_index > 1000:
                    self.helper.plot_two_df_same_axis(data[time_column][start_index:start_index+1000], data['spline_filled_data'][start_index:start_index+1000],'Water Level', 'Spline fitted data',data['poly_interpolated_data'][start_index:start_index+1000], 'Timestamp ', 'Poly Interpolated Water Level', f'Interpolated data Graph{i}- Spline vs polynomial interpolated values (start)')
                    self.helper.plot_two_df_same_axis(data[time_column][end_index-500:end_index], data['spline_filled_data'][end_index-500:end_index],'Water Level', 'Spline fitted data', data['poly_interpolated_data'][end_index-500:end_index], 'Timestamp ', 'Poly Interpolated Water Level', f'Interpolated data Graph{i}- Spline vs polynomial interpolated values (end)')

                self.helper.plot_two_df_same_axis(data[time_column][start_index:end_index], data['spline_filled_data'][start_index:end_index],'Water Level', 'Spline fitted data', data['poly_interpolated_data'][start_index:end_index], 'Timestamp ', 'Poly Interpolated Water Level', f'Interpolated data Graph{i}- Spline vs polynomial interpolated values')
 