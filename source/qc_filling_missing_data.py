####################################################
## Written by frb for GronSL project (2024-2025) ###
####################################################

"""
Contains different simple approaches to fill NaNs in a timeseries:
1. Polynomial interpolation
2. Spline fit
3. Polynomial fit
"""

import numpy as np
import pandas as pandas
import matplotlib.pyplot as plt
import datetime
import os
import random 

from scipy.interpolate import UnivariateSpline

import source.helper_methods as helper

class MissingDataFiller():


    def __init__(self):
        #Split ts in segments
        self.nan_threshold = None
        #to fit a spline
        self.splinelength = None #hours
        self.splinedegree = None 
        #to fit a polynomial curve
        self.order_ploynomial_fil = None

        self.helper = helper.HelperMethods()


    def set_output_folder(self, folder_path):
        folder_path = os.path.join(folder_path,'segmentation and filling')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.helper.set_output_folder(folder_path)

    #Load relevant parameters for this QC test from conig.json
    def set_parameters(self, params):
        #Threshold to split ts in segments
        self.nan_threshold = params['segmentation']['nan_threshold']
        #to fit a spline
        self.splinelength = params['segmentation']['splinelength'] #min
        self.splinedegree = params['segmentation']['splinedegree']
        #to fit a polynomial curve
        self.order_ploynomial_fil = params['segmentation']['order_ploynomial_fil']
        self.polydegree = params['segmentation']['poly_fitted_degree']

    def segmentation_ts(self, data, data_column, segment_column, information):
        """
        Splits timeseries in various segments based on empty periods between measurements. If empty period longer than self.nan_threshold (here: 375 min or 6 hours), then make a new segment.

        Input:
        -Main dataframe [df]
        -Column name with measurements to analyse [str]
        -Column name for segments [str]
        """

        #Generate empty column to save segments later
        data[segment_column] = 0

        # Identify NaN regions
        is_nan = data[data_column].isna()

        # Group consecutive entries and count sizes
        group = is_nan.ne(is_nan.shift()).cumsum()
        group_sizes = group.map(group.value_counts())

        # Set 1 for rows where NaN group size is greater than chosen threshold
        data.loc[(is_nan) & (group_sizes > self.nan_threshold), segment_column] = 1

        #Get change points between segments
        data['segments_filled'] = data[segment_column].ne(data[segment_column].shift()).cumsum()

        #Get number of segments
        shift_points = (data[segment_column] != data[segment_column].shift())
        print(f'This measurement series contains {(data[segment_column][shift_points]==0).sum()} segments with measurements.')
        information.append([f'This measurement series contains {(data[segment_column][shift_points]==0).sum()} segments with measurements.'])

        return data
    

    def polynomial_fill_data_column(self, data, data_column, time_column, segment_column):
        """
        In measurement segments, replace all NaN entries with a polynomial interpolated value. A continuous ts is needed for some QCs later on.

        Input:
        -Main dataframe [df]
        -Column name with measurements to analyse [str]
        -Column name for timestamp [str]
        -Column name for segments [str]
        """
        
        #Get start and end point of each segment
        shift_points = (data[segment_column] != data[segment_column].shift())
        data['poly_interpolated_data'] = np.nan

        #If measurement segment, interpolate NaNs vis existing measurements and polynomial interpolation
        for i in range(0,len(data[segment_column][shift_points]), 1):  
            start_index = data[segment_column][shift_points].index[i]
            if i == len(data[segment_column][shift_points])-1:
                end_index = len(data)
            else:
                end_index = data[segment_column][shift_points].index[i+1]
            if data[segment_column][start_index] == 0:
                y_interpolated = data.loc[start_index:end_index, data_column].interpolate(method='polynomial', order= self.order_ploynomial_fil) #Polynomial order is defined here (currently: 2)
                y_interpolated = y_interpolated.ffill()  #Last elem in segment is NaN as not enough neighboring values for interpolation, so forward-fill previous value
                data.loc[start_index:end_index,'poly_interpolated_data'] = y_interpolated
                
                #Plotting for visual analysis
                if end_index - start_index > 1000:
                    self.helper.plot_two_df_same_axis(data[time_column][start_index:start_index+1000], data[data_column][start_index:start_index+1000],'Water Level', 'Water Level (corrected)', data['poly_interpolated_data'][start_index:start_index+1000], 'Timestamp ', 'Interpolated Water Level', f'Interpolated data Graph{i}- Measured water level vs Interpolated values (start)')
                    self.helper.plot_two_df_same_axis(data[time_column][end_index-500:end_index], data[data_column][end_index-500:end_index],'Water Level', 'Water Level (corrected)', data['poly_interpolated_data'][end_index-500:end_index], 'Timestamp ', 'Interpolated Water Level', f'Interpolated data Graph{i}- Measured water level vs Interpolated values (end)')

                self.helper.plot_two_df_same_axis(data[time_column][start_index:end_index], data[data_column][start_index:end_index],'Water Level', 'Water Level (corrected)', data['poly_interpolated_data'][start_index:end_index], 'Timestamp ', 'Interpolated Water Level', f'Interpolated data Graph{i}- Measured water level vs Interpolated values')
        
        return data
    

    def polynomial_fitted_data_column(self, data, data_column, time_column, segment_column, filled_data):
        """
        In measurement segments, fit a polynomial curve over existing measurements. This genereates a constinuous curve without sudden jumps or missing values. A continuous ts is needed for some QCs later on.
        
        Input:
        -Main dataframe [df]
        -Column name with measurements to analyse [str]
        -Column name for timestamp [str]
        -Column name for segments [str]
        -Column name for polynomial interpolated series (in pervious method) [str]
        """

        #Get start and end point of each segment
        shift_points = (data[segment_column] != data[segment_column].shift())
        #Spline length in min
        winsize = self.splinelength
        data['poly_fitted_data'] = np.nan

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
                for start in range(start_index, end_index, winsize):
                    if (start + winsize*2) > end_index:
                        end = end_index
                        if start - 200 < start_index:
                            start = start_index
                        else:
                            start = start - 200
                    elif start != start_index:
                        end = start + winsize + 100
                        start = start - 200
                    else:
                        end = start + winsize + 100
                    x_window = data.loc[start:end,time_column]
                    x_numeric = (x_window - x_window.min()).dt.total_seconds() / 60  # Convert to minutes
                    y_window = data.loc[start:end, filled_data]

                    coefficients = np.polyfit(x_numeric, y_window, self.polydegree)

                    # Create a polynomial series based on the coefficients and timeseries
                    y_fit = np.polyval(coefficients, x_numeric)
                    if end == end_index:
                        data.loc[start+200:end,'poly_fitted_data'] = y_fit[200:]
                        break
                    elif start == start_index:
                        data.loc[start:end-100,'poly_fitted_data'] = y_fit[:-100]
                    else:
                        data.loc[start+200:end-100,'poly_fitted_data'] = y_fit[200:-100]

                #Plot for visual analysis
                self.helper.plot_two_df_same_axis(data[time_column][start_index:end_index], data[data_column][start_index:end_index],'Water Level', 'Water Level (corrected)', data['poly_fitted_data'][start_index:end_index], 'Timestamp ', 'Polynomial fitted data Water Level', f'Polynomial fitted graph{start_index}')
            
        return data
    

    def spline_fitted_measurement_column(self, data, data_column, time_column, segment_column):
        """
        In measurement segments, fit a polynomial curve over existing measurements. This genereates a constinuous curve without sudden jumps or missing values. A continuous ts is needed for some QCs later on.
        
        Input:
        -Main dataframe [df]
        -Column name with measurements to analyse [str]
        -Column name for timestamp [str]
        -Column name for segments [str]
        """

        interp_values = np.zeros_like(data[data_column])  # Initialize array for results
        
        #Get start and end point of each segment
        shift_points = (data[segment_column] != data[segment_column].shift())
        #Window length in min
        winsize = self.splinelength

        #For measurement segment: polynomial fit (6th degree) to the polynomial interpolated measurement series
        for i in range(0,len(data[segment_column][shift_points]), 1):  
            start_index = data[segment_column][shift_points].index[i]
            if i == len(data[segment_column][shift_points])-1:
                end_index = len(data)
            else:
                end_index = data[segment_column][shift_points].index[i+1]
            #Fit measurement segments with data
            if data[segment_column][start_index] == 0:
                #Fit spline to various segments by looping over the series in windows defined by self.splinelength
                for start in range(start_index, end_index, winsize):
                    #For last spline period in a segment, fit spline with code here
                    if (start_index + (2*winsize)) > end_index:
                        end = end_index
                        x_window = data.index[start:end]
                        y_window = data[data_column][start:end]

                        # Fit a spline to the masked array, skipping NaNs
                        # Mask out the NaNs
                        x_known = x_window[~np.isnan(y_window)]  # x values where y is not NaN
                        y_known = y_window[~np.isnan(y_window)]  # Corresponding y values (non-NaN)
                        
                        #Only fit a spine to a series, if series consist of 1 more measurement than the spline degree
                        if np.sum(~np.isnan(y_window)) < (self.splinedegree+1):
                            #Do not fit a spline then. (This can be assumed for now as it is barely the case.)
                            break

                        spline = UnivariateSpline(x_known, y_known, s=1, k = self.splinedegree)
                                    
                        # Evaluate the spline at all original `x` points (incl. NaN's)
                        fitted_y = spline(x_window)
                                    
                        #Set interpolated values on comparison ts
                        interp_values[start:end] = fitted_y
                        break
                    else:
                        end =start + winsize
                    #For all spline segments in a measurement segments, fit the corresponding spline below
                    x_window = data.index[start:end]
                    y_window = data[data_column][start:end]

                    # Fit a spline to the masked array, skipping NaNs
                    # Mask out the NaNs
                    x_known = x_window[~np.isnan(y_window)]  # x values where y is not NaN
                    y_known = y_window[~np.isnan(y_window)]  # Corresponding y values (non-NaN)

                    #Only fit a spine to a series, if series consist of 1 more measurement than the spline degree
                    if np.sum(~np.isnan(y_window)) < (self.splinedegree+1):
                        #Do not fit a spline then. (This can be assumed for now as it is barely the case.)
                        pass
                    else:
                        #Fit spline to spline segment
                        spline = UnivariateSpline(x_known, y_known, s=1, k = self.splinedegree)
                                    
                        # Evaluate the spline at all original `x` points (incl. NaN's)
                        fitted_y = spline(x_window)
                                    
                        #Set interpolated values on comparison ts
                        interp_values[start:end] = fitted_y

                #Plot for visual analysis
                self.helper.plot_two_df_same_axis(data[time_column][start_index:end_index], data[data_column][start_index:end_index],'Water Level', 'Water Level (corrected)', interp_values[start_index:end_index], 'Timestamp ', 'Interpolated Water Level', f'Spline fitted Graph{start}- Measured water level vs spline values')
                
        data['spline_fitted_data'] = interp_values

        return data
    

    def compare_filled_measurements(self, data, time_column, segment_column):
        """
        Generates graphs to compare the different fitted or interpolated series.

        Input:
        -Main dataframe [df]
        -Column name for timestamp [str]
        -Column name for segments [str]
        """
        #Get different segments
        shift_points = (data[segment_column] != data[segment_column].shift())

        #Loop over different segments
        #If values in segment, then print the segment and make zoomed plots if big segment
        for i in range(0,len(data[segment_column][shift_points]), 1):  
            start_index = data[segment_column][shift_points].index[i]
            if i == len(data[segment_column][shift_points])-1:
                end_index = len(data)
            else:
                end_index = data[segment_column][shift_points].index[i+1]
            if data[segment_column][start_index] == 0:
                
                if end_index - start_index > 1000:
                    self.helper.plot_two_df_same_axis(data[time_column][start_index:start_index+1000], data['spline_fitted_data'][start_index:start_index+1000],'Water Level', 'Spline fitted data',data['poly_interpolated_data'][start_index:start_index+1000], 'Timestamp ', 'Poly Interpolated Water Level', f'Interpolated data Graph{i}- Spline fitted vs polynomial interpolated values (start)')
                    self.helper.plot_two_df_same_axis(data[time_column][end_index-500:end_index], data['spline_fitted_data'][end_index-500:end_index],'Water Level', 'Spline fitted data', data['poly_interpolated_data'][end_index-500:end_index], 'Timestamp ', 'Poly Interpolated Water Level', f'Interpolated data Graph{i}- Spline fitted vs polynomial interpolated values (end)')
                    self.helper.plot_two_df_same_axis(data[time_column][start_index:start_index+1000], data['spline_fitted_data'][start_index:start_index+1000],'Water Level', 'Spline fitted data',data['poly_fitted_data'][start_index:start_index+1000], 'Timestamp ', 'Poly fitted Water Level', f'Interpolated data Graph{i}- Spline vs polynomial fitted values (start)')
                    self.helper.plot_two_df_same_axis(data[time_column][end_index-500:end_index], data['spline_fitted_data'][end_index-500:end_index],'Water Level', 'Spline fitted data', data['poly_fitted_data'][end_index-500:end_index], 'Timestamp ', 'Poly fitted Water Level', f'Interpolated data Graph{i}- Spline vs polynomial fitted values (end)')

                self.helper.plot_two_df_same_axis(data[time_column][start_index:end_index], data['spline_fitted_data'][start_index:end_index],'Water Level', 'Spline fitted data', data['poly_interpolated_data'][start_index:end_index], 'Timestamp ', 'Poly Interpolated Water Level', f'Interpolated data Graph{i}- Spline fitted vs polynomial interpolated values')
                self.helper.plot_two_df_same_axis(data[time_column][start_index:end_index], data['spline_fitted_data'][start_index:end_index],'Water Level', 'Spline fitted data', data['poly_fitted_data'][start_index:end_index], 'Timestamp ', 'Poly Fitted Water Level', f'Interpolated data Graph{i}- Spline vs polynomial fitted values')
