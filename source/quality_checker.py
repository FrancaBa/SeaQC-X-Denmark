#####################################################
## Preprocessing sea level tide gauge data by frb  ##
#####################################################
import os, sys
import pdb
import numpy as np
import pandas as pd
import json

import source.helper_methods as helper
import source.qc_spike as qc_spike
import source.qc_interpolation_detector as qc_interpolated
import source.qc_shifts as qc_shifts

class QualityFlagger():
    
    def __init__(self):
        self.missing_meas_value = 999.000
        self.window_constant_value = 10
        
        self.helper = helper.HelperMethods()

    def load_qf_classification(self, json_path):

        # Open and load JSON file containing the quality flag classification
        with open(json_path, 'r') as file:
            config_data = json.load(file)

        self.qc_classes = config_data['qc_flag_classification']

    #Create output folder to save results
    def set_output_folder(self, folder_path):

        self.folder_path = folder_path 

        #generate output folder for graphs and other docs
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.helper.set_output_folder(folder_path)

    #set input table to correct heading
    def set_column_names(self, time_column, measurement_column, qc_column):
        self.time_column = time_column
        self.measurement_column = measurement_column
        self.qc_column = qc_column

    def import_data(self, path, file):
        #Open file and fix the column names
        self.df_meas = pd.read_csv(os.path.join(path,file), sep="\s+", header=None, names=[self.time_column, self.measurement_column, self.qc_column])
        self.df_meas = self.df_meas[[self.time_column, self.measurement_column]]
        self.df_meas[self.time_column] = pd.to_datetime(self.df_meas[self.time_column], format='%Y%m%d%H%M%S')
        
        #drop seconds
        self.df_meas[self.time_column] = self.df_meas[self.time_column].dt.round('min')

        # Extract the resolution in seconds, minutes, etc.
        # Here, we extract the time difference in minutes
        self.df_meas['time_diff'] = self.df_meas[self.time_column].diff()
        self.df_meas['resolution'] = self.df_meas['time_diff'].dt.total_seconds()/60
        self.df_meas.loc[self.df_meas['resolution'] > 3600, 'resolution'] = np.nan

        # Save the DataFrame to a text file (tab-separated)
        self.df_meas.to_csv(os.path.join(self.folder_path,'time_series_with_resolution.txt'), sep='\t', index=False)
        #Plot the original ts to get an understanding
        self.helper.plot_df(self.df_meas[self.time_column][-2000:-1000], self.df_meas['resolution'][-2000:-1000],'step size','Timestamp ','Time resolution (zoomed)')
        self.helper.plot_df(self.df_meas[self.time_column][33300:33320], self.df_meas['resolution'][33300:33320],'step size','Timestamp ','Time resolution (zoomed 2)')
        self.helper.plot_df(self.df_meas[self.time_column], self.df_meas['resolution'],'step size','Timestamp ','Time resolution')

        #Initial screening of measurements:
        #1. Invalid characters: Set non-float elements to NaN
        # Count the original NaN values
        original_nan_count = self.df_meas[self.measurement_column].isna().sum()
        self.df_meas[self.measurement_column] = self.df_meas[self.measurement_column].apply(lambda x: x if isinstance(x, float) else np.nan)
        # Count the new NaN values after the transformation -> Where there any invalid characters?
        new_nan_count = self.df_meas[self.measurement_column].isna().sum() - original_nan_count
        print('The measurement series contained',new_nan_count,'invalid data entires.')
        
        #2. Replace the missing values with nan
        self.df_meas.loc[self.df_meas[self.measurement_column] == self.missing_meas_value, self.measurement_column] = np.nan

        #Plot for visualisation
        self.helper.plot_df(self.df_meas[self.time_column], self.df_meas[self.measurement_column],'Water Level','Timestamp ','Measured water level')
        
        #Subsequent line makes the code slow, only enable when needed
        #print('length of ts:', len(self.df_meas))
        #self.helper.zoomable_plot_df(self.df_meas[self.time_column], self.df_meas[self.measurement_column],'Water Level','Timestamp ', 'Measured water level','measured water level')
    
    def run(self):
        """
        Different steps taken in the QC approach
        """
        #Set relevant column names
        self.adapted_meas_col_name = 'altered'
        self.quality_column_name = 'quality_flag'

        df_long = self.check_timestamp()
        df_long = self.detect_constant_value(df_long, self.window_constant_value)
        df_long = self.remove_stat_outliers(df_long)

        #Detect spike values
        spike_detection = qc_spike.SpikeDetector()
        spike_detection.set_output_folder(self.folder_path)
        #df_long = spike_detection.remove_spikes_ml(df_long[self.adapted_meas_col_name][-1000000:])
        df_long_dup = df_long.copy()
        df_long_2 = spike_detection.remove_spikes_cotede(df_long, self.adapted_meas_col_name, self.quality_column_name, self.time_column, self.measurement_column, self.qc_classes)
        df_long = spike_detection.remove_spikes_cotede_improved(df_long_dup, self.adapted_meas_col_name, self.quality_column_name, self.time_column, self.measurement_column, self.qc_classes)
        df_long = spike_detection.selene_spike_detection(df_long_dup, self.adapted_meas_col_name, self.quality_column_name, self.time_column, self.measurement_column, self.qc_classes)

        #Detect shifts & deshift values
        #shift_detection = qc_shifts.ShiftDetector()
        #df_long = shift_detection.detect_shifts(df_long, self.adapted_meas_col_name, self.quality_column_name)

        #Detect interpolated values
        interpolated_qc = qc_interpolated.Interpolation_Detector()
        interpolated_qc.set_output_folder(self.folder_path)
        df_long = interpolated_qc.run_interpolation_detection(df_long, self.adapted_meas_col_name, self.time_column, self.qc_classes, self.quality_column_name)

    def check_timestamp(self):

        #Generate a new ts in 1 min timestamp
        start_time = self.df_meas[self.time_column].iloc[0]
        end_time = self.df_meas[self.time_column].iloc[-1]
        ts_full = pd.date_range(start= start_time, end= end_time, freq='min').to_frame(name=self.time_column).reset_index(drop=True)

        #Merge df based on timestamp and plot the outcome
        df_meas_long = pd.merge(ts_full, self.df_meas, on=self.time_column, how = 'outer')
        df_meas_long['resolution'] = df_meas_long['resolution'].bfill()
        df_meas_long[self.quality_column_name] = np.where(np.isnan(df_meas_long[self.measurement_column]), self.qc_classes['missing_data'], self.qc_classes['good_data'])

        #Check specific lines
        print('The new ts is',len(df_meas_long),'entries long.')
        print(df_meas_long[df_meas_long[self.time_column] == '2005-10-24 03:00:00'])
        #print(df_meas_long.index[df_meas_long[self.time_column] == '2005-10-24 03:00:00'].to_list())
        print(df_meas_long[33300:33320])
        print(df_meas_long[:20])

        #plot with new 1-min ts
        self.helper.plot_df(df_meas_long[self.time_column], df_meas_long[self.measurement_column],'Water Level','Timestamp ','Measured water level in 1 min timestamp')
        self.helper.plot_df(df_meas_long[self.time_column][33300:33400], df_meas_long[self.measurement_column][33300:33400],'Water Level','Timestamp ','Measured water level in 1 min timestamp (zoom)')
        #self.helper.zoomable_plot_df(df_meas_long[self.time_column][:33600], df_meas_long_filled[self.measurement_column][:33600],'Water Level','Timestamp ', 'Measured water level time','measured water level time')

        return df_meas_long
    
    def detect_constant_value(self, df_meas_long, window_constant_value):

        # Check if the value is constant over a window of 'self.window_constant_value' min (counts only non-nan)
        # Step 1: Identify where the values change, ignoring NaNs
        # Step 2: Assign a unique group ID for each sequence of the same value
        # Step 3: Count the size of each group, and check if each run is at least 'window_constant_value' entries long
        # Step 4: Create the mask based on the length of consecutive identical values
        is_new_value = (df_meas_long[self.measurement_column] != df_meas_long[self.measurement_column].shift()) | df_meas_long[self.measurement_column].isna()
        groups = is_new_value.cumsum()
        group_sizes = df_meas_long.groupby(groups)[self.measurement_column].transform('size')
        constant_mask = (group_sizes >= window_constant_value) & df_meas_long[self.measurement_column].notna()

        # Get indices where the mask is True (as check that approach works)
        if constant_mask.any():
            true_indices = constant_mask[constant_mask].index
            self.helper.plot_df(df_meas_long[self.time_column][true_indices[0]-30:true_indices[0]+50], df_meas_long[self.measurement_column][true_indices[0]-30:true_indices[0]+50],'Water Level','Timestamp ','Constant period in TS')


        # Mask the constant values
        df_meas_long[self.adapted_meas_col_name] = np.where(constant_mask, np.nan, df_meas_long[self.measurement_column])
        df_meas_long[self.quality_column_name] = np.where(df_meas_long[self.quality_column_name] == self.qc_classes['missing_data'], self.qc_classes['missing_data'],  # Keep it as missing_data
                                        np.where(constant_mask, self.qc_classes['bad_data'], self.qc_classes['good_data']))  # Else set to good or bad based on constant_mask
        self.helper.plot_df(df_meas_long[self.time_column], df_meas_long[self.quality_column_name],'Quality Flags','Timestamp ','Applied quality flags')

        #print details on the constant value check
        if self.qc_classes['bad_data'] in df_meas_long[self.quality_column_name].unique():
            ratio = (len(df_meas_long[df_meas_long[self.quality_column_name] == self.qc_classes['bad_data']])/len(df_meas_long))*100
            print(f"There are {len(df_meas_long[df_meas_long[self.quality_column_name] == self.qc_classes['bad_data']])} constant values in this timeseries. This is {ratio}% of the overall dataset.")
        
        return df_meas_long
    
    def remove_stat_outliers(self, df_meas_long):

        # Quantile Detection for large range outliers
        # Calculate the interquartile range (IQR)
        Q1 = df_meas_long.quantile(0.25)
        Q3 = df_meas_long.quantile(0.75)
        IQR = Q3 - Q1

        # Define the lower and upper bounds for outlier detection
        # normally it is lower_bound = Q1 - 1.5 * IQR and upper_bound = Q3 + 1.5 * IQR
        # We can set them to even larger range to only tidal analysis, coz outlier detection is after tidal analysis 
        lower_bound = Q1 - 2.5 * IQR
        upper_bound = Q3 + 2.5 * IQR

        # Detect outliers and set them to NaN specifically for the self.measurement_column column
        outlier_mask = (df_meas_long[self.adapted_meas_col_name] < lower_bound[self.adapted_meas_col_name]) | (df_meas_long[self.adapted_meas_col_name] > upper_bound[self.adapted_meas_col_name])
        # Mask the outliers
        df_meas_long[self.adapted_meas_col_name] = np.where(outlier_mask, np.nan, df_meas_long[self.adapted_meas_col_name])
        #Flag & remove the outlier
        df_meas_long.loc[(df_meas_long[self.quality_column_name] == self.qc_classes['good_data']) & (outlier_mask), self.quality_column_name] = self.qc_classes['outliers']

        # Get indices where the mask is True (as check that approach works)
        if outlier_mask.any():
            true_indices = outlier_mask[outlier_mask].index
            self.helper.plot_df(df_meas_long[self.time_column][true_indices[0]-10000:true_indices[0]+10000], df_meas_long[self.measurement_column][true_indices[0]-10000:true_indices[0]+10000],'Water Level','Timestamp ','Outlier period in TS')
            self.helper.plot_df(df_meas_long[self.time_column][true_indices[0]-10000:true_indices[0]+10000], df_meas_long[self.adapted_meas_col_name][true_indices[0]-10000:true_indices[0]+10000],'Water Level','Timestamp ','Outlier period in TS (corrected)')
            self.helper.plot_df(df_meas_long[self.time_column], df_meas_long[self.adapted_meas_col_name],'Water Level','Timestamp ','Measured water level wo outliers in 1 min timestamp')
          
        if self.qc_classes['outliers'] in df_meas_long[self.quality_column_name].unique():
            ratio = (len(df_meas_long[df_meas_long[self.quality_column_name] == self.qc_classes['outliers']])/len(df_meas_long))*100
            print(f"There are {len(df_meas_long[df_meas_long[self.quality_column_name] == self.qc_classes['outliers']])} outliers in this timeseries. This is {ratio}% of the overall dataset.")

        return df_meas_long