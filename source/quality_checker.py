#####################################################
## Preprocessing sea level tide gauge data by frb  ##
#####################################################
import os, sys
import numpy as np
import pandas as pd
import json
import random
import matplotlib.pyplot as plt
import builtins



import source.helper_methods as helper
import source.qc_spike as qc_spike
import source.qc_interpolation_detector as qc_interpolated
import source.qc_shifts as qc_shifts
import source.qc_filling_missing_data as qc_fill_data

from sklearn.impute import KNNImputer

class QualityFlagger():
    
    def __init__(self):
        self.missing_meas_value = 999.000
        self.window_constant_value = 3
        self.bound_interquantile = 3.5

        #to fit a spline
        self.splinelength = 14 #hours
        self.splinedegree = 2 #3
        self.helper = helper.HelperMethods()

        #Defines about of NaN needed before creating a new segment
        #Need to depend on timestep, only constant if timestep is also constant
        self.nan_threshold = 375
        self.threshold_unusabel_segment = 1000
        
        #change rate (maximum possible change in 30cm in 10 min)
        self.threshold_max_change = 0.30

        #Define periods between bad data as probably with less then 1 hour of good measurements
        self.probably_good_threshold = 60

    def load_qf_classification(self, json_path):

        # Open and load JSON file containing the quality flag classification
        with open(json_path, 'r') as file:
            config_data = json.load(file)

        self.qc_classes = config_data['qc_flag_classification']
        self.qc_bit_definition = config_data['qc_binary_classification']

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
        altered_ts = self.df_meas[self.measurement_column].apply(lambda x: x if isinstance(x, float) else np.nan)
        self.df_meas['incorrect_format'] = np.where(altered_ts, False, True)
        self.df_meas[self.measurement_column] = altered_ts
        # Count the new NaN values after the transformation -> Where there any invalid characters?
        new_nan_count = self.df_meas[self.measurement_column].isna().sum() - original_nan_count
        print('The measurement series contained',new_nan_count,'invalid data entires.')
        
        #2. Replace the missing values with nan
        self.df_meas['missing_values'] = self.df_meas[self.measurement_column] == self.missing_meas_value
        self.df_meas.loc[self.df_meas[self.measurement_column] == self.missing_meas_value, self.measurement_column] = np.nan
        ratio = (self.df_meas['missing_values'].sum()/len(self.df_meas))*100
        print(f'The measurement series contains {self.df_meas['missing_values'].sum()} missing data entires. This is {ratio}% of the overall dataset.')

        #Plot for visualisation
        self.helper.plot_df(self.df_meas[self.time_column], self.df_meas[self.measurement_column],'Water Level','Timestamp ','Measured water level')
        
        #Plot distribution for analysis
        plt.hist(self.df_meas[self.measurement_column] , bins=3000, edgecolor='black', alpha=0.7)
        plt.title('Distribution of Values')
        plt.xlabel('Water Level')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(self.folder_path,"Distribuiton - WL measurements.png"),  bbox_inches="tight")
        plt.close() 

        #Subsequent line makes the code slow, only enable when needed
        #print('length of ts:', len(self.df_meas))
        #self.helper.zoomable_plot_df(self.df_meas[self.time_column], self.df_meas[self.measurement_column],'Water Level','Timestamp ', 'Measured water level','measured water level')
       
    def run(self):
        """
        Different steps taken in the QC approach
        Importing the data already contains the format check and setting invalid formats to NaN
        """
        #Set relevant column names
        self.adapted_meas_col_name = 'altered'

        #Padding of ts to a homogenous timestep
        df = self.check_global_timestamp(self.df_meas)

        #Tests possible on whole dataset
        #Detect stuck values in ts
        df = self.detect_constant_value(df, self.window_constant_value)
        #Detect outliers in ts
        df = self.remove_stat_outliers(df)

        #Segmentation of ts in empty and filled segments

        #Extract segments and fill them accordingly
        segment_column = 'segments'
        fill_data_qc = qc_fill_data.MissingDataFiller()
        fill_data_qc.set_output_folder(self.folder_path)
        df = fill_data_qc.segmentation_ts(df, self.adapted_meas_col_name, self.time_column, segment_column)

        #Set short measurement periods as not trustworthy periods
        df = self.short_bad_measurement_periods(df, segment_column)

        #fill nans in relevant segments
        df = fill_data_qc.polynomial_fill_data_column(df, self.adapted_meas_col_name, self.time_column, segment_column)
        df = fill_data_qc.spline_fill_measurement_column(df, self.adapted_meas_col_name, self.time_column, segment_column)
        fill_data_qc.compare_filled_measurements(df, self.adapted_meas_col_name, self.time_column, segment_column)

        #Detect interpolated values
        interpolated_qc = qc_interpolated.Interpolation_Detector()
        interpolated_qc.set_output_folder(self.folder_path)
        df = interpolated_qc.run_interpolation_detection(df, self.adapted_meas_col_name, self.time_column)

        #Detect shifts & deshift values
        shift_detection = qc_shifts.ShiftDetector()
        shift_detection.set_output_folder(self.folder_path)
        df_long = shift_detection.detect_shifts_statistical(df, 'poly_interpolated_data', self.time_column, self.measurement_column)
        #test = shift_detection.detect_shifts_ruptures(test, self.adapted_meas_col_name, 'poly_interpolated_data')
        #df_long = shift_detection.unsupervised_outlier_detection(df, self.adapted_meas_col_name, 'poly_interpolated_data', self.time_column)


        #Detect spike values
        spike_detection = qc_spike.SpikeDetector()
        spike_detection.set_output_folder(self.folder_path)
        #spike_detection.filled_ts_tight(self.filled_WL_measurements)
        df_long_dup = df_long.copy()
        df_long_dup_2 = df_long.copy()
        df_long_test = df_long.copy()

        #Detect outliers based on plausible change rate within a certain timestamp f.e. 30 cm in 10 min
        df = self.remove_spikes_plausible_change_rate(df, segment_column)

        df_long_2 = spike_detection.remove_spikes_cotede(df_long, self.adapted_meas_col_name, self.time_column, self.measurement_column)
        df_long = spike_detection.remove_spikes_cotede_improved(df_long_dup, self.adapted_meas_col_name, self.time_column, self.measurement_column)
        #df_long_dup_2 = spike_detection.selene_spike_detection(df_long_dup_2, self.adapted_meas_col_name, self.time_column, self.measurement_column, 'filled_values')
        df_long_dup = spike_detection.remove_spikes_ml(df_long_dup_2, self.adapted_meas_col_name, self.time_column, self.measurement_column)

        #Probbly good data
        #Mark all data as probably good data if it is only a short measurement period be
        #df_long = self.probably_goog_data(df_long)

    def check_global_timestamp(self, data):

        #Generate a new ts in 1 min timestamp
        start_time = data[self.time_column].iloc[0]
        end_time = data[self.time_column].iloc[-1]
        ts_full = pd.date_range(start= start_time, end= end_time, freq='min').to_frame(name=self.time_column).reset_index(drop=True)

        #Merge df based on timestamp and plot the outcome
        df_meas_long = pd.merge(ts_full, data, on=self.time_column, how = 'outer')
        df_meas_long['resolution'] = df_meas_long['resolution'].bfill()
        #Get mask for the new introduced NaNs based on missing_values mask fom before (new NaNs = True)
        df_meas_long['missing_values_padding'] = np.where(df_meas_long['missing_values'].isna(), True, False)
        df_meas_long.loc[df_meas_long['missing_values'] == True, 'missing_values_padding'] = False

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
            self.helper.plot_df(df_meas_long[self.time_column][true_indices[-1]-30:true_indices[-1]+50], df_meas_long[self.measurement_column][true_indices[-1]-30:true_indices[-1]+50],'Water Level','Timestamp ','Constant period in TS (2)')

        # Mask the constant values and add it as a column
        df_meas_long[self.adapted_meas_col_name] = np.where(constant_mask, np.nan, df_meas_long[self.measurement_column])
        df_meas_long['stuck_value'] = constant_mask

        #print details on the constant value check
        if constant_mask.any():
            ratio = (constant_mask.sum()/len(df_meas_long))*100
            print(f"There are {constant_mask.sum()} constant values in this timeseries. This is {ratio}% of the overall dataset.")
        
        return df_meas_long
    
    def remove_stat_outliers(self, df_meas_long):

        # Quantile Detection for large range outliers
        # Calculate the interquartile range (IQR)
        Q1 = df_meas_long[self.adapted_meas_col_name].quantile(0.25)
        Q3 = df_meas_long[self.adapted_meas_col_name].quantile(0.75)
        IQR = Q3 - Q1

        # Define the lower and upper bounds for outlier detection
        # normally it is lower_bound = Q1 - 1.5 * IQR and upper_bound = Q3 + 1.5 * IQR
        # We can set them to even larger range to only tidal analysis, coz outlier detection is after tidal analysis 
        lower_bound = Q1 - self.bound_interquantile * IQR
        upper_bound = Q3 + self.bound_interquantile * IQR

        # Detect outliers and set them to NaN specifically for the self.measurement_column column
        outlier_mask = (df_meas_long[self.adapted_meas_col_name] < lower_bound) | (df_meas_long[self.adapted_meas_col_name] > upper_bound)
        # Mask the outliers
        df_meas_long[self.adapted_meas_col_name] = np.where(outlier_mask, np.nan, df_meas_long[self.adapted_meas_col_name])
        #Add outlier mask as a column
        df_meas_long['outlier'] = outlier_mask

        # Get indices where the mask is True (as check that approach works)
        if outlier_mask.any():
            true_indices = outlier_mask[outlier_mask].index
            self.helper.plot_df(df_meas_long[self.time_column][true_indices[0]-10000:true_indices[0]+10000], df_meas_long[self.measurement_column][true_indices[0]-10000:true_indices[0]+10000],'Water Level','Timestamp ','Outlier period in TS')
            self.helper.plot_df(df_meas_long[self.time_column][true_indices[0]-10000:true_indices[0]+10000], df_meas_long[self.adapted_meas_col_name][true_indices[0]-10000:true_indices[0]+10000],'Water Level','Timestamp ','Outlier period in TS (corrected)')
            self.helper.plot_df(df_meas_long[self.time_column], df_meas_long[self.adapted_meas_col_name],'Water Level','Timestamp ','Measured water level wo outliers in 1 min timestamp')
          
            ratio = (outlier_mask.sum()/len(df_meas_long))*100
            print(f"There are {outlier_mask.sum()} outliers in this timeseries. This is {ratio}% of the overall dataset.")
        
        #Plot distribution for analysis after outlier removal
        plt.hist(df_meas_long[self.adapted_meas_col_name] , bins=300, edgecolor='black', alpha=0.7)
        plt.title('Distribution of Values')
        plt.xlabel('Water Level')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(self.folder_path,"Distribuiton - WL measurements (no outliers).png"),  bbox_inches="tight")
        plt.close() 

        return df_meas_long

    def short_bad_measurement_periods(self, data, segment_column):
        
        #for segment
        data['short_bad_measurement_series'] = False
        shift_points = (data[segment_column] != data[segment_column].shift())
        z = 0

        for i in range(0,len(data[segment_column][shift_points]), 1):  
            start_index = data[segment_column][shift_points].index[i]
            if i == len(data[segment_column][shift_points])-1:
                end_index = len(data)
            else:
                end_index = data[segment_column][shift_points].index[i+1]
            if data[segment_column][start_index] == 0:
                self.helper.plot_df(data[self.time_column][start_index:end_index], data[self.adapted_meas_col_name][start_index:end_index],'Water Level', 'Timestamp',f'Segment graph {start_index}')
                #test_df = data[(data[time_column].dt.year == 2007) & (data[time_column].dt.month == 9)]
                #self.helper.plot_two_df_same_axis(test_df[time_column],test_df[data_column],'Water Level', 'Water Level', test_df[segment_column], 'Timestamp', 'Segment',f'Test Graph 0')
                print(f'Segment is {end_index - start_index} entries long.')
                print(f'This bad period sarts with index {start_index}.')
                print(np.sum(~np.isnan(data[self.adapted_meas_col_name][start_index:end_index]))/len(data[self.adapted_meas_col_name][start_index:end_index]))
                if end_index - start_index < self.threshold_unusabel_segment:
                    data.loc[start_index:end_index, 'short_bad_measurement_series'] = True
                    data.loc[start_index:end_index, self.adapted_meas_col_name] = np.nan
                    data.loc[start_index:end_index, segment_column] = 1
                    z += 1
                    self.helper.plot_df(data[self.time_column][start_index-2000:end_index+2000], data[self.measurement_column][start_index-2000:end_index+2000],'Water Level', 'Timestamp', f'Bad and short periods (monthly) - Graph {i}')
                    self.helper.plot_df(data[self.time_column][start_index-2000:end_index+2000], data[self.adapted_meas_col_name][start_index-2000:end_index+2000],'Water Level', 'Timestamp', f'Bad and short periods (monthly)- Cleaned - Graph{i}')
                elif np.sum(~np.isnan(data[self.adapted_meas_col_name][start_index:end_index]))/len(data[self.adapted_meas_col_name][start_index:end_index]) < 0.05:
                    data.loc[start_index:end_index, 'short_bad_measurement_series'] = True
                    data.loc[start_index:end_index, self.adapted_meas_col_name] = np.nan
                    data.loc[start_index:end_index, segment_column] = 1
                    z += 1
                    self.helper.plot_df(data[self.time_column][start_index-2000:end_index+2000], data[self.measurement_column][start_index-2000:end_index+2000],'Water Level', 'Timestamp', f'Bad and empty periods (monthly) - Graph {i}')
                    self.helper.plot_df(data[self.time_column][start_index-2000:end_index+2000], data[self.adapted_meas_col_name][start_index-2000:end_index+2000],'Water Level', 'Timestamp', f'Bad and empty periods (monthly)- Cleaned - Graph{i}')
        print(f"There are {z} bad segments in this timeseries.")

        #for segment
        shift_points = (data[segment_column] != data[segment_column].shift())
        print(f'Now there are still {len(data[segment_column][shift_points])/2} segments with measurements in this measurement series.')

        return data


    def probably_good_data(self, data):

        boolean_columns = data.select_dtypes(include='bool')
        combined = boolean_columns.any(axis=1)
        data['combined_mask'] = combined

        # Apply the function to the column
        data['probably_good_mask'] = self.mask_fewer_than_x_consecutive_false(data['combined_mask'])
        
    # Vectorized solution to create the mask
    def mask_fewer_than_x_consecutive_false(self, series):
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
        
        # Determine where False segments >= 5 and mask those regions
        for start, end in zip(starts, ends):
            if end - start >= self.probably_good_threshold:
                mask[start:end] = False

        return mask
