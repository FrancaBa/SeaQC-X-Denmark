########################################################################################################
## Written by frb for GronSL project (2024-2025)                                                      ##
## This is the main non-ML script for the QC. From here the different classes and methods are called. ##
########################################################################################################

import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

import source.helper_methods as helper
import source.qc_spike as qc_spike
import source.qc_interpolation_detector as qc_interpolated
import source.qc_shifts as qc_shifts
import source.qc_filling_missing_data as qc_fill_data
import source.qc_marker_probably_good as qc_prob_good
import source.qc_implausible_changes as qc_implausible_change
import source.qc_global_outliers as qc_global_outliers
import source.qc_stuck_values as qc_stuck_values
import source.data_extractor_monthly as extractor
import source.qc_test_results_to_mask as qc_test_results_to_mask

from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.metrics import precision_score

class QualityFlagger():
    
    def __init__(self):

        self.helper = helper.HelperMethods()

        #to fit a spline
        self.splinelength = 14 #hours
        self.splinedegree = 2 #3
        
        #Defines amounts of NaN-values needed before creating a new segment
        #Need to depend on timestep, only constant if timestep is also constant
        self.nan_threshold = 375
        #If segment is short, drop it
        self.threshold_unusabel_segment = 1000
        
        #Empty element to store different texts about QC tests and save as a summary document
        self.information = []

    def set_station(self, station):
        self.station = station
    
    #load config.json file to generate bitmask and flags after QC tests
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
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

        self.helper.set_output_folder(self.folder_path)

    #set input table to correct heading
    def set_column_names(self, time_column, measurement_column, qc_column):
        self.time_column = time_column
        self.measurement_column = measurement_column
        self.qc_column = qc_column

    def set_missing_value_filler(self, missing_meas_value):
        #Dummy value for NaN-values in measurement series
        self.missing_meas_value = missing_meas_value 


    def import_data(self, path, file):
        """
        Importing the data from csv to df. And some first pre-processing:
        -Extracting areas of interest to csv filea. 
        -Fix timestamp column and column names 
        -Get resolution between different measurements
        -Plots for basic understanding
        -Remove invalid characters in measurement period
        -Set missing values to NaN
        already contains the format check and setting invalid formats to NaN

        Input:
        - path to file [str]
        - filename including ending [str]
        """
        if file.endswith(".csv"):
            #Open .csv file and fix the column names
            self.df_meas = pd.read_csv(os.path.join(path,file), sep=",", header=0)
            self.df_meas = self.df_meas[[self.time_column, self.measurement_column]]
            self.df_meas[self.time_column] = pd.to_datetime(self.df_meas[self.time_column], format='%Y%m%d%H%M%S')
        else:
            #Open .dat file and fix the column names
            self.df_meas = pd.read_csv(os.path.join(path,file), sep="\\s+", header=None, names=[self.time_column, self.measurement_column, self.qc_column])
            self.df_meas = self.df_meas[[self.time_column, self.measurement_column]]
            self.df_meas[self.time_column] = pd.to_datetime(self.df_meas[self.time_column], format='%Y%m%d%H%M%S')
        
        #Extract data for manual labelling
        data_extractor = extractor.DataExtractor()
        data_extractor.set_output_folder(self.folder_path, self.station)
        data_extractor.set_missing_value_filler(self.missing_meas_value)
        data_extractor.run(self.df_meas, self.time_column, self.measurement_column)

        #drop seconds
        self.df_meas[self.time_column] = self.df_meas[self.time_column].dt.round('min')

        # Extract the resolution in seconds, minutes, etc.
        # Here, we extract the time difference in minutes
        self.df_meas['time_diff'] = self.df_meas[self.time_column].diff()
        self.df_meas['resolution'] = self.df_meas['time_diff'].dt.total_seconds()/60
        self.df_meas.loc[self.df_meas['resolution'] > 3600, 'resolution'] = np.nan

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
        self.information.append(['The measurement series contained',new_nan_count,'invalid data entires.'])
        
        #2. Replace the missing values with nan
        self.df_meas['missing_values'] = self.df_meas[self.measurement_column] == self.missing_meas_value
        self.df_meas.loc[self.df_meas[self.measurement_column] == self.missing_meas_value, self.measurement_column] = np.nan
        ratio = (self.df_meas['missing_values'].sum()/len(self.df_meas))*100
        print(f'The measurement series contains {self.df_meas['missing_values'].sum()} missing data entires. This is {ratio}% of the overall dataset.')
        self.information.append([f'The measurement series contains {self.df_meas['missing_values'].sum()} missing data entires. This is {ratio}% of the overall dataset.'])
                                
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

    """
    The methods above are to open and preprocess the need information and data for the quality check.
    The run-method below is the core of the QC work. It calls the different QC steps, converts the masks to a large bitmasks and assigns quality flags.
    """
    def run(self):
        #Set relevant column names
        self.adapted_meas_col_name = 'altered'
        
        #Padding of ts to a homogenous timestep
        df = self.set_global_timestamp(self.df_meas)
        df[self.adapted_meas_col_name] = df[self.measurement_column]
        
        #Runs the different steps in the QC algorithm
        df = self.run_qc(df)

        #Convert information of passed and failed tests to flags
        conversion_bitmask = qc_test_results_to_mask.QualityMasking()
        conversion_bitmask.set_output_folder(self.folder_path)
        conversion_bitmask.set_station(self.station)
        conversion_bitmask.set_flags(self.qc_classes)
        conversion_bitmask.set_bitmask(self.qc_bit_definition)
        df = conversion_bitmask.convert_boolean_to_bitmask(df)
        df = conversion_bitmask.convert_bitmask_to_flags(df)
        conversion_bitmask.save_flagged_series(df, self.measurement_column, self.time_column, self.df_meas)

        #Check what unsupervised ML would do
        #self.unsupervised_outlier_detection(df, self.measurement_column, 'poly_interpolated_data', self.time_column, self.segment_column)

        #Save information from QC tests to txt file
        self.save_to_txt()


    def run_qc(self, df):
        """
        As main QC method, it calls the different steps taken in the QC approach. See commented text.
        """
        #Detect stuck values in ts
        stuck_values = qc_stuck_values.StuckValuesDetector()
        stuck_values.set_output_folder(self.folder_path)
        df = stuck_values.run(df, self.time_column, self.adapted_meas_col_name, self.information)

        #Detect global outliers in ts
        global_outliers = qc_global_outliers.OutlierRemover()
        global_outliers.set_output_folder(self.folder_path)
        df = global_outliers.run(df, self.adapted_meas_col_name, self.time_column, self.measurement_column, self.information)

        #Detect interpolated values
        interpolated_qc = qc_interpolated.Interpolation_Detector()
        interpolated_qc.set_output_folder(self.folder_path)
        df = interpolated_qc.run_interpolation_detection(df, self.adapted_meas_col_name, self.time_column, self.information)
       
        #Segmentation of ts in empty and measurement segments
        #Extract measurement segments and fill them accordingly
        self.segment_column = 'segments'
        fill_data_qc = qc_fill_data.MissingDataFiller()
        fill_data_qc.set_output_folder(self.folder_path)
        df = fill_data_qc.segmentation_ts(df, self.adapted_meas_col_name, self.segment_column, self.information)

        #Set short measurement periods between missing data periods as not trustworthy periods
        df = self.short_bad_measurement_periods(df, self.segment_column)

        #Add continuous helper columns
        df = fill_data_qc.polynomial_fill_data_column(df, self.adapted_meas_col_name, self.time_column, self.segment_column)
        df = fill_data_qc.polynomial_fitted_data_column(df, self.adapted_meas_col_name, self.time_column, self.segment_column, 'poly_interpolated_data')
        df = fill_data_qc.spline_fitted_measurement_column(df, self.adapted_meas_col_name, self.time_column, self.segment_column)
        fill_data_qc.compare_filled_measurements(df, self.time_column, self.segment_column)

        #Detect implausible change rate over period
        implausible_change = qc_implausible_change.ImplausibleChangeDetector()
        implausible_change.set_output_folder(self.folder_path)
        df = implausible_change.run(df, self.adapted_meas_col_name, self.time_column, self.information)

        #Detect spike values
        spike_detection = qc_spike.SpikeDetector()
        spike_detection.set_output_folder(self.folder_path)
        self.information.append(['Various Spike detection approaches and their outcomes:'])
        df = spike_detection.detect_spikes_statistical(df, 'poly_interpolated_data', self.time_column, self.adapted_meas_col_name, self.information)
        df = spike_detection.remove_spikes_cotede(df, self.adapted_meas_col_name, self.time_column, self.information)
        df = spike_detection.remove_spikes_cotede_improved(df, self.adapted_meas_col_name, self.time_column, self.information)
        df = spike_detection.selene_spike_detection(df, self.adapted_meas_col_name, self.time_column, 'spline_fitted_data', self.information)
        df = spike_detection.remove_spikes_harmonic(df, 'poly_fitted_data', self.adapted_meas_col_name, self.time_column, self.information)
        #Not really ML and doesn't improve anything (Thus, it is not used!)
        df = spike_detection.remove_spikes_ml(df, 'poly_fitted_data', self.adapted_meas_col_name, self.time_column, self.information)
        
        #Detect shifts & deshift values
        shift_detection = qc_shifts.ShiftDetector()
        shift_detection.set_output_folder(self.folder_path)
        df = shift_detection.detect_shifts_ruptures(df, self.adapted_meas_col_name, self.time_column, 'poly_interpolated_data', self.segment_column, self.information)
        df = shift_detection.detect_shifts_statistical(df, 'poly_interpolated_data', self.time_column, self.adapted_meas_col_name, self.segment_column, self.information)

        #Probably good data
        #Mark all data as probably good data if it is only a short measurement period between bad data
        prob_good = qc_prob_good.ProbablyGoodDataFlagger()
        prob_good.set_output_folder(self.folder_path)
        df = prob_good.run(df, self.adapted_meas_col_name, self.time_column, self.measurement_column, self.information)

        return df


    def set_global_timestamp(self, data):
        """
        Create a constant 1-min timeseries and align the measurements to it. This will introduce a lot of new NaNs for periods without 1-min resolution.

        Input:
        - main dataframe [pandas df]
        """
        #Generate a new ts in 1 min timestamp
        start_time = data[self.time_column].iloc[0]
        end_time = data[self.time_column].iloc[-1]
        ts_full = pd.date_range(start= start_time, end= end_time, freq='min').to_frame(name=self.time_column).reset_index(drop=True)

        #Merge df based on timestamp
        df_meas_long = pd.merge(ts_full, data, on=self.time_column, how = 'outer')
        df_meas_long['resolution'] = df_meas_long['resolution'].bfill()
        #Get mask for the new introduced NaNs based on missing_values mask fom before (new NaNs = True)
        df_meas_long['missing_values'] = df_meas_long['missing_values'].fillna(False).infer_objects(copy=False)
        df_meas_long['incorrect_format'] = df_meas_long['incorrect_format'].fillna(False).infer_objects(copy=False)

        #This information is not needed for now
        #df_meas_long['missing_values_padding'] = np.where(df_meas_long['missing_values'].isna(), True, False)
        #df_meas_long.loc[df_meas_long['missing_values'] == True, 'missing_values_padding'] = False

        print('The new ts is',len(df_meas_long),'entries long.')
        self.information.append(['The new ts is',len(df_meas_long),'entries long.'])

        #plot with new 1-min ts for visual analysis
        self.helper.plot_df(df_meas_long[self.time_column], df_meas_long[self.measurement_column],'Water Level','Timestamp ','Measured water level in 1 min timestamp')
        self.helper.plot_df(df_meas_long[self.time_column][33300:33400], df_meas_long[self.measurement_column][33300:33400],'Water Level','Timestamp ','Measured water level in 1 min timestamp (zoom)')
        #self.helper.zoomable_plot_df(df_meas_long[self.time_column][:33600], df_meas_long_filled[self.measurement_column][:33600],'Water Level','Timestamp ', 'Measured water level time','measured water level time')

        return df_meas_long


    def short_bad_measurement_periods(self, data, segment_column):
        """
        Check if segments are very short or contain a lot of NaN values. If yes, drop those segments as bad segments.

        Input:
        -Main dataframe [pandas df]
        -Column name of segmentation information [str]
        """
        
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
        self.information.append([f"There are {z} bad segments in this timeseries."])

        #for segment
        shift_points = (data[segment_column] != data[segment_column].shift())
        print(f'Now there are still {len(data[segment_column][shift_points])/2} segments with measurements in this measurement series.')
        self.information.append([f'Now there are still {len(data[segment_column][shift_points])/2} segments with measurements in this measurement series.'])

        return data


    def save_to_txt(self):
        """
        Save the print statements from each QC test to a common txt-file. This can be used later on to compare between stations and set-ups.
        """

        # Filepath to save the text file
        filename = f"QC_summary_{self.station}.txt"
        file_path = os.path.join(self.folder_path, filename)

        # Save the list to a .txt file
        with open(file_path, "w") as file:
            for list in self.information:
                 # Write each sublist as a row
                file.write(" ".join(map(str, list)) + "\n")
                # Add an empty line after each row
                file.write("\n")

        print(f"QC test statements have been saved to {file_path}")


    def unsupervised_outlier_detection(self, df, data_column_name, interpolated_data_colum, time_column, segment_column):
        """
        Run a simple unsupervised ML algorithm to see how it performs in grouping the measurments.
        """
        shift_points = (df[segment_column] != df[segment_column].shift())

        for i in range(0,len(df[segment_column][shift_points]), 1):
            start_index = df[segment_column][shift_points].index[i]
            if i == len(df[segment_column][shift_points])-1:
                end_index = len(df)
            else:
                end_index = df[segment_column][shift_points].index[i+1]
            if df[segment_column][start_index] == 0:
                relev_df = df[start_index:end_index]
                true_anomalies = 1 #still needs to be defined
                
                # Prepare data for Isolation Forest
                # Isolation Forest expects input in a 2D array form
                X = df[interpolated_data_colum][start_index:end_index].values.reshape(-1, 1)

                # Grid search for optimal contamination
                contamination_values = np.linspace(0.01, 0.1, 10)  # Test contamination levels from 1% to 10%
                best_contamination = None
                best_precision = 0

                for contamination in contamination_values:
                    isolation_forest = IsolationForest(contamination=contamination, random_state=42)
                    anomaly_score = isolation_forest.fit_predict(X)
                    
                    # Convert anomaly scores to binary (1 for anomalies, 0 for normal)
                    predicted_anomalies = (anomaly_score == -1).astype(int)
                    
                    # Compute precision score (you can also use recall or F1-score)
                    precision = precision_score(true_anomalies, predicted_anomalies, zero_division=0)
                    
                    if precision > best_precision:
                        best_precision = precision
                        best_contamination = contamination


                # Fit Isolation Forest
                model = IsolationForest(contamination=best_contamination, random_state=42)  # Adjust contamination as needed
                anomaly = model.fit_predict(X)

                print('hey')
                # Get indices of non-NaN values in measurement column
                #non_nan_indices = relev_df[~relev_df[data_column_name].isna()].index.to_numpy()
                # Compute the absolute difference between target indices and non-NaN indices using broadcasting
                #distances = np.abs(np.array(anomaly)[:, np.newaxis] - non_nan_indices)
                # Find the index of the minimum distance for each target index
                #anomaly_entries = np.unique(non_nan_indices[np.argmin(distances, axis=1)])

                # Identify anomalies
                anomalies = df[anomaly == -1]

                # Cluster the anomalies using DBSCAN
                dbscan = DBSCAN(eps=1, min_samples=2)
                clusters = dbscan.fit_predict(anomalies[['values']])
                anomalies['cluster'] = clusters

                # Add cluster information back to the main DataFrame
                ts = ts.merge(anomalies[['time', 'cluster']], on='time', how='left')

                # Plot the results
                plt.figure(figsize=(12, 6))
                plt.plot(df[time_column], df[interpolated_data_colum][start_index:end_index], label='Time Series')
                plt.scatter(anomalies[time_column], anomalies[interpolated_data_colum], color='red', label='Anomalies', zorder=5)
                plt.title('Anomaly Detection with Isolation Forest')
                plt.xlabel('Time')
                plt.ylabel('Value')
                plt.legend()
                plt.show()