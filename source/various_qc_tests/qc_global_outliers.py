##########################################################
## Written by frb for GronSL project (2024-2025)        ##
## This script marks global outliers via moving window. ##
##########################################################

import os
import numpy as np
import matplotlib.pyplot as plt
import builtins
import random

import source.helper_methods as helper

class OutlierRemover(): 

    def __init__(self):
        self.helper = helper.HelperMethods()

        #to assess global outliers
        self.bound_interquantile = None

    def set_output_folder(self, folder_path):
        self.folder_path = os.path.join(folder_path,'global outliers')

        #generate output folder for graphs and other docs
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

        self.helper.set_output_folder(self.folder_path)

    #Load relevant parameters for this QC test from conig.json
    def set_parameters(self, params, suffix):
        #to assess global outliers
        self.bound_interquantile = params[f'bound_interquantile{suffix}']
        self.threshold_zscore = params[f'threshold_zscore{suffix}']
        self.window = params[f'rolling_window_outliers']

    def run(self, df_meas_long, adapted_meas_col_name, time_column, measurement_column, information, original_length, suffix):
        """
        Based on z-score detect global outliers. This is a backup-test and not fully implemented. The tests outcomes are not considered when exporting to bitmasks/flags.
      
        Input: 
        -data: Main dataframe [df]
        -adapted_meas_col_name: Column name for measurement series [str]
        -time_column: Column name for timestamp [str]
        -measurement_column: original data series [str]
        -Information list where QC report is collected [lst]
        -Length of original measurement series [int]
        -suffix: ending for columns and graphs in order to run in different modes [str]
        """

        #If your data is normally distributed around zero, using the Z-score method might be better:
        rolling_mean = df_meas_long[adapted_meas_col_name].rolling(window=self.window, center=True, min_periods=48).mean()
        rolling_std = df_meas_long[adapted_meas_col_name].rolling(window=self.window, center=True, min_periods=48).std()
        z_scores = np.abs((df_meas_long[adapted_meas_col_name] - rolling_mean) / rolling_std)

        # Detect outliers and set them to NaN specifically for the self.measurement_column column
        outlier_mask = z_scores >  self.threshold_zscore
        # Mask the outliers
        df_meas_long['test'] = np.where(outlier_mask, df_meas_long[adapted_meas_col_name], np.nan)
        #Add outlier mask as a column
        df_meas_long[f'outlier{suffix}'] = outlier_mask

        # Get indices where the mask is True (as check that approach works)
        if outlier_mask.any():
            true_indices = outlier_mask[outlier_mask].index
            max_range = builtins.min(31, len(true_indices))
            for i in range(1, max_range):
                x = random.choice(range(0,len(true_indices))) 
                min = builtins.max(0,(true_indices[x]-1000))
                max = builtins.min(len(df_meas_long), min+2000)
                self.helper.plot_two_df_same_axis(df_meas_long[time_column][min:max], df_meas_long['test'][min:max], 'Water Level [m]', 'Global outlier', df_meas_long[adapted_meas_col_name][min:max], 'Timestamp', 'Measured Water Level', f'Zscore- Global Outlier{suffix}-{i}')
                

        ratio = (outlier_mask.sum()/original_length)*100
        print(f"There are {outlier_mask.sum()} global outliers in this timeseries according to the z-score over a moving window. This is {ratio}% of the overall dataset.")
        information.append([f"There are {outlier_mask.sum()} global outliers in this timeseries according to the z-score over a moving window. This is {ratio}% of the overall dataset."])

        #Plot distribution for analysis after outlier removal
        plt.hist(df_meas_long[adapted_meas_col_name] , bins=300, edgecolor='black', alpha=0.7)
        plt.title('Distribution of Values')
        plt.xlabel('Water Level')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(self.folder_path,f"Zscore - Distribuiton - WL measurements (no outliers)-{suffix}.png"),  bbox_inches="tight")
        plt.close() 

        del df_meas_long['test']

        #Remove global outliers, so they don't impact the assessment later on
        df_meas_long[adapted_meas_col_name] = np.where(outlier_mask, np.nan, df_meas_long[adapted_meas_col_name])
        
        return df_meas_long
    
    def interquantile_run(self, df_meas_long, adapted_meas_col_name, time_column, measurement_column, information, original_length, suffix):
        """
        Based on interquantile approach detect global outlier, but not a reliable method for spike detection.
      
        Input: 
        -data: Main dataframe [df]
        -adapted_meas_col_name: Column name for measurement series [str]
        -time_column: Column name for timestamp [str]
        -measurement_column: original data series [str]
        -Information list where QC report is collected [lst]
        -Length of original measurement series [int]
        -suffix: ending for columns and graphs in order to run in different modes [str]
        """

        # Quantile Detection for large range outliers
        # Calculate the interquartile range (IQR)
        Q1 = df_meas_long[adapted_meas_col_name].quantile(0.25)
        Q3 = df_meas_long[adapted_meas_col_name].quantile(0.75)
        IQR = Q3 - Q1

        # Define the lower and upper bounds for outlier detection
        # normally it is lower_bound = Q1 - 1.5 * IQR and upper_bound = Q3 + 3.5 * IQR
        # We can set them to even larger range to only tidal analysis, coz outlier detection is after tidal analysis 
        lower_bound = Q1 - self.bound_interquantile * IQR
        upper_bound = Q3 + self.bound_interquantile * IQR

        # Detect outliers and set them to NaN specifically for the self.measurement_column column
        outlier_mask = (df_meas_long[adapted_meas_col_name] < lower_bound) | (df_meas_long[adapted_meas_col_name] > upper_bound)
        # Mask the outliers
        df_meas_long[adapted_meas_col_name] = np.where(outlier_mask, np.nan, df_meas_long[adapted_meas_col_name])
        #Add outlier mask as a column
        df_meas_long[f'outlier{suffix}'] = outlier_mask

        # Get indices where the mask is True (as check that approach works)
        if outlier_mask.any():
            true_indices = outlier_mask[outlier_mask].index
            max_range = builtins.min(31, len(true_indices))
            for i in range(1, max_range):      
                min = builtins.max(0,(true_indices[i]-10000))
                max = builtins.min(len(df_meas_long), min+20000)
                self.helper.plot_df(df_meas_long[time_column][min:max], df_meas_long[measurement_column][min:max],'Water Level','Timestamp ', f'Outlier period in TS {i}{suffix}')
                self.helper.plot_df(df_meas_long[time_column][min:max], df_meas_long[adapted_meas_col_name][min:max],'Water Level','Timestamp ', f'Outlier period in TS (corrected) {i}{suffix}')
                
            
        ratio = (outlier_mask.sum()/original_length)*100
        print(f"There are {outlier_mask.sum()} outliers in this time series. This is {ratio}% of the overall dataset.")
        information.append([f"There are {outlier_mask.sum()} outliers in this time series. This is {ratio}% of the overall dataset."])
        self.helper.plot_df(df_meas_long[time_column], df_meas_long[adapted_meas_col_name],'Water Level','Timestamp ', f'Measured water level wo outliers in 1 min timestamp{suffix}')
        
        #Plot distribution for analysis after outlier removal
        plt.hist(df_meas_long[adapted_meas_col_name] , bins=300, edgecolor='black', alpha=0.7)
        plt.title('Distribution of Values')
        plt.xlabel('Water Level')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(self.folder_path,f"Distribuiton - WL measurements (no outliers)-{suffix}.png"),  bbox_inches="tight")
        plt.close() 

        return df_meas_long