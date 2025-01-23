###################################################
## Written by frb for GronSL project (2024-2025) ##
## This script marks global outliers.            ##
###################################################

import os
import numpy as np
import matplotlib.pyplot as plt

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
    def set_parameters(self, params):
        #to assess global outliers
        self.bound_interquantile = params['bound_interquantile']

    def run(self, df_meas_long, adapted_meas_col_name, time_column, measurement_column, information, original_length):

        # Quantile Detection for large range outliers
        # Calculate the interquartile range (IQR)
        Q1 = df_meas_long[adapted_meas_col_name].quantile(0.25)
        Q3 = df_meas_long[adapted_meas_col_name].quantile(0.75)
        IQR = Q3 - Q1

        # Define the lower and upper bounds for outlier detection
        # normally it is lower_bound = Q1 - 1.5 * IQR and upper_bound = Q3 + 1.5 * IQR
        # We can set them to even larger range to only tidal analysis, coz outlier detection is after tidal analysis 
        lower_bound = Q1 - self.bound_interquantile * IQR
        upper_bound = Q3 + self.bound_interquantile * IQR

        # Detect outliers and set them to NaN specifically for the self.measurement_column column
        outlier_mask = (df_meas_long[adapted_meas_col_name] < lower_bound) | (df_meas_long[adapted_meas_col_name] > upper_bound)
        # Mask the outliers
        df_meas_long[adapted_meas_col_name] = np.where(outlier_mask, np.nan, df_meas_long[adapted_meas_col_name])
        #Add outlier mask as a column
        df_meas_long['outlier'] = outlier_mask

        # Get indices where the mask is True (as check that approach works)
        if outlier_mask.any():
            true_indices = outlier_mask[outlier_mask].index
            self.helper.plot_df(df_meas_long[time_column][true_indices[0]-10000:true_indices[0]+10000], df_meas_long[measurement_column][true_indices[0]-10000:true_indices[0]+10000],'Water Level','Timestamp ','Outlier period in TS')
            self.helper.plot_df(df_meas_long[time_column][true_indices[0]-10000:true_indices[0]+10000], df_meas_long[adapted_meas_col_name][true_indices[0]-10000:true_indices[0]+10000],'Water Level','Timestamp ','Outlier period in TS (corrected)')
            self.helper.plot_df(df_meas_long[time_column], df_meas_long[adapted_meas_col_name],'Water Level','Timestamp ','Measured water level wo outliers in 1 min timestamp')
          
        ratio = (outlier_mask.sum()/original_length)*100
        print(f"There are {outlier_mask.sum()} outliers in this timeseries. This is {ratio}% of the overall dataset.")
        information.append([f"There are {outlier_mask.sum()} outliers in this timeseries. This is {ratio}% of the overall dataset."])

        #Plot distribution for analysis after outlier removal
        plt.hist(df_meas_long[adapted_meas_col_name] , bins=300, edgecolor='black', alpha=0.7)
        plt.title('Distribution of Values')
        plt.xlabel('Water Level')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(self.folder_path,"Distribuiton - WL measurements (no outliers).png"),  bbox_inches="tight")
        plt.close() 

        return df_meas_long