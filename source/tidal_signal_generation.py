##########################################################################################################################################
## Written by frb for GronSL project (2024-2025)                                                                                        ##
## This script generates a tidal signal for measurement series using UTide and creates a detided sea level series for the measurements. ##
##########################################################################################################################################

import os, sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utide
import builtins
import random
import pickle

import source.helper_methods as helper

class TidalSignalGenerator():

    def __init__(self):
        self.helper = helper.HelperMethods()
        lat = []

    def set_gauge_details_path(self, gauge_details_path):
        self.gauge_details_path = gauge_details_path

    def set_station(self, station):
        self.station = station

    #Create output folder to save results
    def set_output_folder(self, folder_path):
        self.folder_path = os.path.join(folder_path,'tidal decomposition')

        #generate output folder for graphs and other docs
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

        self.helper.set_output_folder(self.folder_path)

    def run(self, df, measurement_column, time_column, information):
        """
        Generating a tidal signal over timeframe of measurement series for corresponding station. Removal of tidal signal to create detided series. 
        Needed steps:
        -Get latitude from station from tides.local 
        -Calculate anomaly of measurement
        -Extract relevant timeseries
        -Extract roughly one year of good timeseries and measurement values to UTide constituent generation
        -Use Utide to reconstruct tidal signal
        -Substract tidal signal from measurement anomaly to get detided series

        Input:
        -Main dataframe [pandas df]
        -Column name of measurement series of interest [str]
        -Column name of time & date information [str]
        -Information list where QC report is collected [lst]
        """
        #Extract latitude for relevant station
        lat = self.get_stations_lat(self.gauge_details_path)

        #For tidal analysis, it needs anomaly data. After tidal analysis, one can add the mean again to get the clean original data.
        #Calculate anomaly
        df['anomaly'] = df[measurement_column] - df[measurement_column].mean()

        #Extract good data measurements based on first QC over 1 year for tidal analysis
        boolean_columns = df.select_dtypes(include='bool')
        #del boolean_columns['missing_values']
        df['combined_mask'] = boolean_columns.any(axis=1)
        good_df = df[~df['combined_mask']]
        # Get the year with the most entries and calibrate tide constitents based on that year
        year_counts = good_df[time_column].dt.year.value_counts()
        most_frequent_year = year_counts.idxmax()
        calibration_df = good_df[good_df[time_column].dt.year == most_frequent_year]
        #Extract needed inputs for U-Tide (resample to 5 min for memory issue)
        calibration_df = calibration_df.set_index(time_column)
        calibration_df_resampled = calibration_df.resample('5min').mean().interpolate()
        time_array_good = calibration_df_resampled.index.values
        anomaly_good = calibration_df_resampled['anomaly']
        #time_array_good = calibration_df.index.values
        #anomaly_good = calibration_df['anomaly']

        coef = utide.solve(time_array_good, anomaly_good.values, lat=lat, method="ols", conf_int="MC", trend=False, nodal=True, verbose=False)
        #coef = utide.solve(time_array_good, anomaly_good.values, lat=lat, method="ols", conf_int="MC", trend=True, nodal=True, verbose=False)
       
        #Reconstruct function to generate a tidal signal at the times specified in the time array
        #Split it in 10 parts to avoid memory issues or run it on server!!
        period = round(len(time_array_good)/10)
        period_tide = []
        for i in range(0, len(df), period):
            period_array = df[time_column].values[i:i+period]
            tide = utide.reconstruct(period_array, coef, verbose=False)
            period_tide.append(tide.h)
        combined_tide = np.concatenate(period_tide)
        df['tidal_signal'] = combined_tide
        #df['detided_series'] = df['anomaly'] - combined_tide
        df['detided_series'] = df[measurement_column] - combined_tide

        #Export tidal signal 6 detided series
        file_name = f"{self.station}-TidalSignal-and-detided series.csv"
        df_export = df[[time_column, 'detided_series', 'anomaly', 'tidal_signal']].copy().dropna()
        df_export.to_csv(os.path.join(self.folder_path, file_name), index=False, header=["Time", "Residual", "Obs_anom", "Tidal_comp"])

        #Insight for tidal analysis
        fig, (ax0, ax1, ax2) = plt.subplots(figsize=(17, 5), nrows=3, sharey=False, sharex=True)
        ax0.plot(df[time_column], df['anomaly'], label="SL Measurements", color="C0", marker='o',  markersize=1)
        ax1.plot(df[time_column], combined_tide, label="Tidal prediction", color="C1", marker='o',  markersize=1)
        ax2.plot(df[time_column], df['detided_series'], label="Residual", color="C2", marker='o',  markersize=1)
        fig.legend(ncol=3, loc="upper center")
        plt.savefig(os.path.join(self.folder_path,f"Tidal Decomposition-whole period"),  bbox_inches="tight")
        plt.close()

        for i in range(0,31):
            min = builtins.max(0,(random.choice(df.index))-10000)
            max = builtins.min(min + 20000, len(df))
            self.helper.plot_two_df_same_axis(df[time_column][min:max], df['anomaly'][min:max],'Water Level', 'Water Level (anomaly)', df['tidal_signal'][min:max], 'Timestamp', 'Tidal signal',f'Measurements vs tide signal - Index: {min}')
            self.helper.plot_df(df[time_column][min:max], df['detided_series'][min:max], 'Detided signal', 'Timestamp', f'Detided sea level - Index: {min}')

        # Export parameters to .pkl
        filename = os.path.join(self.folder_path,f"tidal_components_{self.station}.pkl")
        # Save to file
        with open(filename, "wb") as f:
            pickle.dump(coef, f)
        
        print(f"Parameters exported to {filename}")

        print('Tidal signal has been successfully created for this timeseries.')
        information.append(['Tidal signal has been successfully created for this timeseries.'])

        del df['combined_mask']
        del df['anomaly']

        return df
    
    def get_stations_lat(self, gauge_details_path):
        #Read file with saved tide gauge details and load relevant information (for now: latitude)
        with open(gauge_details_path,'r') as file:
            for line in file:
                #-- Read until header begins --
                if 'begin' in line[0:5]:
                    header=line.split()
                    name=header[3] 
                    if name in self.station:
                        lat= float(header[4])
 
        if not lat:
            raise Exception('Tidal signal generation script is executed for a measurement station where needed gauge details do not exist. Script needs to be improved before tidal analysis for this station can be made.')
                
        return lat