import os, sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utide
import builtins
import random

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
        #Extract latitude for relevant station
        lat = self.get_stations_lat(self.gauge_details_path)

        #For tidal analysis, it needs anomaly data. After tidal analysis, one can add the mean again to get the clean original data.
        #Calculate anomaly
        anomaly = df[measurement_column] - df[measurement_column].mean()
        #Define time span for which tidal signal is needed
        time_array =  df[time_column].values

        #Extract good data measurements based on first QC
        boolean_columns = df.select_dtypes(include='bool')
        del boolean_columns['missing_values']
        df['combined_mask'] = boolean_columns.any(axis=1)
        time_array_good = time_array
        anomaly_good = anomaly

        coef = utide.solve(time_array_good, anomaly_good.values, lat=lat, method="ols", conf_int="MC", trend=False, nodal=True, verbose=False)
       
        #Reconstruct function to generate a tidal signal at the times specified in the time array
        tide = utide.reconstruct(time_array, coef, verbose=False)
        df['tidal_signal'] = tide.h
        df['detided_series'] = df[measurement_column] - tide.h

        #Export tidal signal
        file_name = f"{self.station}-TidalSignal.csv"
        df['tidal_signal'].to_csv(os.path.join(self.folder_path, file_name), index=False)

        #Insight for tidal analysis
        fig, (ax0, ax1, ax2) = plt.subplots(figsize=(17, 5), nrows=3, sharey=True, sharex=True)
        ax0.plot(time_array, anomaly, label="Shifted observations", color="C0")
        ax1.plot(time_array, tide.h, label="Tidal prediction", color="C1")
        ax2.plot(time_array, df['detided_series'], label="Residual", color="C2")
        fig.legend(ncol=3, loc="upper center")
        plt.savefig(os.path.join(self.folder_path,f"Tidal Decomposition-whole period"),  bbox_inches="tight")
        plt.close()

        for i in range(0,31):
            min = builtins.max(0,(random.choice(df.index))-10000)
            max = builtins.min(min + 20000, len(df))
            self.helper.plot_two_df_same_axis(df[time_column][min:max], df[anomaly][min:max],'Water Level', 'Water Level (anomaly)', df['tidal_signal'][min:max], 'Timestamp', 'Tidal signal',f'Measurements vs tide signal - Index: {min}')

        print('Tidal signal has been successfully created for this timeseries. The used constituents are',coef,'.')
        information.append(['Tidal signal has been successfully created for this timeseries. The used constituents are',coef,'.'])
    
    def get_stations_lat(self, gauge_details_path):
        #Read file with saved tide gauge details and load corresponding one
        with open(gauge_details_path,'r') as file:
            for line in file:
                #-- Read until header begins --
                if 'begin' in line[0:5]:
                    header=line.split()
                    #stno=header[1]
                    name=header[3] 
                    if name in self.station:
                        lat= float(header[4])
 
        if not lat:
            raise Exception('Tidal signal generation script is executed for a measurement station where needed gauge details do not exist. Script needs to be improved before tidal analysis for this station can be made.')
        
        return lat