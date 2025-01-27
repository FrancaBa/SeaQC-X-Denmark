import os, sys
import datetime
import numpy as np
import pandas as pd
import utide

import source.helper_methods as helper

class TidalSignalGenerator():

    def __init__(self):
        self.helper = helper.HelperMethods()
        self.coef = []

    def set_parameter_path(self, path):
        self.path = path

    def set_station(self, station):
        self.station = station

    def run(self, df, time_column, information):

        #coef = dict_keys(['name', 'aux', 'nR', 'nNR', 'nI', 'weights', 'A', 'g', 'mean', 'g_ci', 'A_ci', 'diagn', 'PE', 'SNR'])
        self.coef = self.load_coef(self.path)
        #define time span where tidal signal is needed
        time_array =  df[time_column].values
        
        #Generate tidal signal
        self.generate_tidal_signal(time_array, coef)

        #Insight for tidal analysis
        print('Tidal signal has been successfully created for this timeseries. The used constituents are',coef,'.')
        information.append(['Tidal signal has been successfully created for this timeseries. The used constituents are',coef,'.'])

    def load_coef(self, path):
        #Read file with saved tidal constituents and open corresponding one
        with open(path,'r') as file:
            for line in file:
                #-- Read until header begins --
                if 'begin' in line[0:5]:
                    header=line.split()
                    stno=header[1]
                    name=header[3]
                    ChD= header[6] 
                    LAT= header[7] 
                    print(name)
                    print(self.station)
                    if name in self.station:
                        print('hey')
                        coef = 1
                    
            if not coef:
                raise Exception('Tidal signal generation script is executed for a measurement station where predefined tidal constituents do not exist. Script needs to be improved before tidal analysis for this station can be made.')
        
        return coef
    
    def generate_tidal_signal(self, time_array, coef):
         # reconstruct function to generate a hindcast or forecast of the tides at the times specified in the time array.
        tide = utide.reconstruct(time_array, coef, verbose=False)