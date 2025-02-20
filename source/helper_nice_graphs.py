###################################################
## Written by frb for GronSL project (2024-2025) ##
## Contains methods for more advanced graphs     ##
###################################################

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

from datetime import datetime

pio.renderers.default = "browser"

class GraphMaker():
        
    def set_output_folder(self, folder_path):
        self.folder_path = os.path.join(folder_path,'0_final graphs')

        #generate output folder for graphs and other docs
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

    def set_station(self, station):
        self.station = station

    def run(self, df, time_column, data_column):

        self.get_relevant_period(df, time_column, data_column)
        self.make_ts_plot(time_column, data_column)

        print('hey')
    
    def get_relevant_period(self, df, time_column, data_column):
       
        if self.station == 'Ittoqqortoormiit':
            self.extract_period(df, time_column, data_column, '2011', '03', '03', '01', '17')
            #self.extract_period(df, time_column, data_column, '2011', '05', '05', '01', '07')
            #self.extract_period(df, time_column, data_column, '2012', '06', '07', '15', '03')
            #self.extract_period(df, time_column, data_column, '2013', '08', '11', '10', '20')
            #self.extract_period(df, time_column, data_column, '2015', '07', '07', '01', '05')
            #self.extract_period(df, time_column, data_column, '2017', '03', '05', '10', '25')  
            #self.extract_period(df, time_column, data_column, '2019', '01', '01', '16', '23')      
            #self.extract_period(df, time_column, data_column, '2019', '05', '05', '11', '26') 
            #self.extract_period(df, time_column, data_column, '2019', '06', '06', '01', '20')
            #self.extract_period(df, time_column, data_column, '2020', '12', '12', '22', '31')
            #self.extract_period(df, time_column, data_column, '2022', '03', '03', '14', '18')
            #self.extract_period(df, time_column, data_column, '2022', '06', '07', '25', '05')
            #self.extract_period(df, time_column, data_column, '2023', '03', '04', '30', '05')
            #self.extract_period(df, time_column, data_column, '2023', '06', '07', '07', '05')
            #self.extract_period(df, time_column, data_column, '2023', '10', '10', '01', '10')
            #self.extract_period(df, time_column, data_column, '2023', '11', '11', '13', '30')
            #self.extract_period(df, time_column, data_column, '2023', '12', '12', '25', '31')
        elif self.station == 'Qaqortoq':
            self.extract_period(df, time_column, data_column, '2006', '02', '03', '01', '10')
            #self.extract_period(df, time_column, data_column, '2008', '06', '07', '20', '05')
            #self.extract_period(df, time_column, data_column, '2008', '10', '10', '01', '20')
            #self.extract_period(df, time_column, data_column, '2011', '11', '11', '15', '24')
            #self.extract_period(df, time_column, data_column, '2014', '01', '01', '22', '30')
            #self.extract_period(df, time_column, data_column, '2015', '07', '07', '02', '23')  
            #self.extract_period(df, time_column, data_column, '2018', '05', '05', '03', '20')
            #self.extract_period(df, time_column, data_column, '2022', '10', '10', '01', '20')
            #self.extract_period(df, time_column, data_column, '2023', '10', '10', '20', '30')
            #self.extract_period(df, time_column, data_column, '2023', '07', '07', '01', '10')
            #self.extract_period(df, time_column, data_column, '2024', '06', '07', '25', '10')
        elif self.station == 'Nuuk':
            self.extract_period(df, time_column, data_column, '2014', '11', '11', '20', '22')
            #self.extract_period(df, time_column, data_column, '2014', '12', '12', '27', '29')
            #self.extract_period(df, time_column, data_column, '2016', '09', '10', '01', '25') 
            #self.extract_period(df, time_column, data_column, '2017', '04', '04', '15', '25') 
            #self.extract_period(df, time_column, data_column, '2020', '11', '11', '15', '30') 
            #self.extract_period(df, time_column, data_column, '2020', '12', '12', '15', '30')
            #self.extract_period(df, time_column, data_column, '2022', '02', '02', '01', '10') 
        elif self.station == 'Nuuk1':
            self.extract_period(df, time_column, data_column, '2022', '11', '12', '27', '12')
            #self.extract_period(df, time_column, data_column, '2023', '04', '04', '10', '25') 
            #self.extract_period(df, time_column, data_column, '2023', '11', '11', '15', '25') 
            #self.extract_period(df, time_column, data_column, '2024', '02', '02', '15', '20') 
        elif self.station == 'Pituffik':
            self.extract_period(df, time_column, data_column, '2007', '07', '08', '22', '05')
            #self.extract_period(df, time_column, data_column, '2007', '09', '09', '09', '21')
            #self.extract_period(df, time_column, data_column, '2008', '08', '08', '10', '20')        
            #self.extract_period(df, time_column, data_column, '2016', '02', '02', '01', '29')
            #self.extract_period(df, time_column, data_column, '2016', '03', '03', '01', '30')
            #self.extract_period(df, time_column, data_column, '2017', '11', '11', '01', '05')
            #self.extract_period(df, time_column, data_column, '2019', '01', '01', '13', '19')
            #self.extract_period(df, time_column, data_column, '2020', '09', '09', '01', '30')
            #self.extract_period(df, time_column, data_column, '2022', '09', '09', '13', '25')
            #self.extract_period(df, time_column, data_column, '2023', '09', '09', '01', '15')
            #self.extract_period(df, time_column, data_column, '2023', '12', '12', '10', '20')
            #self.extract_period(df, time_column, data_column, '2024', '03', '03', '01', '10')
            #self.extract_period(df, time_column, data_column, '2024', '05', '05', '20', '30')
            #self.extract_period(df, time_column, data_column, '2024', '10', '10', '25', '31')
        elif self.station == 'Upernavik1':
            self.extract_period(df, time_column, data_column, '2023', '08', '08', '12', '12', '03', '22')
            #self.extract_period(df, time_column, data_column, '2023', '08', '08', '15', '31')
        elif self.station == 'Upernavik2':   
            self.extract_period(df, time_column, data_column, '2024', '09', '09', '15', '30')
            #self.extract_period(df, time_column, data_column, '2024', '10', '10', '15', '31')       
        else:
            return 

    def extract_period(self, data, time_column, data_column, year, start_month, end_month, start_day='1', end_day='31', start_hour='00', end_hour='23'):
        """
        Extract relevant periods based on inputs to a new shorter dataframe. 
        This new dataframe is saved as csv with only relevant columns and also plotted for visual analysis.

        Input:
        -Main dataframe [df]
        -Column name for timestamp [str]
        -Column name for relevant measurement series [str]
        -Year [str] (periods cannot go between years!)
        -Start month (possible: 01-12) [str]
        -End month (possible: 01-12) [str]
        -Start day (by default 1, but can be overwritten) (possible: 01-31) [str]
        -End day (by default 31, but can be overwritten) (possible: 01-31) [str]
        -Start hour (by default 00, but can be overwritten) (possible: 00-22) [str]
        -End hour (by default 23, but can be overwritten) (possible: 01-23) [str]
        """

        start_date = datetime(int(year), int(start_month), int(start_day), int(start_hour))
        end_date = datetime(int(year), int(end_month), int(end_day), int(end_hour))
        print(start_date, end_date)
        self.relev_df = data[(data[time_column] >= start_date) & (data[time_column]<= end_date)]

    def make_ts_plot(self, time_column, data_column):
        print(self.relev_df)

        #Visualization of different spike detection methods
        relev_columns = ['spike_value_statistical', 'cotede_spikes', 'cotede_improved_spikes', 'selene_spikes', 'selene_improved_spikes', 'harmonic_detected_spikes']
        lable_title = ['Implausible change rate', 'Neighboring outlier', 'Neighboring outlier (improved)', 'Spline fit', 'Spline fit (improved)', 'Polynomial offset']
        markers = ["s" , "o" , "v" , "D" , "*", "d"]

        # Generate colors from a single-hue colormap
        color_map = plt.cm.plasma  # Choose a colormap (other good ones: Viridis, Cividis, Plasma)
        colors = [color_map(i /len(relev_columns)) for i in range(len(relev_columns))]
        #colors = ['red','green','blue','cyan','magenta', 'darkorange', 'lime']        
        offset_value = np.zeros(len(self.relev_df))
        title = 'Comparison of Spike Detection Methods'
        plt.figure(figsize=(14, 7), dpi=300)
        plt.plot(self.relev_df[time_column], self.relev_df[data_column], label='Measurements', color='black', marker='o',  markersize=0.5, linestyle='None')
        # Stack markers using offsets
        for i in range(len(relev_columns)):
            mask = self.relev_df[relev_columns[i]]
            scatter_marker = markers[i]
            scatter_colors = colors[i]
            plt.scatter(self.relev_df[time_column][mask], self.relev_df[data_column][mask] + offset_value[mask], color=scatter_colors, marker=scatter_marker, s=25, alpha=0.7, edgecolors='black', linewidth=0.5, label=lable_title[i])
            offset_value[mask] += 0.01

        plt.legend(title="Spike Detection Methods:", fontsize=10, title_fontsize=10, frameon=False).set_title("Spike Detection Methods:", prop={"weight": "bold"}) 
        plt.xlabel('Time', fontsize=10)
        plt.ylabel('Water Level [m]', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder_path,f"{title}- Date: {self.relev_df[time_column].iloc[0]}.png"),  bbox_inches="tight")
        plt.close()
