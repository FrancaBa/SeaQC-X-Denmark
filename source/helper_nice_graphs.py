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

pio.renderers.default = "browser"

import source.data_extractor_monthly as extractor

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

        print('hey')
    
    def get_relevant_period(self, df, time_column, data_column):

        data_extractor = extractor.DataExtractor()
        data_extractor.set_output_folder(self.folder_path, self.station)
        
        if self.station == 'Ittoqqortoormiit':
            data_extractor.extract_period(df, time_column, data_column, '2011', '03', '03', '01', '17')
            data_extractor.extract_period(df, time_column, data_column, '2011', '05', '05', '01', '07')
            data_extractor.extract_period(df, time_column, data_column, '2012', '06', '07', '15', '03')
            data_extractor.extract_period(df, time_column, data_column, '2013', '08', '11', '10', '20')
            data_extractor.extract_period(df, time_column, data_column, '2015', '07', '07', '01', '05')
            data_extractor.extract_period(df, time_column, data_column, '2017', '03', '05', '10', '25')  
            data_extractor.extract_period(df, time_column, data_column, '2019', '01', '01', '16', '23')      
            data_extractor.extract_period(df, time_column, data_column, '2019', '05', '05', '11', '26') 
            data_extractor.extract_period(df, time_column, data_column, '2019', '06', '06', '01', '20')
            data_extractor.extract_period(df, time_column, data_column, '2020', '12', '12', '22', '31')
            data_extractor.extract_period(df, time_column, data_column, '2022', '03', '03', '14', '18')
            data_extractor.extract_period(df, time_column, data_column, '2022', '06', '07', '25', '05')
            data_extractor.extract_period(df, time_column, data_column, '2023', '03', '04', '30', '05')
            data_extractor.extract_period(df, time_column, data_column, '2023', '06', '07', '07', '05')
            data_extractor.extract_period(df, time_column, data_column, '2023', '10', '10', '01', '10')
            data_extractor.extract_period(df, time_column, data_column, '2023', '11', '11', '13', '30')
            data_extractor.extract_period(df, time_column, data_column, '2023', '12', '12', '25', '31')
        elif self.station == 'Qaqortoq':
            data_extractor.extract_period(df, time_column, data_column, '2006', '02', '03', '01', '10')
            data_extractor.extract_period(df, time_column, data_column, '2008', '06', '07', '20', '05')
            data_extractor.extract_period(df, time_column, data_column, '2008', '10', '10', '01', '20')
            data_extractor.extract_period(df, time_column, data_column, '2011', '11', '11', '15', '24')
            data_extractor.extract_period(df, time_column, data_column, '2014', '01', '01', '22', '30')
            data_extractor.extract_period(df, time_column, data_column, '2015', '07', '07', '02', '23')  
            data_extractor.extract_period(df, time_column, data_column, '2018', '05', '05', '03', '20')
            data_extractor.extract_period(df, time_column, data_column, '2022', '10', '10', '01', '20')
            data_extractor.extract_period(df, time_column, data_column, '2023', '10', '10', '20', '30')
            data_extractor.extract_period(df, time_column, data_column, '2023', '07', '07', '01', '10')
            data_extractor.extract_period(df, time_column, data_column, '2024', '06', '07', '25', '10')
        elif self.station == 'Nuuk':
            data_extractor.extract_period(df, time_column, data_column, '2014', '11', '11', '20', '22')
            data_extractor.extract_period(df, time_column, data_column, '2014', '12', '12', '27', '29')
            data_extractor.extract_period(df, time_column, data_column, '2016', '09', '10', '01', '25') 
            data_extractor.extract_period(df, time_column, data_column, '2017', '04', '04', '15', '25') 
            data_extractor.extract_period(df, time_column, data_column, '2020', '11', '11', '15', '30') 
            data_extractor.extract_period(df, time_column, data_column, '2020', '12', '12', '15', '30')
            data_extractor.extract_period(df, time_column, data_column, '2022', '02', '02', '01', '10') 
        elif self.station == 'Nuuk1':
            data_extractor.extract_period(df, time_column, data_column, '2022', '11', '12', '27', '12')
            data_extractor.extract_period(df, time_column, data_column, '2023', '04', '04', '10', '25') 
            data_extractor.extract_period(df, time_column, data_column, '2023', '11', '11', '15', '25') 
            data_extractor.extract_period(df, time_column, data_column, '2024', '02', '02', '15', '20') 
        elif self.station == 'Pituffik':
            data_extractor.extract_period(df, time_column, data_column, '2007', '07', '08', '22', '05')
            data_extractor.extract_period(df, time_column, data_column, '2007', '09', '09', '09', '21')
            data_extractor.extract_period(df, time_column, data_column, '2008', '08', '08', '10', '20')        
            data_extractor.extract_period(df, time_column, data_column, '2016', '02', '02', '01', '29')
            data_extractor.extract_period(df, time_column, data_column, '2016', '03', '03', '01', '30')
            data_extractor.extract_period(df, time_column, data_column, '2017', '11', '11', '01', '05')
            data_extractor.extract_period(df, time_column, data_column, '2019', '01', '01', '13', '19')
            data_extractor.extract_period(df, time_column, data_column, '2020', '09', '09', '01', '30')
            data_extractor.extract_period(df, time_column, data_column, '2022', '09', '09', '13', '25')
            data_extractor.extract_period(df, time_column, data_column, '2023', '09', '09', '01', '15')
            data_extractor.extract_period(df, time_column, data_column, '2023', '12', '12', '10', '20')
            data_extractor.extract_period(df, time_column, data_column, '2024', '03', '03', '01', '10')
            data_extractor.extract_period(df, time_column, data_column, '2024', '05', '05', '20', '30')
            data_extractor.extract_period(df, time_column, data_column, '2024', '10', '10', '25', '31')
        elif self.station == 'Upernavik1':
            data_extractor.extract_period(df, time_column, data_column, '2023', '08', '08', '01', '15')
            data_extractor.extract_period(df, time_column, data_column, '2023', '08', '08', '15', '31')
        elif self.station == 'Upernavik2':   
            data_extractor.extract_period(df, time_column, data_column, '2024', '09', '09', '15', '30')
            data_extractor.extract_period(df, time_column, data_column, '2024', '10', '10', '15', '31')       
        else:
            return 
