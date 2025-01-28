######################################################################################################################################
## Extract interesting measurement periods by date (written by frb for GronSL project (2024-2025))                                  ##
## This script is used to generate csv files which are the baseline for manual labelling in Trainset (https://trainset.geocene.com/)##
######################################################################################################################################

import os, sys
import numpy as np
import pandas as pd

from datetime import datetime, timedelta

import source.helper_methods as helper

class DataExtractor():
    
    def __init__(self):
        self.missing_meas_value = None
        self.list_relev_section = []
        self.helper = helper.HelperMethods()
    
    #Define relevant station and where to save output to
    def set_output_folder(self, folder, station):
        self.folder_path = os.path.join(folder,'interesting ts and their graphs')

        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

        self.helper.set_output_folder(self.folder_path)
        self.station = station

    def set_missing_value_filler(self, missing_meas_value):
        #Dummy value for NaN-values in measurement series
        self.missing_meas_value = missing_meas_value 

    #Main method 
    def run(self, df, time_column, data_column, suffix = ''):
        """
        Extract relevant periods to csv for manual quality check based on station. 
        After extracting the periods, merge all relevant periods to a long csv file.
        Rearrange timestamp to put a max timestep of 10 days between relevant periods (makes manual labelling more convenient).

        Input:
        -Main dataframe [df]
        -Column name for timestamp [str]
        -Column name for relevant measurement series [str]
        """
        #to differentiate between detided and tided data
        self.suffix = suffix

        filtered_df = df.copy()
        filtered_df['label'] = None
        filtered_df['series'] = 'seriesA'

        if self.station == 'Ittoqqortoormiit':
            self.extract_period(filtered_df, time_column, data_column, '2011', '03', '03', '01', '17')
            self.extract_period(filtered_df, time_column, data_column, '2011', '05', '05', '01', '07')
            self.extract_period(filtered_df, time_column, data_column, '2012', '06', '07', '15', '03')
            self.extract_period(filtered_df, time_column, data_column, '2013', '08', '11', '10', '20')
            self.extract_period(filtered_df, time_column, data_column, '2015', '07', '07', '01', '05')
            self.extract_period(filtered_df, time_column, data_column, '2017', '03', '05', '10', '25')  
            self.extract_period(filtered_df, time_column, data_column, '2019', '01', '01', '16', '23')      
            self.extract_period(filtered_df, time_column, data_column, '2019', '05', '05', '11', '26') 
            self.extract_period(filtered_df, time_column, data_column, '2019', '06', '06', '01', '20')
            self.extract_period(filtered_df, time_column, data_column, '2020', '12', '12', '22', '31')
            self.extract_period(filtered_df, time_column, data_column, '2022', '03', '03', '14', '18')
            self.extract_period(filtered_df, time_column, data_column, '2022', '06', '07', '25', '05')
            self.extract_period(filtered_df, time_column, data_column, '2023', '03', '04', '30', '05')
            self.extract_period(filtered_df, time_column, data_column, '2023', '06', '07', '07', '05')
            self.extract_period(filtered_df, time_column, data_column, '2023', '10', '10', '01', '10')
            self.extract_period(filtered_df, time_column, data_column, '2023', '11', '11', '13', '30')
            self.extract_period(filtered_df, time_column, data_column, '2023', '12', '12', '25', '31')
        elif self.station == 'Qaqortoq':
            self.extract_period(filtered_df, time_column, data_column, '2006', '02', '03', '01', '10')
            self.extract_period(filtered_df, time_column, data_column, '2008', '06', '07', '20', '05')
            self.extract_period(filtered_df, time_column, data_column, '2008', '10', '10', '01', '20')
            self.extract_period(filtered_df, time_column, data_column, '2011', '11', '11', '15', '24')
            self.extract_period(filtered_df, time_column, data_column, '2014', '01', '01', '22', '30')
            self.extract_period(filtered_df, time_column, data_column, '2015', '07', '07', '02', '23')  
            self.extract_period(filtered_df, time_column, data_column, '2018', '05', '05', '03', '20')
            self.extract_period(filtered_df, time_column, data_column, '2022', '10', '10', '01', '20')
            self.extract_period(filtered_df, time_column, data_column, '2023', '10', '10', '20', '30')
            self.extract_period(filtered_df, time_column, data_column, '2023', '07', '07', '01', '10')
            self.extract_period(filtered_df, time_column, data_column, '2024', '06', '07', '25', '10')
        elif self.station == 'Nuuk':
            self.extract_period(filtered_df, time_column, data_column, '2014', '11', '11', '20', '22')
            self.extract_period(filtered_df, time_column, data_column, '2014', '12', '12', '27', '29')
            self.extract_period(filtered_df, time_column, data_column, '2016', '09', '10', '01', '25') 
            self.extract_period(filtered_df, time_column, data_column, '2017', '04', '04', '15', '25') 
            self.extract_period(filtered_df, time_column, data_column, '2020', '11', '11', '15', '30') 
            self.extract_period(filtered_df, time_column, data_column, '2020', '12', '12', '15', '30')
            self.extract_period(filtered_df, time_column, data_column, '2022', '02', '02', '01', '10') 
        elif self.station == 'Nuuk1':
            self.extract_period(filtered_df, time_column, data_column, '2022', '11', '12', '27', '12')
            self.extract_period(filtered_df, time_column, data_column, '2023', '04', '04', '10', '25') 
            self.extract_period(filtered_df, time_column, data_column, '2023', '11', '11', '15', '25') 
            self.extract_period(filtered_df, time_column, data_column, '2024', '02', '02', '15', '20') 
        elif self.station == 'Pituffik':
            self.extract_period(filtered_df, time_column, data_column, '2007', '07', '08', '22', '05')
            self.extract_period(filtered_df, time_column, data_column, '2007', '09', '09', '09', '21')
            self.extract_period(filtered_df, time_column, data_column, '2008', '08', '08', '10', '20')        
            self.extract_period(filtered_df, time_column, data_column, '2016', '02', '02', '01', '29')
            self.extract_period(filtered_df, time_column, data_column, '2016', '03', '03', '01', '30')
            self.extract_period(filtered_df, time_column, data_column, '2017', '11', '11', '01', '05')
            self.extract_period(filtered_df, time_column, data_column, '2019', '01', '01', '13', '19')
            self.extract_period(filtered_df, time_column, data_column, '2020', '09', '09', '01', '30')
            self.extract_period(filtered_df, time_column, data_column, '2022', '09', '09', '13', '25')
            self.extract_period(filtered_df, time_column, data_column, '2023', '09', '09', '01', '15')
            self.extract_period(filtered_df, time_column, data_column, '2023', '12', '12', '10', '20')
            self.extract_period(filtered_df, time_column, data_column, '2024', '03', '03', '01', '10')
            self.extract_period(filtered_df, time_column, data_column, '2024', '05', '05', '20', '30')
            self.extract_period(filtered_df, time_column, data_column, '2024', '10', '10', '25', '31')
        elif self.station == 'Upernavik1':
            self.extract_period(filtered_df, time_column, data_column, '2023', '08', '08', '01', '15')
            self.extract_period(filtered_df, time_column, data_column, '2023', '08', '08', '15', '31')
        elif self.station == 'Upernavik2':   
            self.extract_period(filtered_df, time_column, data_column, '2024', '09', '09', '15', '30')
            self.extract_period(filtered_df, time_column, data_column, '2024', '10', '10', '15', '31')       
        else:
            return 

        #Combine all relevant sections to a long ts
        long_df = pd.concat(self.list_relev_section, ignore_index=True)
        file_name = f"{self.station}-WLdata-long{self.suffix}.csv"
        long_df.to_csv(os.path.join(self.folder_path, file_name), index=False)

        #modified gabs between relevant periods
        combined_df = self.list_relev_section[0]
        
        for i in range(0,len(self.list_relev_section)-1):
            end_date = pd.to_datetime(self.list_relev_section[i][time_column].iloc[-1])
            start_date = pd.to_datetime(self.list_relev_section[i+1][time_column].iloc[0])
            if (start_date - end_date).days < 10:
                combined_df = pd.concat([combined_df, self.list_relev_section[i+1]], ignore_index=True)
            else:
                diff_days = ((start_date - end_date).days)-10
                self.list_relev_section[i+1].loc[:, time_column] = pd.to_datetime(self.list_relev_section[i+1][time_column]) - timedelta(days=diff_days)
                self.list_relev_section[i+1].loc[:, time_column] = pd.to_datetime(self.list_relev_section[i+1][time_column]).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
                combined_df = pd.concat([combined_df, self.list_relev_section[i+1]], ignore_index=True)

        # Save to a CSV file with comma-delimited format
        file_name = f"{self.station}-WLdata{self.suffix}.csv"
        combined_df.to_csv(os.path.join(self.folder_path, file_name), index=False)

        print('Long csv file for manual labelling has been saved.')

    #make a subset of whole timeseries based on provided start and end date & save the subset to csv
    def extract_period(self, data, time_column, data_column, year, start_month, end_month, start_day='1', end_day='31'):
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
        """

        start_date = datetime(int(year), int(start_month), int(start_day))
        end_date = datetime(int(year), int(end_month), int(end_day))
        print(start_date, end_date)
        relev_df = data[(data[time_column] >= start_date) & (data[time_column]<= end_date)]
        if not relev_df.empty:

            relev_df.loc[relev_df[data_column] == self.missing_meas_value, data_column] = None

            self.helper.plot_df(relev_df[time_column],relev_df[data_column], 'Water Level', 'Timestamp',f'WL measurement in {start_month}.{year} at {self.station} - {self.suffix}')

            relev_df_cleaned = relev_df.dropna(subset=[data_column])

            relev_df_cleaned.loc[:, time_column] = relev_df_cleaned[time_column].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
            relev_df_cleaned = relev_df_cleaned.rename(columns={data_column: "value"})

            # Select only the columns you want to save
            columns_to_save = ['series', time_column, 'value', 'label']  # Specify the desired columns
            filtered_df_short = relev_df_cleaned[columns_to_save]

            # Save to a CSV file with comma-delimited format
            file_name = f"{self.station}-WLdata-{start_day,start_month,year}-{self.suffix}.csv"
            filtered_df_short.to_csv(os.path.join(self.folder_path, file_name), index=False)

            #Feedback
            print(f"Filtered columns saved to {file_name}")

            self.list_relev_section.append(filtered_df_short)



    
