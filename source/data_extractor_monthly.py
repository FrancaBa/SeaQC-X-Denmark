####################################################################################################
## Extrac interesting measurement periods by date (written by frb for GronSL project (2024-2025)) ##
####################################################################################################

import os, sys
import numpy as np
import pandas as pd

from datetime import datetime, timedelta

import source.helper_methods as helper

class DataExtractor():
    
    def __init__(self):
        self.missing_meas_value = 999.000
        self.list_relev_section = []
        self.helper = helper.HelperMethods()
    
    def set_output_folder(self, folder, station):
        self.folder_path = os.path.join(folder,'interesting ts and their graphs')

        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

        self.helper.set_output_folder(self.folder_path)
        self.station = station

    def run(self, df, time_column, data_column):

        filtered_df = df.copy()
        filtered_df['label'] = None
        filtered_df['series'] = 'seriesA'

        if self.station == 'Ittoqqortoormiit':
            self.extract_period(filtered_df, time_column, data_column, '2009', '06', '06', '07', '13')
            self.extract_period(filtered_df, time_column, data_column, '2011', '03', '03', '05', '17')
            self.extract_period(filtered_df, time_column, data_column, '2013', '08', '11', '20', '08')
            self.extract_period(filtered_df, time_column, data_column, '2017', '03', '05', '10', '25')        
            self.extract_period(filtered_df, time_column, data_column, '2019', '06', '06', '01', '10')
            self.extract_period(filtered_df, time_column, data_column, '2022', '06', '06', '25', '30')
            self.extract_period(filtered_df, time_column, data_column, '2023', '03', '04', '30', '05')
            self.extract_period(filtered_df, time_column, data_column, '2023', '10', '10', '01', '10')
            self.extract_period(filtered_df, time_column, data_column, '2023', '11', '11', '13', '30')
            self.extract_period(filtered_df, time_column, data_column, '2023', '12', '12', '25', '31')

        elif self.station == 'Qaqortoq':
            self.extract_period(filtered_df, time_column, data_column, '2008', '10', '10', '1', '10')
            self.extract_period(filtered_df, time_column, data_column, '2014', '01', '01', '22', '30')
            self.extract_period(filtered_df, time_column, data_column, '2015', '07', '07', '02', '23')  
            self.extract_period(filtered_df, time_column, data_column, '2018', '05', '05', '03', '20')
            self.extract_period(filtered_df, time_column, data_column, '2023', '07', '07', '01', '10')
                     

        #Combine all relevant sections to a long ts
        long_df = pd.concat(self.list_relev_section, ignore_index=True)
        file_name = f"{self.station}-WLdata-long.csv"
        long_df.to_csv(os.path.join(self.folder_path, file_name), index=False)

        #modified gabs between relevant periods
        combined_df = self.list_relev_section[0]

        for i in range(0,len(self.list_relev_section)-1):
            end_date = pd.to_datetime(self.list_relev_section[i]['timestamp'].iloc[-1])
            start_date = pd.to_datetime(self.list_relev_section[i+1]['timestamp'].iloc[0])
            if (start_date - end_date).days < 10:
                combined_df = pd.concat([combined_df, self.list_relev_section[i+1]], ignore_index=True)
            else:
                diff_days = ((start_date - end_date).days)-10
                self.list_relev_section[i+1].loc[:, 'timestamp'] = pd.to_datetime(self.list_relev_section[i+1]['timestamp']) - timedelta(days=diff_days)
                self.list_relev_section[i+1].loc[:, 'timestamp'] = pd.to_datetime(self.list_relev_section[i+1]['timestamp']).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
                combined_df = pd.concat([combined_df, self.list_relev_section[i+1]], ignore_index=True)

        # Save to a CSV file with comma-delimited format
        file_name = f"{self.station}-WLdata.csv"
        combined_df.to_csv(os.path.join(self.folder_path, file_name), index=False)

        print('long csv file for manual labelling has been saved.')


    def extract_period(self, data, time_column, data_column, year, start_month, end_month, start_day='1', end_day='31'):

        start_date = datetime(int(year), int(start_month), int(start_day))
        end_date = datetime(int(year), int(end_month), int(end_day))
        print(start_date, end_date)
        relev_df = data[(data[time_column] >= start_date) & (data[time_column]<= end_date)]
        if not relev_df.empty:

            relev_df.loc[relev_df[data_column] == self.missing_meas_value, data_column] = None

            self.helper.plot_df(relev_df[time_column],relev_df[data_column], 'Water Level', 'Timestamp',f'WL measurement in {start_month}.{year} at {self.station}')

            relev_df_cleaned = relev_df.dropna(subset=[data_column])

            relev_df_cleaned.loc[:, 'timestamp'] = relev_df_cleaned[time_column].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
            relev_df_cleaned = relev_df_cleaned.rename(columns={data_column: "value"})

            # Select only the columns you want to save
            columns_to_save = ['series', 'timestamp', 'value', 'label']  # Specify the desired columns
            filtered_df_short = relev_df_cleaned[columns_to_save]

            # Save to a CSV file with comma-delimited format
            file_name = f"{self.station}-WLdata-{start_month,year}.csv"
            filtered_df_short.to_csv(os.path.join(self.folder_path, file_name), index=False)

            #Feedback
            print(f"Filtered columns saved to {file_name}")

            self.list_relev_section.append(filtered_df_short)



    
