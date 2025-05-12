#############################################################################################################################
#This script reads and saves to csv the DTU Space Greenland measurements saved in yearly .zip files in the same base folder.#
#For folder structure and data see: /net/isilon/ifs/arch/home/ocean/sealevel/Greenland_data_DTU                             #
#Written by frb (November 2023) for GronSL project                                                                          #
#_met data is currently not considered!!                                                                                    #
#############################################################################################################################

import os, sys
import numpy as np
import pandas as pd
import zipfile

class DataConverter():
    
    def __init__(self):
        self.data_path = None
        self.data_df = []

    def load_data_path(self, data_path):
        self.data_path = data_path

    def set_output_path(self, output_path):
        self.output_path = output_path

        #generate output folder for graphs and other docs
        if not os.path.exists(output_path):
            os.makedirs(output_path)

    def set_relev_station(self, station):
        """
        Define where measurement file is from.
        """

        self.station = station

        if station == 'Upernavik':
            self.file_prefix = 'uper_'
        elif station == 'Nuuk':
            self.file_prefix = 'nuuk_'
        elif station == 'Nuuk1':
            self.file_prefix = 'nuk1_'
        elif station == 'Qaqortoq':
            self.file_prefix = 'qaqo_'
        elif station == 'Ittoqqortoormiit':
            self.file_prefix = 'scor_'
        elif station == 'Pituffik':
            self.file_prefix = 'thul_'
        else:
            raise Exception('Prefix to this station is unknown. Make sure that script knows this station and how files connected to this station are saved.')
    
    def get_measured_parameter(self, bla, base_filename):
        """
        Define what is measurement in which file.
        """
        if '_cond' in base_filename:
            measurement_name = 'Conductivity'
        elif '_height.' in base_filename:
            measurement_name = 'Height'
        elif '_height_1013.' in base_filename:
            measurement_name = 'Height(assumed pressure)'
        elif '_press' in base_filename:
            measurement_name = 'Pressure'
        elif '_salin' in base_filename:
            measurement_name = 'Salinity'
        elif '_temp' in base_filename:
            measurement_name = 'Temperature'
        elif '_met' in base_filename:
            measurement_name = 'atmospheric_pressure'
        else:
            raise Exception('Text file contains a measurement which is unkown to the current version of this script.')

        return measurement_name
    
    def run(self):
        """
        Main Method: Open, Convert and save measurements combined to a long csv file.
        """
         
        self.load_data()
        df = self.convert_data_to_df()
        #sort df by time of measurements
        df = df.sort_values(by='Timestamp')
        #fix time as first column and error_flag as last colum
        final_df = df[['Timestamp'] + [col for col in df.columns if col not in ['Timestamp', 'error_flag']] + ['error_flag']]
        self.export_to_csv(final_df)


    def export_to_csv(self, df):
        """
        Save combined df for each station over all years to a csv file.
        """
        df.to_csv(os.path.join(self.output_path, f'{self.station}_data.csv'), index=False)


    def load_data(self):
        """
        Open the different zip. files for each year. Open the correct files for the respective station as monthly dataframe and append them all to a long list of dfs. Different measurements and months have the data differently saved.
        Based on the column count in the first line in the monthly file, decide on how to open the csv file.

        Currently, there are three options:
        1. 4 columns: time, date, measurement & error_flag, Combine time and date to a Timestamp column
        2. 3 columns: time, date & measurement, add error_flag column in code and also combine time and date to a Timestamp column
        3. 2 columns: Timestamp & measurement, add error_flag column in code

        Note: _met measurements are currently ignored as Timestamp is off.
        """

        # Iterate through each file in the directory
        for file_name in os.listdir(self.data_path):
            # Process only .zip files
            if file_name.lower().endswith('.zip'):
                # Get the full path of the .zip file
                zip_path = os.path.join(self.data_path, file_name)
                # Open the .zip file
                with zipfile.ZipFile(zip_path, 'r') as zip_file:
                    # Iterate over each file in the .zip archive
                    for zip_info in zip_file.infolist():
                        # Check if it's a .txt file and starts with the desired prefix (only relevant station)
                        base_filename = os.path.basename(zip_info.filename)
                        if base_filename.endswith('.txt') and base_filename.startswith(self.file_prefix):
                            # Open and read the content of the .txt file
                            with zip_file.open(zip_info) as txt_file:
                                print(f"Loaded DataFrame from {zip_info.filename} in {file_name}")
                                #Extract which measurement it is
                                measurement_name = self.get_measured_parameter(zip_path, base_filename)
                                #check formate of the txt.file
                                with zip_file.open(zip_info) as txt_file2:
                                    df = pd.read_csv(txt_file2, sep='\s+', nrows=1)
                                file_columns = set(df.columns)
                                if len(file_columns) == 4:
                                    #Extract measurements the content of the .txt file
                                    monthly_df = pd.read_csv(txt_file, sep='\s+', header=None, names=['date', 'time', measurement_name, 'error_flag'])
                                    if '_met_' in base_filename:
                                        print('At the moment, the _met meassurements are ignored as they are measured on a different Timestamp. All other measurements come probably from the same sensor as they have the same registration periods.')
                                        continue
                                    initial_row_count = len(monthly_df)
                                    monthly_df['date'] = pd.to_datetime(monthly_df['date'], errors='coerce')
                                    monthly_df['time'] = pd.to_datetime(monthly_df['time'], format='%H:%M:%S', errors='coerce')
                                    monthly_df = monthly_df.dropna(subset=['date', 'time'])
                                    if initial_row_count != len(monthly_df):
                                        print(f'{initial_row_count- len(monthly_df)} rows were deleted from measurements, because they had an invalid date (f.e. 32.01.2007).')
                                    monthly_df['time'] = monthly_df['time'].dt.time.astype(str)
                                    monthly_df['date'] = monthly_df['date'].dt.date.astype(str)
                                    monthly_df['Timestamp'] = pd.to_datetime(monthly_df['date']+ ' ' + monthly_df['time']).dt.strftime('%Y%m%d%H%M%S')
                                    del monthly_df['date']
                                    del monthly_df['time']
                                elif len(file_columns) == 3:
                                    #Extract measurements the content of the .txt file
                                    monthly_df = pd.read_csv(txt_file, sep='\s+', header=None, names=['date', 'time', measurement_name])
                                    initial_row_count = len(monthly_df)
                                    monthly_df['date'] = pd.to_datetime(monthly_df['date'], errors='coerce')
                                    monthly_df['time'] = pd.to_datetime(monthly_df['time'], format='%H:%M:%S', errors='coerce')
                                    monthly_df = monthly_df.dropna(subset=['date', 'time'])
                                    if initial_row_count != len(monthly_df):
                                        print(f'{initial_row_count- len(monthly_df)} rows were deleted from measurements, because they had an invalid date (f.e. 32.01.2007).')
                                    monthly_df['time'] = monthly_df['time'].dt.time.astype(str)
                                    monthly_df['date'] = monthly_df['date'].dt.date.astype(str)
                                    monthly_df['Timestamp'] = pd.to_datetime(monthly_df['date']+ ' ' + monthly_df['time']).dt.strftime('%Y%m%d%H%M%S')
                                    monthly_df['error_flag'] = np.nan
                                    del monthly_df['date']
                                    del monthly_df['time']
                                elif len(file_columns)==2:
                                    monthly_df = pd.read_csv(txt_file, sep=',', header=None, names=['Timestamp', measurement_name])
                                    monthly_df['Timestamp'] = pd.to_datetime(monthly_df['Timestamp'], format='%Y/%m/%d-%H:%M')
                                    monthly_df['Timestamp'] = monthly_df['Timestamp'].dt.strftime('%Y%m%d%H%M%S')
                                    monthly_df['error_flag'] = np.nan
                                else:
                                    raise Exception('Content of the .txt a format which is not compatibel with the current code version.')
                                if self.station == 'Qaqortoq' and measurement_name == 'Temperature':
                                    #Do not append temperature measurements as they are shifted during the first two years until roughly mid 2007
                                    print('At the moment, the temperature meassurements for Qaqortoq are ignored as they have a shift error in their time stamp. This needs to be further analysed before merging the measurement records.')
                                    continue
                                else:
                                    self.data_df.append(monthly_df)


    def convert_data_to_df(self):
        """
        Convert list of monthly dataframes and different measurements dfs to one long dataframe for this station. Append the same measurement to the columns, respectively.
        """
                
        # Initialize a dataframe with the first monthly dataframe from the list
        result_df = self.data_df[0]

        # Iterate over each subsequent monthly dataframe in the list and append it
        for df in self.data_df[1:]:
            #Get common column names between previous and current df
            common_columns = result_df.columns.intersection(df.columns)
            #If common column name only 2 (tiestamp and error_flag), append the new values to those column and add the new measurement type as a new column
            if len(common_columns)== 2:
                merged_df = pd.merge(result_df, df, on=list(common_columns), how='left')
            #if all measurement types exist in the big df, fill rows and columns for all months accordingly
            elif len(common_columns) == 3:
                #Sort that common columns are always in the same order
                unknown_column = [col for col in common_columns if col not in ['error_flag', 'Timestamp']][0]
                df = df[['error_flag', 'Timestamp', unknown_column]]
                df_only = df[~df['Timestamp'].isin(result_df['Timestamp'])]
                if df_only.empty:
                    # Merge df_A and df_B on 'error_flag' and 'Timestamp', using a left join to retain all rows from df_A
                    merged_df = pd.merge(result_df, df[list(common_columns)], on=[df.columns[1]], how='left', suffixes=('', '_new'))
                    # Fill NaNs in the value column from big df with the values from the measurement df
                    merged_df[df.columns[2]] = merged_df[df.columns[2]].fillna(merged_df[f'{df.columns[2]}_new'])
                    merged_df[df.columns[0]] = merged_df[df.columns[0]].fillna(merged_df[f'{df.columns[0]}_new'])
                    # Drop the extra column
                    merged_df.drop(columns=[f'{df.columns[2]}_new', f'{df.columns[0]}_new'], inplace=True)
                else:
                    merged_df = pd.merge(result_df, df[list(common_columns)], on=[list(common_columns)[1], list(common_columns)[0], list(common_columns)[2]], how='outer')
            else:
                raise Exception('Dfs are in a format which is not compatibel with the current code version.')
            
            result_df = merged_df

            #Lines can be used for Qaqortoq temperature debugging later on.
            #if str(df['Timestamp'].iloc[0]).startswith('2005'):
            #    df_subset = merged_df[merged_df['Timestamp'].astype(str).str.startswith('2005', na=False)].reset_index(drop=True)
            #    print(df_subset.iloc[0])
            #    print('hey')

        return result_df