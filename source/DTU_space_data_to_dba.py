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
        else:
            raise Exception('Text file contains a measurement which is unkown to the current version of this script.')

        return measurement_name
    
    def run(self):
        self.load_data()
        df = self.convert_data_to_df()
        #sort df by time of measurements
        df = df.sort_values(by='timestamp')
        #fix time as first column and error_flag as last colum
        final_df = df[['timestamp'] + [col for col in df.columns if col not in ['timestamp', 'error_flag']] + ['error_flag']]
        self.export_to_dba(final_df)

    def export_to_dba(self, df):
        df.to_csv(os.path.join(self.output_path, f'{self.station}_data.dba'), sep=' ', index=False)

    def load_data(self):

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
                        # Check if it's a .txt file and starts with the desired prefix
                        base_filename = os.path.basename(zip_info.filename)
                        if base_filename.endswith('.txt') and base_filename.startswith(self.file_prefix):
                            if '_met_' in base_filename:
                                continue
                            else:
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
                                        monthly_df['timestamp'] = pd.to_datetime(monthly_df['date'] + ' ' + monthly_df['time']).dt.strftime('%Y%m%d%H%M%S')
                                        self.data_df.append(monthly_df)
                                        del monthly_df['date']
                                        del monthly_df['time']
                                    elif len(file_columns) == 3:
                                        #Extract measurements the content of the .txt file
                                        monthly_df = pd.read_csv(txt_file, sep='\s+', header=None, names=['date', 'time', measurement_name])
                                        monthly_df['timestamp'] = pd.to_datetime(monthly_df['date'] + ' ' + monthly_df['time']).dt.strftime('%Y%m%d%H%M%S')
                                        monthly_df['error_flag'] = np.nan
                                        del monthly_df['date']
                                        del monthly_df['time']
                                    elif len(file_columns)==2:
                                        monthly_df = pd.read_csv(txt_file, sep=',', header=None, names=['timestamp', measurement_name])
                                        monthly_df['timestamp'] = pd.to_datetime(monthly_df['timestamp'], format='%Y/%m/%d-%H:%M')
                                        monthly_df['timestamp'] = monthly_df['timestamp'].dt.strftime('%Y%m%d%H%M%S')
                                        monthly_df['error_flag'] = np.nan
                                    else:
                                        raise Exception('Content of the .txt a format which is not compatibel with the current code version.')
                                    self.data_df.append(monthly_df)
                                    
    def convert_data_to_df(self):
        # Initialize a dataframe with the first dataframe from the list
        result_df = self.data_df[0]

        # Iterate over each subsequent dataframe in the list
        for df in self.data_df[1:]:
            # Merge matching rows
            common_columns = result_df.columns.intersection(df.columns)
            if len(common_columns)== 2:
                merged_df = pd.merge(result_df, df, on=list(common_columns), how='inner')
            elif len(common_columns) == 3:
                #Sort that common columns are always in the same order
                unknown_column = [col for col in common_columns if col not in ['error_flag', 'timestamp']][0]
                df = df[['error_flag', 'timestamp', unknown_column]]
                df_only = df[~df['timestamp'].isin(result_df['timestamp'])]
                if df_only.empty:
                    # Merge df_A and df_B on 'error_flag' and 'timestamp', using a left join to retain all rows from df_A
                    merged_df = pd.merge(result_df, df[list(common_columns)], on=[df.columns[1], df.columns[0]], how='left', suffixes=('', '_new'))
                    # Fill NaNs in the value column from big df with the values from the measurement df
                    merged_df[df.columns[2]] = merged_df[df.columns[2]].fillna(merged_df[f'{df.columns[2]}_new'])
                    # Drop the extra column
                    merged_df.drop(columns=[f'{df.columns[2]}_new'], inplace=True)
                else:
                    merged_df = pd.merge(result_df, df[list(common_columns)], on=[list(common_columns)[1], list(common_columns)[0], list(common_columns)[2]], how='outer')
            else:
                raise Exception('Dfs are in a format which is not compatibel with the current code version.')
            
            result_df = merged_df

        return result_df