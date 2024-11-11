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
        elif '_height' in base_filename:
            measurement_name = 'Height'
        elif '_height_1013' in base_filename:
            measurement_name = 'Height (assumed pressure)'
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

        # Iterate through each file in the directory
        for file_name in os.listdir(self.data_path):
            # Process only .zip files
            print(file_name)
            if file_name.lower().endswith('.zip'):
                # Get the full path of the .zip file
                zip_path = os.path.join(self.data_path, file_name)
                # Open the .zip file
                with zipfile.ZipFile(zip_path, 'r') as zip_file:
                    # Iterate over each file in the .zip archive
                    for zip_info in zip_file.infolist():
                        # Check if it's a .txt file and starts with the desired prefix
                        base_filename = os.path.basename(zip_info.filename)
                        print(base_filename)
                        if base_filename.endswith('.txt') and base_filename.startswith(self.file_prefix):
                            if '_met_' in base_filename:
                                continue
                            else:
                                # Open and read the content of the .txt file
                                with zip_file.open(zip_info) as txt_file:
                                    #Extract which measurement it is
                                    measurement_name = self.get_measured_parameter(zip_path, base_filename)
                                    # Process the content of the .txt file
                                    monthly_df = pd.read_csv(txt_file, sep='\s+', header=None, names=['date', 'time', measurement_name, 'error_flag'])
                                    self.data_df.append(monthly_df)
                                    print(f"Loaded DataFrame from {zip_info.filename} in {file_name}")

        # Combine all DataFrames into one and save it
        combined_df = pd.concat(self.data_df, ignore_index=True)
        combined_df.to_csv(os.path.join(self.output_path, f'{self.station}_data.dba'), index=False)
        
