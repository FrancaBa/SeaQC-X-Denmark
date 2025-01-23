####################################################################################
## Written by frb for GronSL project (2024-2025)                                  ##
## Conversion various test outcomes to bitmaks/QC flags as defined in config.json ##
####################################################################################

import os
import pandas as pd
import numpy as np

import source.helper_methods as helper

class QualityMasking():

    def __init__(self):
        self.helper = helper.HelperMethods()
    
    def set_output_folder(self, folder_path):
        self.folder_path = os.path.join(folder_path,'Flagged ts')

        #generate output folder for graphs and other docs
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

        self.helper.set_output_folder(self.folder_path)
    
    def set_flags(self, flag_def):
        self.flag_def = flag_def

    def set_bitmask(self, bitmask_def):
        self.bitmask_def = bitmask_def

    def set_station(self, station):
        self.station = station

    def set_active_tests(self, active_tests):
        self.active_tests = active_tests

    def convert_boolean_to_bitmask(self, df):

        zero_bit = 0

        for col_name in df.columns:
            if col_name == 'incorrect_format' and self.active_tests['incorrect_format']:
                bitmask_elem = self.bitmask_def['incorrect_format']
                df[f'bit_{bitmask_elem}'] = [bitmask_elem if value else zero_bit for value in df[col_name]]
            elif col_name == 'missing_values' and self.active_tests['missing_data']:
                bitmask_elem = self.bitmask_def['missing_data']
                df[f'bit_{bitmask_elem}'] = [bitmask_elem if value else zero_bit for value in df[col_name]]
            elif col_name == 'stuck_value' and self.active_tests['stuck_value']:
                bitmask_elem = self.bitmask_def['stuck_value']
                df[f'bit_{bitmask_elem}'] = [bitmask_elem if value else zero_bit for value in df[col_name]]
            elif col_name == 'outlier' and self.active_tests['global_outliers']:
                bitmask_elem = self.bitmask_def['global_outliers']
                df[f'bit_{bitmask_elem}'] = [bitmask_elem if value else zero_bit for value in df[col_name]]
            elif col_name == 'interpolated_value' and self.active_tests['interpolated_value']:
                bitmask_elem = self.bitmask_def['interpolated_value']
                df[f'bit_{bitmask_elem}'] = [bitmask_elem if value else zero_bit for value in df[col_name]]
            elif col_name == 'short_bad_measurement_series' and self.active_tests['bad_segment']:
                bitmask_elem = self.bitmask_def['bad_segment']
                df[f'bit_{bitmask_elem}'] = [bitmask_elem if value else zero_bit for value in df[col_name]]
            elif col_name == 'outlier_change_rate' and self.active_tests['outlier_change_rate']:
                bitmask_elem = self.bitmask_def['spikes']
                df['new'] = [bitmask_elem if value else zero_bit for value in df[col_name]]
                if f'bit_{bitmask_elem}' in df.columns:
                    df[f'bit_{bitmask_elem}'] = df.apply(lambda row: row[f'bit_{bitmask_elem}'] if row[f'bit_{bitmask_elem}'] == bitmask_elem else row['new'], axis=1)
                else:
                    df[f'bit_{bitmask_elem}'] = [bitmask_elem if value else zero_bit for value in df[col_name]]
                del df['new']
            elif col_name == 'noisy_period' and self.active_tests['noisy_period']:
                bitmask_elem = self.bitmask_def['noisy_period']
                df[f'bit_{bitmask_elem}'] = [bitmask_elem if value else zero_bit for value in df[col_name]]
            elif col_name == 'spike_value_statistical' and self.active_tests['spike_value_statistical']:
                bitmask_elem = self.bitmask_def['spikes']
                df['new'] = [bitmask_elem if value else zero_bit for value in df[col_name]]
                if f'bit_{bitmask_elem}' in df.columns:
                    df[f'bit_{bitmask_elem}'] = df.apply(lambda row: row[f'bit_{bitmask_elem}'] if row[f'bit_{bitmask_elem}'] == bitmask_elem else row['new'], axis=1)
                else:
                    df[f'bit_{bitmask_elem}'] = [bitmask_elem if value else zero_bit for value in df[col_name]]
                del df['new']
            elif col_name == 'cotede_spikes' and self.active_tests['cotede_spikes']:
                bitmask_elem = self.bitmask_def['spikes']
                df['new'] = [bitmask_elem if value else zero_bit for value in df[col_name]]
                if f'bit_{bitmask_elem}' in df.columns:
                    df[f'bit_{bitmask_elem}'] = df.apply(lambda row: row[f'bit_{bitmask_elem}'] if row[f'bit_{bitmask_elem}'] == bitmask_elem else row['new'], axis=1)
                else:
                    df[f'bit_{bitmask_elem}'] = [bitmask_elem if value else zero_bit for value in df[col_name]]
                del df['new']
            elif col_name == 'cotede_improved_spikes' and self.active_tests['cotede_improved_spikes']:
                bitmask_elem = self.bitmask_def['spikes']
                df['new'] = [bitmask_elem if value else zero_bit for value in df[col_name]]
                if f'bit_{bitmask_elem}' in df.columns:
                    df[f'bit_{bitmask_elem}'] = df.apply(lambda row: row[f'bit_{bitmask_elem}'] if row[f'bit_{bitmask_elem}'] == bitmask_elem else row['new'], axis=1)
                else:
                    df[f'bit_{bitmask_elem}'] = [bitmask_elem if value else zero_bit for value in df[col_name]]
                del df['new']
            elif col_name == 'selene_spikes' and self.active_tests['selene_spikes']:
                bitmask_elem = self.bitmask_def['spikes']
                df['new'] = [bitmask_elem if value else zero_bit for value in df[col_name]]
                if f'bit_{bitmask_elem}' in df.columns:
                    df[f'bit_{bitmask_elem}'] = df.apply(lambda row: row[f'bit_{bitmask_elem}'] if row[f'bit_{bitmask_elem}'] == bitmask_elem else row['new'], axis=1)
                else:
                    df[f'bit_{bitmask_elem}'] = [bitmask_elem if value else zero_bit for value in df[col_name]]
                del df['new']
            elif col_name == 'selene_improved_spikes' and self.active_tests['selene_improved_spikes']:
                bitmask_elem = self.bitmask_def['spikes']
                df['new'] = [bitmask_elem if value else zero_bit for value in df[col_name]]
                if f'bit_{bitmask_elem}' in df.columns:
                    df[f'bit_{bitmask_elem}'] = df.apply(lambda row: row[f'bit_{bitmask_elem}'] if row[f'bit_{bitmask_elem}'] == bitmask_elem else row['new'], axis=1)
                else:
                    df[f'bit_{bitmask_elem}'] = [bitmask_elem if value else zero_bit for value in df[col_name]]
                del df['new']
            elif col_name == 'harmonic_detected_spikes' and self.active_tests['harmonic_detected_spikes']:
                bitmask_elem = self.bitmask_def['spikes']
                df['new'] = [bitmask_elem if value else zero_bit for value in df[col_name]]
                if f'bit_{bitmask_elem}' in df.columns:
                    df[f'bit_{bitmask_elem}'] = df.apply(lambda row: row[f'bit_{bitmask_elem}'] if row[f'bit_{bitmask_elem}'] == bitmask_elem else row['new'], axis=1)
                else:
                    df[f'bit_{bitmask_elem}'] = [bitmask_elem if value else zero_bit for value in df[col_name]]
                del df['new']
            elif col_name == 'ml_detected_spikes' and self.active_tests['ml_detected_spikes']:
                bitmask_elem = self.bitmask_def['spikes']
                df['new'] = [bitmask_elem if value else zero_bit for value in df[col_name]]
                if f'bit_{bitmask_elem}' in df.columns:
                    df[f'bit_{bitmask_elem}'] = df.apply(lambda row: row[f'bit_{bitmask_elem}'] if row[f'bit_{bitmask_elem}'] == bitmask_elem else row['new'], axis=1)
                else:
                    df[f'bit_{bitmask_elem}'] = [bitmask_elem if value else zero_bit for value in df[col_name]]
                del df['new']
            elif col_name == 'shifted_period' and self.active_tests['shifted_value']:
                bitmask_elem = self.bitmask_def['shifted_value']
                df['new'] = [bitmask_elem if value else zero_bit for value in df[col_name]]
                if f'bit_{bitmask_elem}' in df.columns:
                    df[f'bit_{bitmask_elem}'] = df.apply(lambda row: row[f'bit_{bitmask_elem}'] if row[f'bit_{bitmask_elem}'] == bitmask_elem else row['new'], axis=1)
                else:
                    df[f'bit_{bitmask_elem}'] = [bitmask_elem if value else zero_bit for value in df[col_name]]
                del df['new']
            elif col_name == 'shifted_ruptures' and self.active_tests['shifted_ruptures']:
                bitmask_elem = self.bitmask_def['shifted_value']
                df['new'] = [bitmask_elem if value else zero_bit for value in df[col_name]]
                if f'bit_{bitmask_elem}' in df.columns:
                    df[f'bit_{bitmask_elem}'] = df.apply(lambda row: row[f'bit_{bitmask_elem}'] if row[f'bit_{bitmask_elem}'] == bitmask_elem else row['new'], axis=1)
                else:
                    df[f'bit_{bitmask_elem}'] = [bitmask_elem if value else zero_bit for value in df[col_name]]
                del df['new']
            elif col_name == 'probably_good_mask' and self.active_tests['probably_good_data']:
                bitmask_elem = self.bitmask_def['probably_good_data']
                df[f'bit_{bitmask_elem}'] = [bitmask_elem if value else zero_bit for value in df[col_name]]

        bit_dfs = df.filter(regex='^bit_')
        df['combined_int'] = np.bitwise_or.reduce(bit_dfs, axis=1)
        df['combined_bitmask'] = df['combined_int'].apply(lambda x: format(x, '010b'))

        return df

    def convert_bitmask_to_flags(self, df):
        #Order is important
        df['quality_flag'] = df['combined_int'].apply(lambda x: self.flag_def["good_data"] if x == 0 else 0) #flag good data
        df['quality_flag'] = df.apply(lambda row: self.flag_def["probably_good_data"] if (row['combined_int'] & self.bitmask_def['probably_good_data']) == self.bitmask_def['probably_good_data'] else row['quality_flag'], axis=1) #flag probably good data

        df['quality_flag'] = df.apply(lambda row: self.flag_def["bad_data_correctable"] if (row['combined_int'] & self.bitmask_def['noisy_period']) == self.bitmask_def['noisy_period'] else row['quality_flag'], axis=1) #flag bad correctable data (noisy period)
        df['quality_flag'] = df.apply(lambda row: self.flag_def["bad_data_correctable"] if (row['combined_int'] & self.bitmask_def['bad_segment']) == self.bitmask_def['bad_segment'] else row['quality_flag'], axis=1) #flag bad correctable data (bad segment)

        df['quality_flag'] = df.apply(lambda row: self.flag_def["bad_data"] if (row['combined_int'] & self.bitmask_def['global_outliers']) == self.bitmask_def['global_outliers'] else row['quality_flag'], axis=1)  #flag bad not correctable data (global outliers)
        df['quality_flag'] = df.apply(lambda row: self.flag_def["bad_data"] if (row['combined_int'] & self.bitmask_def['incorrect_format']) == self.bitmask_def['incorrect_format'] else row['quality_flag'], axis=1) #flag bad not correctable data (incorrect format)
        
        df['quality_flag'] = df.apply(lambda row: self.flag_def["spikes"] if (row['combined_int'] & self.bitmask_def['spikes']) == self.bitmask_def['spikes'] else row['quality_flag'], axis=1) #flag spikes
        df['quality_flag'] = df.apply(lambda row: self.flag_def["shifted_value"] if (row['combined_int'] & self.bitmask_def['shifted_value']) == self.bitmask_def['shifted_value'] else row['quality_flag'], axis=1) #flag shifted value     
        
        df['quality_flag'] = df.apply(lambda row: self.flag_def["stuck_value"] if (row['combined_int'] & self.bitmask_def['stuck_value']) == self.bitmask_def['stuck_value'] else row['quality_flag'], axis=1) #flag stuck value
        df['quality_flag'] = df.apply(lambda row: self.flag_def["interpolated_value"] if (row['combined_int'] & self.bitmask_def['interpolated_value']) == self.bitmask_def['interpolated_value'] else row['quality_flag'], axis=1) #flag interpolated value
        df['quality_flag'] = df.apply(lambda row: self.flag_def["missing_data"] if (row['combined_int'] & self.bitmask_def['missing_data']) == self.bitmask_def['missing_data'] else row['quality_flag'], axis=1) #flag missing data
 
        #Raise error if any 0 in 'quality check' column (this cannot ne happening as all points are tested)
        #0 = no QC
        if (df['quality_flag']==0).any():
            raise Exception('There are columns with flag 0. However, this cannot be as all values are quality checked.')
        return df

    def save_flagged_series(self, df, meas_col_name, time_column, short_df):
        #Extract relevant columns for export and reset timestamp
        checked_df = df[[time_column, meas_col_name, 'quality_flag','combined_bitmask']]
        relev_short_df = short_df[[time_column, meas_col_name]]
        export_df = pd.merge(checked_df, relev_short_df, on=[time_column, meas_col_name], how='right')

        #Save df to csv
        file_name = f"{self.station}_quality_checked_WL.csv"
        export_df.to_csv(os.path.join(self.folder_path, file_name), index=False)
