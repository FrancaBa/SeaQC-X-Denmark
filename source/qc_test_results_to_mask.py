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
    
    #Defines relevant QC flags based on config.json
    def set_flags(self, flag_def):
        self.flag_def = flag_def

    #Defines applied bitmask based on config.json
    def set_bitmask(self, bitmask_def):
        self.bitmask_def = bitmask_def

    #Defines station based on test input
    def set_station(self, station):
        self.station = station

    #Defines which tests are activ based on config.json
    def set_active_tests(self, active_tests):
        self.active_tests = active_tests

    def convert_boolean_to_bitmasks(self, df, suffix=''):
        """
        Converts boolean test columns to bitmask, respectively.

        Input:
        -Main dataframe [pandas df]
        -suffix: ending for columns and graphs in order to run in different modes [str]
        """

        zero_bit = 0

        for col_name in df.columns:
            if col_name == f'incorrect_format{suffix}' and self.active_tests['incorrect_format']:
                bitmask_elem = self.bitmask_def[f'incorrect_format{suffix}']
                df[f'bit_{bitmask_elem}{suffix}'] = [bitmask_elem if value else zero_bit for value in df[col_name]]
            elif col_name == f'missing_values{suffix}' and self.active_tests['missing_data']:
                bitmask_elem = self.bitmask_def[f'missing_data{suffix}']
                df[f'bit_{bitmask_elem}{suffix}'] = [bitmask_elem if value else zero_bit for value in df[col_name]]
            elif col_name == f'stuck_value{suffix}' and self.active_tests['stuck_value']:
                bitmask_elem = self.bitmask_def[f'stuck_value{suffix}']
                df[f'bit_{bitmask_elem}{suffix}'] = [bitmask_elem if value else zero_bit for value in df[col_name]]
            elif col_name == f'outlier{suffix}' and self.active_tests['global_outliers']:
                bitmask_elem = self.bitmask_def[f'global_outliers{suffix}']
                df[f'bit_{bitmask_elem}{suffix}'] = [bitmask_elem if value else zero_bit for value in df[col_name]]
            elif col_name == f'interpolated_value{suffix}' and self.active_tests['interpolated_value']:
                bitmask_elem = self.bitmask_def[f'interpolated_value{suffix}']
                df[f'bit_{bitmask_elem}{suffix}'] = [bitmask_elem if value else zero_bit for value in df[col_name]]
            elif col_name == f'short_bad_measurement_series{suffix}' and self.active_tests['bad_segment']:
                bitmask_elem = self.bitmask_def[f'bad_segment{suffix}']
                df[f'bit_{bitmask_elem}{suffix}'] = [bitmask_elem if value else zero_bit for value in df[col_name]]
            elif col_name == f'outlier_change_rate{suffix}' and self.active_tests['outlier_change_rate']:
                bitmask_elem = self.bitmask_def[f'spikes{suffix}']
                df['new'] = [bitmask_elem if value else zero_bit for value in df[col_name]]
                if f'bit_{bitmask_elem}{suffix}' in df.columns:
                    df[f'bit_{bitmask_elem}{suffix}'] = df.apply(lambda row: row[f'bit_{bitmask_elem}{suffix}'] if row[f'bit_{bitmask_elem}{suffix}'] == bitmask_elem else row['new'], axis=1)
                else:
                    df[f'bit_{bitmask_elem}{suffix}'] = [bitmask_elem if value else zero_bit for value in df[col_name]]
                del df['new']
            elif col_name == f'noisy_period{suffix}' and self.active_tests['noisy_period']:
                bitmask_elem = self.bitmask_def[f'noisy_period{suffix}']
                df[f'bit_{bitmask_elem}{suffix}'] = [bitmask_elem if value else zero_bit for value in df[col_name]]
            elif col_name == f'spike_value_statistical{suffix}' and self.active_tests['spike_value_statistical']:
                bitmask_elem = self.bitmask_def[f'spikes{suffix}']
                df['new'] = [bitmask_elem if value else zero_bit for value in df[col_name]]
                if f'bit_{bitmask_elem}{suffix}' in df.columns:
                    df[f'bit_{bitmask_elem}{suffix}'] = df.apply(lambda row: row[f'bit_{bitmask_elem}{suffix}'] if row[f'bit_{bitmask_elem}{suffix}'] == bitmask_elem else row['new'], axis=1)
                else:
                    df[f'bit_{bitmask_elem}{suffix}'] = [bitmask_elem if value else zero_bit for value in df[col_name]]
                del df['new']
            elif col_name == f'cotede_spikes{suffix}' and self.active_tests['cotede_spikes']:
                bitmask_elem = self.bitmask_def[f'spikes{suffix}']
                df['new'] = [bitmask_elem if value else zero_bit for value in df[col_name]]
                if f'bit_{bitmask_elem}{suffix}' in df.columns:
                    df[f'bit_{bitmask_elem}{suffix}'] = df.apply(lambda row: row[f'bit_{bitmask_elem}{suffix}'] if row[f'bit_{bitmask_elem}{suffix}'] == bitmask_elem else row['new'], axis=1)
                else:
                    df[f'bit_{bitmask_elem}{suffix}'] = [bitmask_elem if value else zero_bit for value in df[col_name]]
                del df['new']
            elif col_name == f'cotede_improved_spikes{suffix}' and self.active_tests['cotede_improved_spikes']:
                bitmask_elem = self.bitmask_def[f'spikes{suffix}']
                df['new'] = [bitmask_elem if value else zero_bit for value in df[col_name]]
                if f'bit_{bitmask_elem}{suffix}' in df.columns:
                    df[f'bit_{bitmask_elem}{suffix}'] = df.apply(lambda row: row[f'bit_{bitmask_elem}{suffix}'] if row[f'bit_{bitmask_elem}{suffix}'] == bitmask_elem else row['new'], axis=1)
                else:
                    df[f'bit_{bitmask_elem}{suffix}'] = [bitmask_elem if value else zero_bit for value in df[col_name]]
                del df['new']
            elif col_name == f'selene_spikes{suffix}' and self.active_tests['selene_spikes']:
                bitmask_elem = self.bitmask_def[f'spikes{suffix}']
                df['new'] = [bitmask_elem if value else zero_bit for value in df[col_name]]
                if f'bit_{bitmask_elem}{suffix}' in df.columns:
                    df[f'bit_{bitmask_elem}{suffix}'] = df.apply(lambda row: row[f'bit_{bitmask_elem}{suffix}'] if row[f'bit_{bitmask_elem}{suffix}'] == bitmask_elem else row['new'], axis=1)
                else:
                    df[f'bit_{bitmask_elem}{suffix}'] = [bitmask_elem if value else zero_bit for value in df[col_name]]
                del df['new']
            elif col_name == f'selene_improved_spikes{suffix}' and self.active_tests['selene_improved_spikes']:
                bitmask_elem = self.bitmask_def[f'spikes{suffix}']
                df['new'] = [bitmask_elem if value else zero_bit for value in df[col_name]]
                if f'bit_{bitmask_elem}{suffix}' in df.columns:
                    df[f'bit_{bitmask_elem}{suffix}'] = df.apply(lambda row: row[f'bit_{bitmask_elem}{suffix}'] if row[f'bit_{bitmask_elem}{suffix}'] == bitmask_elem else row['new'], axis=1)
                else:
                    df[f'bit_{bitmask_elem}{suffix}'] = [bitmask_elem if value else zero_bit for value in df[col_name]]
                del df['new']
            elif col_name == f'harmonic_detected_spikes{suffix}' and self.active_tests['harmonic_detected_spikes']:
                bitmask_elem = self.bitmask_def[f'spikes{suffix}']
                df['new'] = [bitmask_elem if value else zero_bit for value in df[col_name]]
                if f'bit_{bitmask_elem}{suffix}' in df.columns:
                    df[f'bit_{bitmask_elem}{suffix}'] = df.apply(lambda row: row[f'bit_{bitmask_elem}{suffix}'] if row[f'bit_{bitmask_elem}{suffix}'] == bitmask_elem else row['new'], axis=1)
                else:
                    df[f'bit_{bitmask_elem}{suffix}'] = [bitmask_elem if value else zero_bit for value in df[col_name]]
                del df['new']
            elif col_name == f'ml_detected_spikes{suffix}' and self.active_tests['ml_detected_spikes']:
                bitmask_elem = self.bitmask_def[f'spikes{suffix}']
                df['new'] = [bitmask_elem if value else zero_bit for value in df[col_name]]
                if f'bit_{bitmask_elem}{suffix}' in df.columns:
                    df[f'bit_{bitmask_elem}{suffix}'] = df.apply(lambda row: row[f'bit_{bitmask_elem}{suffix}'] if row[f'bit_{bitmask_elem}{suffix}'] == bitmask_elem else row['new'], axis=1)
                else:
                    df[f'bit_{bitmask_elem}{suffix}'] = [bitmask_elem if value else zero_bit for value in df[col_name]]
                del df['new']
            elif col_name == f'shifted_period{suffix}' and self.active_tests['shifted_value']:
                bitmask_elem = self.bitmask_def[f'shifted_value{suffix}']
                df['new'] = [bitmask_elem if value else zero_bit for value in df[col_name]]
                if f'bit_{bitmask_elem}{suffix}' in df.columns:
                    df[f'bit_{bitmask_elem}{suffix}'] = df.apply(lambda row: row[f'bit_{bitmask_elem}{suffix}'] if row[f'bit_{bitmask_elem}{suffix}'] == bitmask_elem else row['new'], axis=1)
                else:
                    df[f'bit_{bitmask_elem}{suffix}'] = [bitmask_elem if value else zero_bit for value in df[col_name]]
                del df['new']
            elif col_name == f'shifted_ruptures{suffix}' and self.active_tests['shifted_ruptures']:
                bitmask_elem = self.bitmask_def[f'shifted_value{suffix}']
                df['new'] = [bitmask_elem if value else zero_bit for value in df[col_name]]
                if f'bit_{bitmask_elem}{suffix}' in df.columns:
                    df[f'bit_{bitmask_elem}{suffix}'] = df.apply(lambda row: row[f'bit_{bitmask_elem}{suffix}'] if row[f'bit_{bitmask_elem}{suffix}'] == bitmask_elem else row['new'], axis=1)
                else:
                    df[f'bit_{bitmask_elem}{suffix}'] = [bitmask_elem if value else zero_bit for value in df[col_name]]
                del df['new']
            elif col_name == f'probably_good_mask{suffix}' and self.active_tests['probably_good_data']:
                bitmask_elem = self.bitmask_def[f'probably_good_data{suffix}']
                df[f'bit_{bitmask_elem}{suffix}'] = [bitmask_elem if value else zero_bit for value in df[col_name]]

        return df
    
    def merge_bitmasks(self, df, detide_mode, information, suffix):
        """
        Merges different bits to a combined bitmask and merged corresponding bit from detided and tide mode to one.

        Input:
        -Main dataframe [pandas df]
        -Detide mode [boolean]
        -Information list where QC report is collected [lst]
        -suffix: ending for columns and graphs in order to run in different modes [str]
        """
        #merge the various bitmasks to one combined column        
        bit_dfs = df.filter(regex='^bit_')
        df['combined_int_orig'] = np.bitwise_or.reduce(bit_dfs, axis=1)
        df['combined_bitmask'] = df['combined_int_orig'].apply(lambda x: format(x, '020b'))

        if detide_mode:
            #merge same columns (if with or without tide to one column)
            relev_qc_tests = list(self.bitmask_def.keys())
            relev_qc_tests = [test for test in relev_qc_tests if not test.endswith("_detided")]
            relev_qc_tests = [x for x in relev_qc_tests if x not in ['probably_good_data', 'incorrect_format', 'missing_data']]
            for elem in relev_qc_tests:
                #Extract relevant columns (detided and tided) for each QC test
                column_with_tide = self.bitmask_def[f'{elem}']
                column_without_tide = self.bitmask_def[f'{elem}{suffix}']
                filtered_df = df[[f'bit_{column_with_tide}',f'bit_{column_without_tide}{suffix}']]
                #save how the test fail respectively for tided or detided series
                positives_tidal = (df[f'bit_{column_with_tide}'] != 0).sum()
                positives_detided = (df[f'bit_{column_without_tide}{suffix}'] != 0).sum()
                print(f'The QC test {elem} fails {positives_tidal} times for measurement series with tides and {positives_detided} times for detided data. This is a difference of {positives_tidal-positives_detided} flagged entries.')
                information.append([f'The QC test {elem} fails {positives_tidal} times for measurement series with tides and {positives_detided} times for detided data. This is a difference of {positives_tidal-positives_detided} flagged entries.'])
                #merge columns to assign flags irrespectively if test fails for detided or for tidal measurements (this overwrites the original bits!)
                df[f'bit_{column_with_tide}'] = filtered_df.max(axis=1).replace(column_without_tide, column_with_tide)
        
            #Overwrite 'combined_int' by numbers representing merged columns for the same test based on detided and tided data
            bit_dfs = df.filter(regex='^bit_')
            cols_to_drop = bit_dfs.filter(regex=f'^bit_.*{suffix}$').columns
            bit_dfs = bit_dfs.drop(columns=cols_to_drop)
            df['combined_int'] = np.bitwise_or.reduce(bit_dfs, axis=1)

        return df

    def convert_bitmask_to_flags(self, df):
        """
        Convert bits to QC flags as defined in config.json.

        Input:
        -Main dataframe [pandas df]
        """

        #Order is important
        df['quality_flag'] = df['combined_int'].apply(lambda x: self.flag_def["good_data"] if x == 0 else 0) #flag good data
        df['quality_flag'] = df.apply(lambda row: self.flag_def["probably_good_data"] if (row['combined_int'] & self.bitmask_def[f'probably_good_data']) == self.bitmask_def[f'probably_good_data'] else row['quality_flag'], axis=1) #flag probably good data

        df['quality_flag'] = df.apply(lambda row: self.flag_def["bad_data_correctable"] if (row['combined_int'] & self.bitmask_def[f'noisy_period']) == self.bitmask_def[f'noisy_period'] else row['quality_flag'], axis=1) #flag bad correctable data (noisy period)
        df['quality_flag'] = df.apply(lambda row: self.flag_def["bad_data_correctable"] if (row['combined_int'] & self.bitmask_def[f'bad_segment']) == self.bitmask_def[f'bad_segment'] else row['quality_flag'], axis=1) #flag bad correctable data (bad segment)

        df['quality_flag'] = df.apply(lambda row: self.flag_def["bad_data"] if (row['combined_int'] & self.bitmask_def[f'global_outliers']) == self.bitmask_def[f'global_outliers'] else row['quality_flag'], axis=1)  #flag bad not correctable data (global outliers)
        df['quality_flag'] = df.apply(lambda row: self.flag_def["bad_data"] if (row['combined_int'] & self.bitmask_def[f'incorrect_format']) == self.bitmask_def[f'incorrect_format'] else row['quality_flag'], axis=1) #flag bad not correctable data (incorrect format)
        
        df['quality_flag'] = df.apply(lambda row: self.flag_def["spikes"] if (row['combined_int'] & self.bitmask_def[f'spikes']) == self.bitmask_def[f'spikes'] else row['quality_flag'], axis=1) #flag spikes
        df['quality_flag'] = df.apply(lambda row: self.flag_def["shifted_value"] if (row['combined_int'] & self.bitmask_def[f'shifted_value']) == self.bitmask_def[f'shifted_value'] else row['quality_flag'], axis=1) #flag shifted value     
        
        df['quality_flag'] = df.apply(lambda row: self.flag_def["stuck_value"] if (row['combined_int'] & self.bitmask_def[f'stuck_value']) == self.bitmask_def[f'stuck_value'] else row['quality_flag'], axis=1) #flag stuck value
        df['quality_flag'] = df.apply(lambda row: self.flag_def["interpolated_value"] if (row['combined_int'] & self.bitmask_def[f'interpolated_value']) == self.bitmask_def[f'interpolated_value'] else row['quality_flag'], axis=1) #flag interpolated value
        df['quality_flag'] = df.apply(lambda row: self.flag_def["missing_data"] if (row['combined_int'] & self.bitmask_def[f'missing_data']) == self.bitmask_def[f'missing_data'] else row['quality_flag'], axis=1) #flag missing data
 
        #Raise error if any 0 in 'quality check' column (this cannot ne happening as all points are tested)
        #0 = no QC
        if (df['quality_flag']==0).any():
            raise Exception('There are columns with flag 0. However, this cannot be as all values are quality checked.')
        return df

    def save_flagged_series(self, df, meas_col_name, time_column, short_df):
        """
        Save outputs from QC steps and QC flags to csv.

        Input:
        -Main dataframe [pandas df]
        -Column name of measurement series of interest [str]
        -Column name of time & date information [str]
        -Original dataframe [pandas df]
        """
        #Extract relevant columns for export and reset timestamp
        checked_df = df[[time_column, meas_col_name, 'quality_flag','combined_bitmask']]
        relev_short_df = short_df[[time_column, meas_col_name]]
        export_df = pd.merge(checked_df, relev_short_df, on=[time_column, meas_col_name], how='right')

        #Save df to csv
        file_name = f"{self.station}_quality_checked_WL.csv"
        export_df.to_csv(os.path.join(self.folder_path, file_name), index=False)
