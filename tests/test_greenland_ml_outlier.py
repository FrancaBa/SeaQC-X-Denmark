########################################################################################
## Test quality check script for sea level tide gauge data by frb for GronSL (2024/25)##
## This script runs the QC for all 4/5 stations (1 method for each station)           ##
########################################################################################

import os, sys
from pathlib import Path
import shutil
import unittest
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import source.qc_ml_detection as qc_ml_detector
import source.qc_ml_unsupervised_detection as qc_ml_unsupervised_detector


class Test_QC_ML_Station(unittest.TestCase):

    #Always run for each test
    def setUp(self):
        #All existing measurements in Greenland
        self.stations = ['Qaqortoq', 'Ittoqqortoormiit', 'Nuuk', 'Nuuk1', 'Pituffik', 'Upernavik1', 'Upernavik2'] 

        #Set path to measurements
        #self.datadir = '/dmidata/users/frb/greenland_data_raw/manual_labelled_GL_data/double_checked_labelled'
        self.datadir = '/home/frb/Documents/Franca_Project/double_checked_labelled'
        self.datadir_tides = os.path.join(os.getcwd(), 'tests', 'tidal_information')

        #Set other parameter accordingly
        self.missing_meas_value = 999.999 #999.000
        self.param = 'value'

        #Set path to config json and tidal constituents
        self.json_path = os.path.join(os.getcwd(), 'config.json')

    def test_quality_check_Upernavik(self):
        #select the station (here: Upernavik 2023)
        #station = 'Upernavik1'
        station = 'Upernavik2'
        #station = 'Upernavik'

        #select output folder
        output_path = os.path.join(os.getcwd(),'output', 'ml_classes', station)
        if Path(output_path).exists(): shutil.rmtree(Path(output_path))

        data_flagging_ml = qc_ml_detector.MLOutlierDetection()
        data_flagging_ml.set_output_folder(output_path)
        data_flagging_ml.set_column_names('Timestamp', self.param, 'label')
        data_flagging_ml.set_station(station)
        data_flagging_ml.set_tidal_components_file(self.datadir_tides)
        dfs_station_subsets, dfs_training = data_flagging_ml.import_data(self.datadir)
        #For running one specific file connected to a station as training data (choose station accordingl f.e. Upernavik1)
        df_dict, df_test = data_flagging_ml.run(dfs_station_subsets)
        #For running all files connected to a station as training data (choose station accordingly f.e. Upernavik)
        #df_dict, df_test = data_flagging_ml.run(dfs_training)
        outcomes = data_flagging_ml.run_testing(df_dict, df_test)

    def test_quality_check_Nuuk(self):
        #select the station (here: Nuuk)
        station = 'Nuuk'

        #select output folder
        output_path = os.path.join(os.getcwd(),'output', 'ml_classes', station)
        if Path(output_path).exists(): shutil.rmtree(Path(output_path))

        data_flagging_ml = qc_ml_detector.MLOutlierDetection()
        data_flagging_ml.set_output_folder(output_path)
        data_flagging_ml.set_column_names('Timestamp', self.param, 'label')
        data_flagging_ml.set_station(station)
        data_flagging_ml.set_tidal_components_file(self.datadir_tides)
        dfs_station_subsets, dfs_training = data_flagging_ml.import_data(self.datadir)
        #For running one specific file connected to a station as training data (choose station accordingly)
        #df_dict, df_test = data_flagging_ml.run(dfs_station_subsets)
        #For running all files connected to a station as training data (choose station accordingly)
        df_dict, df_test = data_flagging_ml.run(dfs_training)
        outcomes = data_flagging_ml.run_testing(df_dict, df_test)


    def backup_test_quality_check_Qaqortoq(self):
        #select the station (here: Qaqortoq)
        station = 'Qaqortoq'

        #select output folder
        output_path = os.path.join(os.getcwd(),'output', 'ml_classes', station)
        if Path(output_path).exists(): shutil.rmtree(Path(output_path))

        data_flagging_ml = qc_ml_detector.MLOutlierDetection()
        data_flagging_ml.set_output_folder(output_path)
        data_flagging_ml.set_column_names('Timestamp', self.param, 'label')
        data_flagging_ml.set_station(station)
        data_flagging_ml.set_tidal_components_file(self.datadir_tides)
        dfs_station_subsets, dfs_training = data_flagging_ml.import_data(self.datadir)
        #For running one specific file connected to a station as training data (choose station accordingly)
        df_dict, df_test = data_flagging_ml.run(dfs_station_subsets)
        #For running all files connected to a station as training data (choose station accordingly)
        #df_dict, df_test = data_flagging_ml.run(dfs_training)
        outcomes = data_flagging_ml.run_testing(df_dict, df_test)


    def test_quality_check_Pituffik(self):
        #select the station (here: Pituffik)
        #station = 'Pituffik'
        #station = 'Pituffik1'
        #station = 'Pituffik2'
        station = 'Pituffik3'

        #select output folder
        output_path = os.path.join(os.getcwd(),'output', 'ml_classes', station)
        if Path(output_path).exists(): shutil.rmtree(Path(output_path))

        data_flagging_ml = qc_ml_detector.MLOutlierDetection()
        data_flagging_ml.set_output_folder(output_path)
        data_flagging_ml.set_column_names('Timestamp', self.param, 'label')
        data_flagging_ml.set_station(station)
        data_flagging_ml.set_tidal_components_file(self.datadir_tides)
        dfs_station_subsets, dfs_training = data_flagging_ml.import_data(self.datadir)
        #For running all files connected to a station as training data (choose station accordingly)
        df_dict, df_test = data_flagging_ml.run(dfs_station_subsets)
        #For running one specific file connected to a station as training data (choose station accordingly)
        #df_dict, df_test = data_flagging_ml.run(dfs_training)
        outcomes = data_flagging_ml.run_testing(df_dict, df_test)


    def test_quality_check_all(self):
        #select the station (here: Pituffik)
        station = 'combined'

        #select output folder
        output_path = os.path.join(os.getcwd(),'output', 'ml_classes', station)
        if Path(output_path).exists(): shutil.rmtree(Path(output_path))

        data_flagging_ml = qc_ml_detector.MLOutlierDetection()
        data_flagging_ml.set_output_folder(output_path)
        data_flagging_ml.set_column_names('Timestamp', self.param, 'label')
        data_flagging_ml.set_station(station)
        data_flagging_ml.set_tidal_components_file(self.datadir_tides)
        dfs_station_subsets, dfs_training = data_flagging_ml.import_data(self.datadir)
        #Training data equals to X% of manual labelled data from all the files for ALL the stations combined (f.e: 50% of labelled data from each station is used for training)
        dfs_testing_new = data_flagging_ml.run_combined_training(dfs_training)
        outcomes = data_flagging_ml.run_testing(dfs_testing_new)

    def test_quality_check_unsupervised(self):
        station = 'unsupervised_learning'

        #select output folder
        output_path = os.path.join(os.getcwd(),'output', 'ml_classes', station)
        if Path(output_path).exists(): shutil.rmtree(Path(output_path))

        data_flagging_ml = qc_ml_unsupervised_detector.MLOutlierDetectionUNSU()
        data_flagging_ml.set_output_folder(output_path)
        data_flagging_ml.set_column_names('Timestamp', self.param, 'label')
        data_flagging_ml.set_station(station)
        data_flagging_ml.set_tidal_components_file(self.datadir_tides)
        data_flagging_ml.import_data(self.datadir)
        data_flagging_ml.run_unsupervised()