###################################################################################################################################
## Test for ML small-scale detection test as part of QC algorithm developed by frb for GronSL (2024/25)                          ##
## This script runs the ML QC step based on provided input data (Tests for Denmark and Greenland)                                ##
## It runs with .csv files in trainset-formate (https://trainset.geocene.com/) as that website has been used for manual labeling ##
###################################################################################################################################

import os
from pathlib import Path
import shutil
import unittest

import source.various_qc_tests.qc_ml_detection as qc_ml_detector

class Test_QC_ML_Station(unittest.TestCase):

    #Always run for each test
    def setUp(self):
        
        #Set path to labelled measurements which are used as training and testing data
        #BE AWARE: Depending on which dataset is used the subsequent information need to be adapted or expanded; not all unittests can be run with the same set-up 


        #For first paper: Only trained for Greenland based on training set containing flagged small scale anomaly periods in Greenland
        self.datadir = '/dmidata/users/frb/QC Work/1_Greenland/manual_labelling_files/Manual labelled training data/Correct timestamp in merged files (final)'
        #For second paper: Only analysed for Denmark, but training data is a mix of flagged small scale anomaly periods in Greenland and longer periods (2014-2018) for 6 stations in Denmark
        #self.datadir = '/dmidata/users/frb/QC Work/2_Greenland&Denmark/training_data/Training_data(Franca_labelled_DKandGL))'
        #Test with various other training data sets:
        #Add Max QC data for training (to DK and Greenland), but his quality is not good enough (as those are not his final results)
        #self.datadir = '/dmidata/users/frb/QC Work/2_Greenland&Denmark/training_data/Training_data(Franca_labelled_DKandGLandMax)'
        #Training set just consists of Max QC data, but his quality is not good enough (as those are not his final results)
        #self.datadir = '/dmidata/users/frb/QC Work/2_Greenland&Denmark/training_data/Max_labelled'
        #Training set just consists of manual labelled QC data in 6 danish stations with very strict flagging of smaller outliers 
        #self.datadir = '/dmidata/users/frb/QC Work/2_Greenland&Denmark/training_data/Danish_stations (finer_labelled)'

        #Set path to folder where tidal signal for all stations is saved as it is needed as a feature for ML algorithm
        self.datadir_tides = '/dmidata/users/frb/QC Work/tidal_information'
        
        #Set path to original measurements which include salinity, temperature and so on besides water level in order to use these information for multivariate analysis
        #This only work for Greenlandic training as test cases for now, as the Danish corresponding data is not saved
        self.datadir_measurements = '/dmidata/users/frb/QC Work/1_Greenland/greenland_data_raw/Collected_Oct2024'

        #Set other parameter accordingly
        self.missing_meas_value = 999.999 #999.000
        self.param = 'value'

        #Set path to config json and tidal constituents
        self.json_path = os.path.join(os.getcwd(), 'config.json')

        #Define classes which mark bad data points (for now following IOC in manual labelled data: 3 (probably bad) and 4 (bad)); QC class 1 is currentlt hard coded to be 'good'
        self.qc_classes = [3,4]

    def test_quality_check_Upernavik(self):
        #select the station (here: Upernavik 2023)
        #station = 'Upernavik1'
        #station = 'Upernavik2'
        station = 'Upernavik'

        #select output folder
        output_path = os.path.join(os.getcwd(),'output', 'ml_classes', station)
        if Path(output_path).exists(): shutil.rmtree(Path(output_path))

        data_flagging_ml = qc_ml_detector.MLOutlierDetection()
        data_flagging_ml.set_output_folder(output_path)
        data_flagging_ml.set_column_names('Timestamp', self.param, 'label')
        data_flagging_ml.set_station(station)
        data_flagging_ml.set_qcclasses(self.qc_classes)
        tidal_infos = data_flagging_ml.set_tidal_series_file(self.datadir_tides)
        #Comment the following line in or out to add multivariate analysis or not (Temperature does not work as it is not part of the csv file)
        #data_flagging_ml.load_multivariate_analysis(self.datadir_measurements, 'Conductivity', 'Pressure', 'Temperature')
        dfs_station_subsets, dfs_training = data_flagging_ml.import_data(self.datadir, tidal_infos)
        #For running one specific file connected to a station as training data (choose station accordingl f.e. Upernavik1)
        #df_dict, dfs_test = data_flagging_ml.run_training(dfs_station_subsets)
        #For running all files connected to a station as training data (choose station accordingly f.e. Upernavik)
        df_dict, dfs_test = data_flagging_ml.run_training(dfs_training)
        outcomes = data_flagging_ml.run_testing(df_dict, dfs_test)

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
        data_flagging_ml.set_qcclasses(self.qc_classes)
        tidal_infos = data_flagging_ml.set_tidal_series_file(self.datadir_tides)
        data_flagging_ml.load_multivariate_analysis(self.datadir_measurements, 'Conductivity', 'Pressure', 'Temperature')
        dfs_station_subsets, dfs_training = data_flagging_ml.import_data(self.datadir, tidal_infos)
        #For running all files connected to a station as training data (choose station accordingly)
        df_dict, dfs_test = data_flagging_ml.run_training(dfs_training)
        outcomes = data_flagging_ml.run_testing(df_dict, dfs_test)

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
        data_flagging_ml.set_qcclasses(self.qc_classes)
        tidal_infos = data_flagging_ml.set_tidal_series_file(self.datadir_tides)
        #data_flagging_ml.load_multivariate_analysis(self.datadir_measurements, 'Conductivity', 'Pressure', 'Temperature')
        dfs_station_subsets, dfs_training = data_flagging_ml.import_data(self.datadir, tidal_infos)
        #For running one specific file connected to a station as training data (choose station accordingly)
        df_dict, dfs_test = data_flagging_ml.run_training(dfs_station_subsets)
        #For running all files connected to a station as training data (choose station accordingly)
        #df_dict, dfs_test = data_flagging_ml.run_training(dfs_training)
        outcomes = data_flagging_ml.run_testing(df_dict, dfs_test)

    def test_quality_check_all(self):
        station = 'combined'

        #select output folder
        output_path = os.path.join(os.getcwd(),'output', 'ml_classes', station)
        if Path(output_path).exists(): shutil.rmtree(Path(output_path))

        data_flagging_ml = qc_ml_detector.MLOutlierDetection()
        data_flagging_ml.set_output_folder(output_path)
        data_flagging_ml.set_column_names('Timestamp', self.param, 'label')
        data_flagging_ml.set_station(station)
        data_flagging_ml.set_qcclasses(self.qc_classes)
        tidal_infos = data_flagging_ml.set_tidal_series_file(self.datadir_tides)
        #data_flagging_ml.load_multivariate_analysis(self.datadir_measurements, 'Conductivity', 'Pressure', 'Temperature')
        dfs_station_subsets, dfs_training = data_flagging_ml.import_data(self.datadir, tidal_infos)
        df_dict, dfs_testing = data_flagging_ml.run_training(dfs_training, combined_training = True)
        outcomes = data_flagging_ml.run_testing(df_dict, dfs_testing)

    def test_quality_check_unsupervised(self):
        station = 'unsupervised_learning'

        #select output folder
        output_path = os.path.join(os.getcwd(),'output', 'ml_classes', station)
        if Path(output_path).exists(): shutil.rmtree(Path(output_path))

        data_flagging_ml = qc_ml_detector.MLOutlierDetection()
        data_flagging_ml.set_output_folder(output_path)
        data_flagging_ml.set_column_names('Timestamp', self.param, 'label')
        data_flagging_ml.set_station(station)
        data_flagging_ml.set_qcclasses(self.qc_classes)
        tidal_infos = data_flagging_ml.set_tidal_series_file(self.datadir_tides)
        #data_flagging_ml.load_multivariate_analysis(self.datadir_measurements, 'Conductivity', 'Pressure', 'Temperature')
        dfs_station_subsets, dfs_training = data_flagging_ml.import_data(self.datadir, tidal_infos)
        df_dict, dfs_testing = data_flagging_ml.run_training(dfs_training, combined_training = True, unsupervised_model = True)
        outcomes = data_flagging_ml.run_testing(df_dict, dfs_testing)

    def test_quality_check_all_DK(self):
        station = 'Greenland&DK'

        #For second paper: Only analysed for Denmark, but training data is a mix of flagged small scale anomaly periods in Greenland and longer periods (2014-2018) for 6 stations in Denmark
        self.datadir = '/dmidata/users/frb/QC Work/2_Greenland&Denmark/training_data/Training_data(Franca_labelled_DKandGL))'

        #select output folder
        output_path = os.path.join(os.getcwd(),'output', 'ml_classes', station)
        if Path(output_path).exists(): shutil.rmtree(Path(output_path))

        data_flagging_ml = qc_ml_detector.MLOutlierDetection()
        data_flagging_ml.set_output_folder(output_path)
        data_flagging_ml.set_column_names('Timestamp', self.param, 'label')
        data_flagging_ml.set_station(station)
        data_flagging_ml.set_qcclasses(self.qc_classes)
        tidal_infos = data_flagging_ml.set_tidal_series_file(self.datadir_tides)
        dfs_station_subsets, dfs_training = data_flagging_ml.import_data(self.datadir, tidal_infos)
        df_dict, dfs_testing = data_flagging_ml.run_training(dfs_training, combined_training = True)
        #Outcomes is needed when plugged into complete QC algorithm
        outcomes = data_flagging_ml.run_testing(df_dict, dfs_testing)
        #Run complete ts (training and testing) to be controlled by ML-Algorithm and save flagged timeseries
        data_flagging_ml.save_flagged_series(dfs_station_subsets)