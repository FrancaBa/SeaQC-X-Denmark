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
        self.datadir = os.path.join(os.getcwd(), 'data/Training_data(labelled_DKandGL)')

        #Set path to folder where tidal signal for all stations is saved as it is needed as a feature for ML algorithm
        self.datadir_tides = os.path.join(os.getcwd(), 'data/tidal_information')

        #Set other parameter accordingly
        self.missing_meas_value = 999
        self.param = 'value'

        #Set path to config json and tidal constituents
        self.json_path = os.path.join(os.getcwd(), 'config.json')

        #Define classes which mark bad data points (for now following IOC in manual labelled data: 3 (probably bad) and 4 (bad)); QC class 1 is currentlt hard coded to be 'good'
        self.qc_classes = [3,4]

    def test_quality_check_ml_DK(self):
        
        station = 'Greenland&DK'

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