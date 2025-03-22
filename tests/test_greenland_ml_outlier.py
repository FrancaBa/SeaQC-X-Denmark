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

    def test_quality_check_upernavik1(self):
        #select the station (here: Upernavik 2023)
        station = 'Upernavik1'
        #station = 'Upernavik'

        #select output folder
        output_path = os.path.join(os.getcwd(),'output', 'ml_classes', station)
        if Path(output_path).exists(): shutil.rmtree(Path(output_path))

        data_flagging_ml = qc_ml_detector.MLOutlierDetection()
        data_flagging_ml.set_output_folder(output_path)
        data_flagging_ml.set_column_names('Timestamp', self.param, 'label')
        data_flagging_ml.set_station(station)
        data_flagging_ml.set_tidal_components_file(self.datadir_tides)
        data_flagging_ml.import_data(self.datadir)
        data_flagging_ml.run()
        #data_flagging_ml.run_unsupervised()
    

    def test_quality_check_Nuuk1(self):
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
        data_flagging_ml.import_data(self.datadir)
        data_flagging_ml.run()
        #data_flagging_ml.run_unsupervised()