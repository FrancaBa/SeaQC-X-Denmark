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

import source.quality_checker as qc_generator

class Test_QA_Station(unittest.TestCase):

    #Always run for each test
    def setUp(self):
        #All existing measurements in Greenland
        self.stations = ['Qaqortoq', 'Ittoqqortoormiit', 'Nuuk', 'Nuuk1', 'Pituffik', 'Upernavik1', 'Upernavik2'] 

        #Coordinates of stations
        coord = {
            'Qaqortoq': (60.7, -46),
            'Ittoqqortoormiit': (70.48, -21.98),
            'Nuuk': (64.16, -51.73),
            'Nuuk1': (64.16, -51.73),
            'Pituffik': (76.54, -68.8626),
            'Upernavik1': (np.nan, np.nan),
            'Upernavik2': (np.nan, np.nan)
        }

        #Generates a dic containing all stations
        dic_stations = {station: coord[station] for station in self.stations}
        print(dic_stations)

        #Set path to measurements
        self.datadir = '/dmidata/users/frb/greenland_data_raw/Collected_Oct2024'
        #self.datadir = '/dmidata/users/frb/greenland_data_raw/Collected_raw'

        #Set other parameter accordingly
        if self.datadir.endswith('Collected_Oct2024'):
            self.missing_meas_value = 999.999
            self.ending = '_data.csv'
            self.param = 'Height'

        if self.datadir.endswith('Collected_raw'):
            self.missing_meas_value = 999.000
            self.ending = '.dba'
            self.param = 'WaterLevel'

        #Set path to config json
        self.json_path = os.path.join(os.getcwd(), 'config.json')

    def test_quality_check_qaqortoq(self):

        #select the station (here: Qaqortoq)
        index_station=0 
        station = self.stations[index_station]
        sta_filename = station + self.ending
        
        print(os.getcwd())

        #select output folder
        output_path = os.path.join(os.getcwd(),'output', station)
        if Path(output_path).exists(): shutil.rmtree(Path(output_path))

        data_flagging = qc_generator.QualityFlagger()
        data_flagging.set_output_folder(output_path)
        data_flagging.load_config_json(self.json_path)
        data_flagging.set_column_names('Timestamp', self.param, 'Flag')
        data_flagging.set_station(station)
        data_flagging.set_missing_value_filler(self.missing_meas_value)
        data_flagging.import_data(self.datadir, sta_filename)
        data_flagging.run()

    def test_quality_check_ittoq(self):

        #select the station (here: Ittoqqortoormiit)
        index_station=1
        station = self.stations[index_station]
        sta_filename = station + self.ending
        
        print(os.getcwd())

        #select output folder
        output_path = os.path.join(os.getcwd(),'output', station)
        if Path(output_path).exists(): shutil.rmtree(Path(output_path))

        data_flagging = qc_generator.QualityFlagger()
        data_flagging.set_output_folder(output_path)
        data_flagging.load_config_json(self.json_path)
        data_flagging.set_column_names('Timestamp', self.param, 'Flag')
        data_flagging.set_station(station)
        data_flagging.set_missing_value_filler(self.missing_meas_value)
        data_flagging.import_data(self.datadir, sta_filename)
        data_flagging.run()
    
    def test_quality_check_nuuk(self):

        #select the station (here: Nuuk)
        index_station=2
        station = self.stations[index_station]
        sta_filename = station + self.ending

        print(os.getcwd())

        #select output folder
        output_path = os.path.join(os.getcwd(),'output', station)
        if Path(output_path).exists(): shutil.rmtree(Path(output_path))

        data_flagging = qc_generator.QualityFlagger()
        data_flagging.set_output_folder(output_path)
        data_flagging.load_config_json(self.json_path)
        data_flagging.set_column_names('Timestamp', self.param, 'Flag')
        data_flagging.set_station(station)
        data_flagging.set_missing_value_filler(self.missing_meas_value)
        data_flagging.import_data(self.datadir, sta_filename)
        data_flagging.run()

    def test_quality_check_nuuk1(self):

        #select the station (here: Nuuk1)
        index_station=3
        station = self.stations[index_station]
        sta_filename = station + self.ending

        print(os.getcwd())

        #select output folder
        output_path = os.path.join(os.getcwd(),'output', station)
        if Path(output_path).exists(): shutil.rmtree(Path(output_path))

        data_flagging = qc_generator.QualityFlagger()
        data_flagging.set_output_folder(output_path)
        data_flagging.load_config_json(self.json_path)
        data_flagging.set_column_names('Timestamp', self.param, 'Flag')
        data_flagging.set_station(station)
        data_flagging.set_missing_value_filler(self.missing_meas_value)
        data_flagging.import_data(self.datadir, sta_filename)
        data_flagging.run()

    def test_quality_check_pituffik(self):

        #select the station (here: Pituffik)
        index_station=4
        station = self.stations[index_station]
        sta_filename = station + self.ending

        print(os.getcwd())

        #select output folder
        output_path = os.path.join(os.getcwd(),'output', station)
        if Path(output_path).exists(): shutil.rmtree(Path(output_path))

        data_flagging = qc_generator.QualityFlagger()
        data_flagging.set_output_folder(output_path)
        data_flagging.load_config_json(self.json_path)
        data_flagging.set_column_names('Timestamp', self.param, 'Flag')
        data_flagging.set_station(station)
        data_flagging.set_missing_value_filler(self.missing_meas_value)
        data_flagging.import_data(self.datadir, sta_filename)
        data_flagging.run()

    def test_quality_check_upernavik1(self):

        #select the station (here: Upernavik 2023)
        index_station=5
        station = self.stations[index_station]
        sta_filename = station + self.ending

        print(os.getcwd())

        #select output folder
        output_path = os.path.join(os.getcwd(),'output', station)
        if Path(output_path).exists(): shutil.rmtree(Path(output_path))

        data_flagging = qc_generator.QualityFlagger()
        data_flagging.set_output_folder(output_path)
        data_flagging.load_config_json(self.json_path)
        data_flagging.set_column_names('Timestamp', self.param, 'Flag')
        data_flagging.set_station(station)
        data_flagging.set_missing_value_filler(self.missing_meas_value)
        data_flagging.import_data(self.datadir, sta_filename)
        data_flagging.run()

    def test_quality_check_upernavik2(self):

        #select the station (here: Upernavik 2024)
        index_station=6
        station = self.stations[index_station]
        sta_filename = station + self.ending

        print(os.getcwd())

        #select output folder
        output_path = os.path.join(os.getcwd(),'output', station)
        if Path(output_path).exists(): shutil.rmtree(Path(output_path))

        data_flagging = qc_generator.QualityFlagger()
        data_flagging.set_output_folder(output_path)
        data_flagging.load_config_json(self.json_path)
        data_flagging.set_column_names('Timestamp', self.param, 'Flag')
        data_flagging.set_station(station)
        data_flagging.set_missing_value_filler(self.missing_meas_value)
        data_flagging.import_data(self.datadir, sta_filename)
        data_flagging.run()


if __name__ == '__main__':
    unittest.main()