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

        #Set path to measurements
        self.datadir = '/net/isilon/ifs/arch/home/ocean/sealevel/WL_data'
        self.missing_meas_value = -9.99 #in dataset -999, but this has to be m and not cm

        #Set path to config json
        self.json_path = os.path.join(os.getcwd(), 'config.json')
        self.gauge_details_path = os.path.join(os.getcwd(), 'tides.local')

    def test_quality_check_aabenraa(self):

        #select the station (here: Aabenraa)
        station = 'Aabenraa'
        sta_filename = 'WL_Aabenraa_26239.txt'
        print(os.getcwd())

        #select output folder
        output_path = os.path.join(os.getcwd(),'output', station)
        if Path(output_path).exists(): shutil.rmtree(Path(output_path))

        data_flagging = qc_generator.QualityFlagger()
        data_flagging.set_output_folder(output_path)
        data_flagging.load_config_json(self.json_path)
        data_flagging.set_column_names('Timestamp', 'Height', 'Flag')
        data_flagging.set_station(station)
        data_flagging.set_gauge_details(self.gauge_details_path)
        data_flagging.set_missing_value_filler(self.missing_meas_value)
        data_flagging.import_data(self.datadir, sta_filename)
        data_flagging.run()

    def test_quality_check_aarhus(self):

        #select the station (here: Aabenraa)
        station = 'Aarhus'
        sta_filename = 'WL_Aarhus_22331.txt'
        print(os.getcwd())

        #select output folder
        output_path = os.path.join(os.getcwd(),'output', station)
        if Path(output_path).exists(): shutil.rmtree(Path(output_path))

        data_flagging = qc_generator.QualityFlagger()
        data_flagging.set_output_folder(output_path)
        data_flagging.load_config_json(self.json_path)
        data_flagging.set_column_names('Timestamp', 'Height', 'Flag')
        data_flagging.set_station(station)
        data_flagging.set_gauge_details(self.gauge_details_path)
        data_flagging.set_missing_value_filler(self.missing_meas_value)
        data_flagging.import_data(self.datadir, sta_filename)
        data_flagging.run()



if __name__ == '__main__':
    unittest.main()