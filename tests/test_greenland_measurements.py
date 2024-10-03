####################################################################
## Test quality check script for sea level tide gauge data by frb ##
####################################################################
import os, sys
from pathlib import Path
import shutil
import unittest
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import source.preprocessing as pre_proc

class Test_QA_Station(unittest.TestCase):

    #Always run for each test
    def setUp(self):
        #All existing measurements in Greenland
        self.stations = ['Qaqortoq', 'Ittoqqortoormiit', 'Nuuk', 'Pituffik', 'Upernavik'] 

        # Coordinates of stations
        coord = {
            'Qaqortoq': (60.7, -46),
            'Ittoqqortoormiit': (70.48, -21.98),
            'Nuuk': (64.16, -51.73),
            'Pituffik': (76.54, -68.8626),
            'Upernavik': (np.nan, np.nan),
        }

        # dic containing all stations
        dic_stations = {station: coord[station] for station in self.stations}
        print(dic_stations)

        #get measurements
        self.datadir = '/dmidata/users/frb/greenland_data_raw/Collected_raw'

    def test_quality_check_qaqortoq(self):

        #select the station (here: Qaqortoq)
        index_station=0 
        station = self.stations[index_station]
        sta_filename = station +'.dba'

        print(os.getcwd())

        #select output folder
        output_path = os.path.join(os.getcwd(),'output', station)
        if Path(output_path).exists(): shutil.rmtree(Path(output_path))

        preprocessing = pre_proc.PreProcessor()
        preprocessing.set_output_folder(output_path)
        preprocessing.read_data(self.datadir, sta_filename)
        preprocessing.check_timestamp()
        preprocessing.remove_stat_outliers()
        preprocessing.remove_spikes()

    def test_quality_check_ittoq(self):

        #select the station (here: Ittoqqortoormiit)
        index_station=1
        station = self.stations[index_station]
        sta_filename = station +'.dba'

        print(os.getcwd())

        #select output folder
        output_path = os.path.join(os.getcwd(),'output', station)
        if Path(output_path).exists(): shutil.rmtree(Path(output_path))

        preprocessing = pre_proc.PreProcessor()
        preprocessing.set_output_folder(output_path)
        preprocessing.read_data(self.datadir, sta_filename)
        preprocessing.check_timestamp()
        preprocessing.remove_stat_outliers()
        preprocessing.remove_spikes()

if __name__ == '__main__':
    unittest.main()