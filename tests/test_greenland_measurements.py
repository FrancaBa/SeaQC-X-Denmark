####################################################################
## Test quality check script for sea level tide gauge data by frb ##
####################################################################
import sys
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

    def test_quality_check_qaqortoq(self):

        #select the station (here: Qaqortoq)
        index_station=0 
        station = self.stations[index_station]
        sta_filename = station +'.dba'
        datadir = '/dmidata/users/frb/greenland_data_raw/Collected_raw'

        preprocessing = pre_proc.PreProcessor()
        preprocessing.read_data(datadir, sta_filename)
        preprocessing.check_timestamp()
        preprocessing.remove_stat_outliers()

if __name__ == '__main__':
    unittest.main()