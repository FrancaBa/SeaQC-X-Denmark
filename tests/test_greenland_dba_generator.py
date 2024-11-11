####################################################################
## Test conversion DTU Space measurements to .dba script test by frb ##
####################################################################
import os, sys
from pathlib import Path
import shutil
import unittest
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import source.DTU_space_data_to_dba as data_converter


class Test_DTUSpace_DataConverter(unittest.TestCase):

    #Always run for each test
    def setUp(self):

        #potential stations 
        self.stations = ['Qaqortoq', 'Ittoqqortoormiit', 'Nuuk', 'Nuuk1', 'Pituffik', 'Upernavik']

        #get measurements from folder
        self.datadir = '/dmidata/users/frb/DTU__GL_data_2024'

    def test_data_Upernavik_conversion(self):
       
        print(os.getcwd())

        #select the station (here: Upernavik)
        index_station=5
        station = self.stations[index_station]

        #select output folder
        output_path = os.path.join(os.getcwd(),'output', station)
        if Path(output_path).exists(): shutil.rmtree(Path(output_path))

        data_conversion = data_converter.DataConverter()
        data_conversion.set_output_path(output_path)
        data_conversion.load_data_path(self.datadir)
        data_conversion.set_relev_station(station)
        data_conversion.run()

if __name__ == '__main__':
    unittest.main()