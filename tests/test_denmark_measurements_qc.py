########################################################################################
## Test quality check script for sea level tide gauge data by frb for GronSL (2024/25)##
## This script runs the QC for some Danish Stations as test (1 test per station)      ##
########################################################################################

import os
from pathlib import Path
import shutil
import unittest

import source.main as qc_generator

class Test_QC_Station(unittest.TestCase):

    #Always run for each test
    def setUp(self):

        #Set path to measurements
        self.datadir = '/net/isilon/ifs/arch/home/ocean/sealevel/WL_data'
        self.missing_meas_value = -9.99 #in dataset -999, but this has to be in m and not cm

        #Set path to config json and and labelled data
        self.json_path = os.path.join(os.getcwd(), 'config.json')
        self.gauge_details_path = os.path.join(os.getcwd(), 'tides.local')
        self.datadir_labels = 'TBD'

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
        data_flagging.set_training_data(self.datadir_labels)
        data_flagging.import_data(self.datadir, sta_filename)
        data_flagging.run()

    def test_quality_check_aarhus(self):

        #select the station (here: Aarhus)
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
        data_flagging.set_training_data(self.datadir_labels)
        data_flagging.import_data(self.datadir, sta_filename)
        data_flagging.run()

    def test_quality_check_esbjerg(self):

        #select the station (here: Esbjerg)
        station = 'Esbjerg'
        sta_filename = 'WL_Esbjerg_25149.txt'
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
        data_flagging.set_training_data(self.datadir_labels)
        data_flagging.import_data(self.datadir, sta_filename)
        data_flagging.run()

    def test_quality_check_koege(self):

        #select the station (here: Koege)
        station = 'KogeHavn'
        sta_filename = 'WL_Koege_30479.txt'
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
        data_flagging.set_training_data(self.datadir_labels)
        data_flagging.import_data(self.datadir, sta_filename)
        data_flagging.run()

    def test_quality_check_tejn(self):

        #select the station (here: Tejn)
        station = 'Tejn'
        sta_filename = 'WL_Tejn_32048.txt'
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
        data_flagging.set_training_data(self.datadir_labels)
        data_flagging.import_data(self.datadir, sta_filename)
        data_flagging.run()

    def test_quality_check_roskilde(self):

        #select the station (here: Roskilde)
        station = 'Roskilde'
        sta_filename = 'WL_Roskilde_30407.txt'
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
        data_flagging.set_training_data(self.datadir_labels)
        data_flagging.import_data(self.datadir, sta_filename)
        data_flagging.run()

    def test_quality_check_slipshavn(self):

        #select the station (here: Slipshavn)
        station = 'Slipshavn'
        sta_filename = 'WL_Slipshavn_28234.txt'
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
        data_flagging.set_training_data(self.datadir_labels)
        data_flagging.import_data(self.datadir, sta_filename)
        data_flagging.run()

    def test_quality_check_rodby(self):

        #select the station (here: Rodby)
        station = 'Rodby'
        sta_filename = 'WL_Rodby_31573.txt'
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
        data_flagging.set_training_data(self.datadir_labels)
        data_flagging.import_data(self.datadir, sta_filename)
        data_flagging.run()



if __name__ == '__main__':
    unittest.main()