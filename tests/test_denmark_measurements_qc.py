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
        self.datadir = '/dmidata/users/frb/QC Work/2_Greenland&Denmark/Input_data/temporal/raw_data'
        self.missing_meas_value = -9.99 #in dataset -999, but this has to be in m and not cm

        #Set path to config json and and labelled data
        self.json_path = os.path.join(os.getcwd(), 'config.json')
        self.gauge_details_path = os.path.join(os.getcwd(), 'tides.local')
        self.datadir_labels = '/dmidata/users/frb/QC Work/2_Greenland&Denmark/training_data/Training_data(Franca_labelled_DKandGL))'
        self.datadir_tides_ml = '/dmidata/users/frb/QC Work/tidal_information'
        self.datadir_tides = '/dmidata/users/frb/QC Work/tidal_information'

        #Define classes which mark bad data points (for now following IOC in manual labelled data: 3 and 4 (bad)); QC class 0 is currently hard coded to be 'good'
        self.qc_classes = [3,4]

    def test_quality_check_Frederikshavn(self):

        #select the station (here: Frederikshavn)
        station = 'Frederikshavn'
        sta_filename = '20101.seadb_0'
        print(os.getcwd())

        #select output folder
        output_path = os.path.join(os.getcwd(),'output', station)
        if Path(output_path).exists(): shutil.rmtree(Path(output_path))

        #Set path to tidal signal file
        data_tide = os.path.join(self.datadir_tides, 'Frederikshavn.tmp')

        data_flagging = qc_generator.QualityFlagger()
        data_flagging.set_output_folder(output_path)
        data_flagging.load_config_json(self.json_path)
        data_flagging.set_column_names('Timestamp', 'Height', 'Flag')
        data_flagging.set_station(station)
        data_flagging.set_gauge_details(self.gauge_details_path)
        data_flagging.set_missing_value_filler(self.missing_meas_value)
        data_flagging.set_tide_series(data_tide)
        data_flagging.set_qcclasses(self.qc_classes)
        data_flagging.set_ml_training_data(self.datadir_labels, self.datadir_tides_ml)
        data_flagging.import_data(self.datadir, sta_filename)
        data_flagging.run()

    #Main Method used for second QC Paper
    def test_quality_check_DMI(self):

        #select the relevant data
        #Link to data needed to run spatial testing data
        self.datadir = '/dmidata/users/frb/QC Work/2_Greenland&Denmark/Input_data/spatial/raw_data'
        #Link to data needed to run temporal testing data
        #self.datadir = '/dmidata/users/frb/QC Work/2_Greenland&Denmark/Input_data/temporal/raw_data'

        #select the station
        station = 'Hirtshals'
        sta_filename = '20047.seadb_0'
        print(os.getcwd())

        #select output folder
        output_path = os.path.join(os.getcwd(), 'output', station)
        if Path(output_path).exists(): shutil.rmtree(Path(output_path))

        #Set path to tidal signal file
        data_tide = os.path.join(self.datadir_tides, 'Hirtshals.tmp')

        data_flagging = qc_generator.QualityFlagger()
        data_flagging.set_output_folder(output_path)
        data_flagging.load_config_json(self.json_path)
        data_flagging.set_column_names('Timestamp', 'Height', 'Flag')
        data_flagging.set_station(station)
        data_flagging.set_gauge_details(self.gauge_details_path)
        data_flagging.set_missing_value_filler(self.missing_meas_value)
        data_flagging.set_tide_series(data_tide)
        data_flagging.set_qcclasses(self.qc_classes)
        data_flagging.set_ml_training_data(self.datadir_labels, self.datadir_tides)
        data_flagging.import_data(self.datadir, sta_filename)
        data_flagging.run()

    def test_quality_check_Kobenhavn(self):
        #Set path to measurements
        self.datadir = '/dmidata/users/frb/QC Work/2_Greenland&Denmark/Input_data/temporal/raw_data'

        #select the station (here: Kobenhavn)
        station = 'Kobenhavn'
        sta_filename = '30336.seadb_0'
        print(os.getcwd())

        #select output folder
        output_path = os.path.join(os.getcwd(),'output', station)
        if Path(output_path).exists(): shutil.rmtree(Path(output_path))

        #Set path to tidal signal file - tidal signal is missing
        data_tide = os.path.join(self.datadir_tides, 'Kobenhavn.tmp')

        data_flagging = qc_generator.QualityFlagger()
        data_flagging.set_output_folder(output_path)
        data_flagging.load_config_json(self.json_path)
        data_flagging.set_column_names('Timestamp', 'Height', 'Flag')
        data_flagging.set_station(station)
        data_flagging.set_gauge_details(self.gauge_details_path)
        data_flagging.set_missing_value_filler(self.missing_meas_value)
        data_flagging.set_tide_series(data_tide)
        data_flagging.set_qcclasses(self.qc_classes)
        data_flagging.set_ml_training_data(self.datadir_labels, self.datadir_tides_ml)
        data_flagging.import_data(self.datadir, sta_filename)
        data_flagging.run()

    def test_quality_check_several(self):

        #Set path to measurements
        self.datadir = '/dmidata/users/frb/QC Work/2_Greenland&Denmark/Input_data/temporal/raw_data'

        #list of station for QC        
        stations = ['Hornbaek', 'Esbjerg', 'Gedser', 'Tejn']
        sta_filename_all = ['30336.seadb_0', '25149.seadb_0', '31616.seadb_0', '32048.seadb_0']
        tidal_signals = ['Hornbaek.tmp', 'Esbjerg.tmp', 'Gedser.tmp', 'Tejn.tmp']

        for i in range(0,len(stations)):
            sta_filename = sta_filename_all[i]
            station = stations[i]
            tidal_signal = tidal_signals[i]
            print(station)

            #select output folder
            output_path = os.path.join(os.getcwd(),'output', station)
            if Path(output_path).exists(): shutil.rmtree(Path(output_path))

            #Set path to tidal signal file
            data_tide = os.path.join(self.datadir_tides, tidal_signal)

            data_flagging = qc_generator.QualityFlagger()
            data_flagging.set_output_folder(output_path)
            data_flagging.load_config_json(self.json_path)
            data_flagging.set_column_names('Timestamp', 'Height', 'Flag')
            data_flagging.set_station(station)
            data_flagging.set_gauge_details(self.gauge_details_path)
            data_flagging.set_missing_value_filler(self.missing_meas_value)
            data_flagging.set_tide_series(data_tide)
            data_flagging.set_qcclasses(self.qc_classes)
            data_flagging.set_ml_training_data(self.datadir_labels, self.datadir_tides_ml)
            data_flagging.import_data(self.datadir, sta_filename)
            data_flagging.run()


if __name__ == '__main__':
    unittest.main()