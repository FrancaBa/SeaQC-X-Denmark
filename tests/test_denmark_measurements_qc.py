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
        #self.datadir = '/home/frb/Documents/Franca_Project/WL_data'
        self.missing_meas_value = -9.99 #in dataset -999, but this has to be in m and not cm

        #Set path to config json and and labelled data
        self.json_path = os.path.join(os.getcwd(), 'config.json')
        self.gauge_details_path = os.path.join(os.getcwd(), 'tides.local')
        self.datadir_labels = '/home/frb/Documents/Franca_Project/double_checked_labelled/correct_date'
        self.datadir_tides_ml = '/home/frb/Documents/Franca_Project/double_checked_labelled/tidal_information'
        self.datadir_tides = '/home/frb/Documents/Franca_Project/Niels_data/tides'

    def test_quality_check_hesnaes(self):

        #select the station (here: Hesnaes)
        station = 'Hesnaes'
        sta_filename = 'WL_Hesnaes_31493.txt'
        #sta_filename = '26239.seadb_0'
        print(os.getcwd())

        #select output folder
        output_path = os.path.join(os.getcwd(),'output', station)
        if Path(output_path).exists(): shutil.rmtree(Path(output_path))

        #Set path to tidal signal file
        data_tide = os.path.join(self.datadir_tides, '31493.tmp')

        data_flagging = qc_generator.QualityFlagger()
        data_flagging.set_output_folder(output_path)
        data_flagging.load_config_json(self.json_path)
        data_flagging.set_column_names('Timestamp', 'Height', 'Flag')
        data_flagging.set_station(station)
        data_flagging.set_gauge_details(self.gauge_details_path)
        data_flagging.set_missing_value_filler(self.missing_meas_value)
        data_flagging.set_tide_series(data_tide)
        data_flagging.set_ml_training_data(self.datadir_labels, self.datadir_tides_ml)
        data_flagging.import_data(self.datadir, sta_filename)
        data_flagging.run()

    def test_quality_check_aabenraa(self):

        #select the station (here: Aabenraa)
        self.datadir_tides = '/home/frb/Documents/Franca_Project/ComparingQCMethods/tides'
        self.datadir = '/home/frb/Documents/Franca_Project/ComparingQCMethods/raw_data'
        station = 'Aabenraa'
        #sta_filename = 'WL_Aabenraa_26239.txt'
        sta_filename = '26239.seadb_0'
        print(os.getcwd())

        #select output folder
        output_path = os.path.join(os.getcwd(),'output', station)
        if Path(output_path).exists(): shutil.rmtree(Path(output_path))

        #Set path to tidal signal file - tidal signal is missing
        data_tide = os.path.join(self.datadir_tides, 'Aabenraa.tmp')

        data_flagging = qc_generator.QualityFlagger()
        data_flagging.set_output_folder(output_path)
        data_flagging.load_config_json(self.json_path)
        data_flagging.set_column_names('Timestamp', 'Height', 'Flag')
        data_flagging.set_station(station)
        data_flagging.set_gauge_details(self.gauge_details_path)
        data_flagging.set_missing_value_filler(self.missing_meas_value)
        data_flagging.set_tide_series(data_tide)
        data_flagging.set_ml_training_data(self.datadir_labels, self.datadir_tides_ml)
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

        #Set path to tidal signal file
        data_tide = os.path.join(self.datadir_tides, '22331.tmp')

        data_flagging = qc_generator.QualityFlagger()
        data_flagging.set_output_folder(output_path)
        data_flagging.load_config_json(self.json_path)
        data_flagging.set_column_names('Timestamp', 'Height', 'Flag')
        data_flagging.set_station(station)
        data_flagging.set_gauge_details(self.gauge_details_path)
        data_flagging.set_missing_value_filler(self.missing_meas_value)
        data_flagging.set_tide_series(data_tide)
        data_flagging.set_ml_training_data(self.datadir_labels, self.datadir_tides_ml)
        data_flagging.import_data(self.datadir, sta_filename)
        data_flagging.run()

    def test_quality_check_ballen(self):

        #select the station (here: Ballen)
        station = 'Ballen'
        sta_filename = 'WL_Ballen_27084.txt'
        print(os.getcwd())

        #select output folder
        output_path = os.path.join(os.getcwd(),'output', station)
        if Path(output_path).exists(): shutil.rmtree(Path(output_path))

        #Set path to tidal signal file
        data_tide = os.path.join(self.datadir_tides, '27084.tmp')

        data_flagging = qc_generator.QualityFlagger()
        data_flagging.set_output_folder(output_path)
        data_flagging.load_config_json(self.json_path)
        data_flagging.set_column_names('Timestamp', 'Height', 'Flag')
        data_flagging.set_station(station)
        data_flagging.set_gauge_details(self.gauge_details_path)
        data_flagging.set_missing_value_filler(self.missing_meas_value)
        data_flagging.set_tide_series(data_tide)
        data_flagging.set_ml_training_data(self.datadir_labels, self.datadir_tides_ml)
        data_flagging.import_data(self.datadir, sta_filename)
        data_flagging.run()

    def test_quality_check_juelsminde(self):

        #select the station (here: Juelsminde)
        station = 'Juelsminde'
        sta_filename = 'WL_Juelsminde_23132.txt'
        print(os.getcwd())

        #select output folder
        output_path = os.path.join(os.getcwd(),'output', station)
        if Path(output_path).exists(): shutil.rmtree(Path(output_path))

        #Set path to tidal signal file
        data_tide = os.path.join(self.datadir_tides, '23132.tmp')

        data_flagging = qc_generator.QualityFlagger()
        data_flagging.set_output_folder(output_path)
        data_flagging.load_config_json(self.json_path)
        data_flagging.set_column_names('Timestamp', 'Height', 'Flag')
        data_flagging.set_station(station)
        data_flagging.set_gauge_details(self.gauge_details_path)
        data_flagging.set_missing_value_filler(self.missing_meas_value)
        data_flagging.set_tide_series(data_tide)
        data_flagging.set_ml_training_data(self.datadir_labels, self.datadir_tides_ml)
        data_flagging.import_data(self.datadir, sta_filename)
        data_flagging.run()

    def backup_test_quality_check_niels(self):

        #list of station for QC
        print(os.getcwd())
        #stations = ['Aarhus','Ballen','Esbjerg','Faaborg','Fredericia','Frederikshavn','Fynshav','Gedser','Grenaa','Hals-BarreHals','Hanstholm','Hesnaes','Hirtshals','Holbaek','Hornbak','Hvide-Sande',
        #            'Juelsminde','Kalundborg','Karrebaeksminde','Kerteminde','Kobenhavn','Koege','Korsor','Randers','Ribe','Rodby','Rodvig','Roskilde','Slipshavn','Tejn','Thyboroen','Vidaa']
        #sta_filename_all = ['WL_Aarhus_22331.txt','WL_Ballen_27084.txt','WL_Esbjerg_25149.txt','WL_Faaborg_28397.txt','WL_Fredericia_23293.txt',
        #                'WL_Frederikshavn_20101.txt','WL_Fynshav_26457.txt','WL_Gedser_31616.txt','WL_Grenaa_22121.txt','WL_Hals-Barre_20252.txt',
        #                'WL_Hanstholm_21009.txt','WL_Hesnaes_31494.txt','WL_Hirtshals_20047.txt','WL_Holbaek_29038.txt','WL_Hornbak_30017.txt',
        #                'WL_Hvide-Sande_24342.txt','WL_Juelsminde_23132.txt','WL_Kalundborg_29141.txt','WL_Karrebaeksminde_31171.txt','WL_Kerteminde_28198.txt',
        #                'WL_Kobenhavn_30336.txt','WL_Koege_30479.txt','WL_Korsor_29393.txt','WL_Randers_22058.txt','WL_Ribe_25343.txt','WL_Rodby_31573.txt',
        #                'WL_Rodvig_31063.txt','WL_Roskilde_30407.txt','WL_Slipshavn_28234.txt','WL_Tejn_32048.txt','WL_Thyboroen_24006.txt','WL_Vidaa_26361.txt']
        #tidal_signals = ['22331.tmp','27084.tmp','25149.tmp','28397.tmp','23293.tmp','20101.tmp','26457.tmp','31616.tmp','22121.tmp','20252.tmp','21009.tmp','31493.tmp','20047.tmp','29038.tmp',
        #                '30017.tmp','24342.tmp','23132.tmp','29141.tmp','31171.tmp','28198.tmp','30336.tmp','30479.tmp','29393.tmp','22058.tmp','25343.tmp','31573.tmp','31063.tmp','30409.tmp','28234.tmp',
        #                '32048.tmp','24006.tmp','26361.tmp']
        
        stations = ['Kerteminde', 'Korsor']
        sta_filename_all = ['WL_Kerteminde_28198.txt','WL_Korsor_29393.txt']
        tidal_signals = ['28198.tmp', '29393.tmp']

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
            data_flagging.set_ml_training_data(self.datadir_labels, self.datadir_tides_ml)
            data_flagging.import_data(self.datadir, sta_filename)
            data_flagging.run()


if __name__ == '__main__':
    unittest.main()