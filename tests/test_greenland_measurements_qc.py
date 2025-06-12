##########################################################################################
## Test quality check script for sea level tide gauge data by frb for GronSL (2024/25)  ##
## This script runs the QC for all 4/5 stations in Greenland (1 method for each station)##
##########################################################################################

import os
from pathlib import Path
import shutil
import unittest

import source.main as qc_generator

class Test_QC_Station(unittest.TestCase):

    #Always run for each test
    def setUp(self):
        #All existing measurements in Greenland
        self.stations = ['Qaqortoq', 'Ittoqqortoormiit', 'Nuuk', 'Nuuk1', 'Pituffik', 'Upernavik1', 'Upernavik2'] 

        #Coordinates of stations
        self.coord = {
            'Qaqortoq': (60.718, -46.035),
            'Ittoqqortoormiit': (70.4836, -21.9621),
            'Nuuk': (64.1713, -51.7198),
            'Pituffik': (76.535, -68.84),
            'Upernavik': (72.7887, -56.146)
        }

        #Set path to measurements
        self.datadir = '/dmidata/users/frb/greenland_data_raw/Collected_Oct2024'

        #Set other parameter accordingly
        if self.datadir.endswith('Collected_Oct2024'):
            self.missing_meas_value = 999.999
            self.ending = '_data.csv'
            self.param = 'Height'

        if self.datadir.endswith('Collected_raw'):
            self.missing_meas_value = 999.000
            self.ending = '.dba'
            self.param = 'WaterLevel'

        #Set path to config json and and labelled data
        self.json_path = os.path.join(os.getcwd(), 'config.json')
        self.gauge_details_path = os.path.join(os.getcwd(), 'tides.local')
        self.datadir_labels = '/home/frb/Documents/Franca_Project/double_checked_labelled/correct_date'
        self.datadir_tides = '/home/frb/Documents/Franca_Project/double_checked_labelled/tidal_information'

    def test_quality_check_qaqortoq(self):

        #select the station (here: Qaqortoq)
        index_station=0 
        station = self.stations[index_station]
        sta_filename = station + self.ending
        data_tide = os.path.join(self.datadir_tides, 'Qaqortoq.tmp')
        
        print(os.getcwd())

        #select output folder
        output_path = os.path.join(os.getcwd(),'output', station)
        if Path(output_path).exists(): shutil.rmtree(Path(output_path))

        data_flagging = qc_generator.QualityFlagger()
        data_flagging.set_output_folder(output_path)
        data_flagging.load_config_json(self.json_path)
        data_flagging.set_column_names('Timestamp', self.param, 'Flag')
        data_flagging.set_station(station)
        data_flagging.set_gauge_details(self.gauge_details_path)
        data_flagging.set_missing_value_filler(self.missing_meas_value)
        data_flagging.set_tide_series(data_tide)
        data_flagging.set_ml_training_data(self.datadir_labels, self.datadir_tides)
        data_flagging.import_data(self.datadir, sta_filename)
        data_flagging.run()

    def test_quality_check_ittoq(self):

        #select the station (here: Ittoqqortoormiit)
        index_station=1
        station = self.stations[index_station]
        sta_filename = station + self.ending
        data_tide = os.path.join(self.datadir_tides, 'Ittoqqortoormiit.tmp')
        
        print(os.getcwd())

        #select output folder
        output_path = os.path.join(os.getcwd(),'output', station)
        if Path(output_path).exists(): shutil.rmtree(Path(output_path))

        data_flagging = qc_generator.QualityFlagger()
        data_flagging.set_output_folder(output_path)
        data_flagging.load_config_json(self.json_path)
        data_flagging.set_column_names('Timestamp', self.param, 'Flag')
        data_flagging.set_station(station)
        data_flagging.set_gauge_details(self.gauge_details_path)
        data_flagging.set_missing_value_filler(self.missing_meas_value)
        data_flagging.set_tide_series(data_tide)
        data_flagging.set_ml_training_data(self.datadir_labels, self.datadir_tides)
        data_flagging.import_data(self.datadir, sta_filename)
        data_flagging.run()
    
    def test_quality_check_nuuk(self):

        #select the station (here: Nuuk)
        index_station=2
        station = self.stations[index_station]
        sta_filename = station + self.ending
        data_tide = os.path.join(self.datadir_tides, 'Nuuk.tmp')

        print(os.getcwd())

        #select output folder
        output_path = os.path.join(os.getcwd(),'output', station)
        if Path(output_path).exists(): shutil.rmtree(Path(output_path))

        data_flagging = qc_generator.QualityFlagger()
        data_flagging.set_output_folder(output_path)
        data_flagging.load_config_json(self.json_path)
        data_flagging.set_column_names('Timestamp', self.param, 'Flag')
        data_flagging.set_station(station)
        data_flagging.set_gauge_details(self.gauge_details_path)
        data_flagging.set_missing_value_filler(self.missing_meas_value)
        data_flagging.set_tide_series(data_tide)
        data_flagging.set_ml_training_data(self.datadir_labels, self.datadir_tides)
        data_flagging.import_data(self.datadir, sta_filename)
        data_flagging.run()

    def test_quality_check_nuuk1(self):

        #select the station (here: Nuuk1)
        index_station=3
        station = self.stations[index_station]
        sta_filename = station + self.ending
        data_tide = os.path.join(self.datadir_tides, 'Nuuk.tmp')

        print(os.getcwd())

        #select output folder
        output_path = os.path.join(os.getcwd(),'output', station)
        if Path(output_path).exists(): shutil.rmtree(Path(output_path))

        data_flagging = qc_generator.QualityFlagger()
        data_flagging.set_output_folder(output_path)
        data_flagging.load_config_json(self.json_path)
        data_flagging.set_column_names('Timestamp', self.param, 'Flag')
        data_flagging.set_station(station)
        data_flagging.set_gauge_details(self.gauge_details_path)
        data_flagging.set_missing_value_filler(self.missing_meas_value)
        data_flagging.set_tide_series(data_tide)
        data_flagging.set_ml_training_data(self.datadir_labels, self.datadir_tides)
        data_flagging.import_data(self.datadir, sta_filename)
        data_flagging.run()

    def test_quality_check_pituffik(self):

        #select the station (here: Pituffik)
        index_station=4
        station = self.stations[index_station]
        sta_filename = station + self.ending
        data_tide = os.path.join(self.datadir_tides, 'Pituffik.tmp')

        print(os.getcwd())

        #select output folder
        output_path = os.path.join(os.getcwd(),'output', station)
        if Path(output_path).exists(): shutil.rmtree(Path(output_path))

        data_flagging = qc_generator.QualityFlagger()
        data_flagging.set_output_folder(output_path)
        data_flagging.load_config_json(self.json_path)
        data_flagging.set_column_names('Timestamp', self.param, 'Flag')
        data_flagging.set_station(station)
        data_flagging.set_gauge_details(self.gauge_details_path)
        data_flagging.set_missing_value_filler(self.missing_meas_value)
        data_flagging.set_tide_series(data_tide)
        data_flagging.set_ml_training_data(self.datadir_labels, self.datadir_tides)
        data_flagging.import_data(self.datadir, sta_filename)
        data_flagging.run()

    def test_quality_check_upernavik1(self):

        #select the station (here: Upernavik 2023)
        index_station=5
        station = self.stations[index_station]
        sta_filename = station + self.ending
        data_tide = os.path.join(self.datadir_tides, 'Upernavik.tmp')

        print(os.getcwd())

        #select output folder
        output_path = os.path.join(os.getcwd(),'output', station)
        if Path(output_path).exists(): shutil.rmtree(Path(output_path))

        data_flagging = qc_generator.QualityFlagger()
        data_flagging.set_output_folder(output_path)
        data_flagging.load_config_json(self.json_path)
        data_flagging.set_column_names('Timestamp', self.param, 'Flag')
        data_flagging.set_station(station)
        data_flagging.set_gauge_details(self.gauge_details_path)
        data_flagging.set_missing_value_filler(self.missing_meas_value)
        data_flagging.set_tide_series(data_tide)
        data_flagging.set_ml_training_data(self.datadir_labels, self.datadir_tides)
        data_flagging.import_data(self.datadir, sta_filename)
        data_flagging.run()



    def test_quality_check_upernavik2(self):

        #select the station (here: Upernavik 2024)
        index_station=6
        station = self.stations[index_station]
        sta_filename = station + self.ending
        data_tide = os.path.join(self.datadir_tides, 'Upernavik.tmp')

        print(os.getcwd())

        #select output folder
        output_path = os.path.join(os.getcwd(),'output', station)
        if Path(output_path).exists(): shutil.rmtree(Path(output_path))

        data_flagging = qc_generator.QualityFlagger()
        data_flagging.set_output_folder(output_path)
        data_flagging.load_config_json(self.json_path)
        data_flagging.set_column_names('Timestamp', self.param, 'Flag')
        data_flagging.set_station(station)
        data_flagging.set_gauge_details(self.gauge_details_path)
        data_flagging.set_missing_value_filler(self.missing_meas_value)
        data_flagging.set_tide_series(data_tide)
        data_flagging.set_ml_training_data(self.datadir_labels, self.datadir_tides)
        data_flagging.import_data(self.datadir, sta_filename)
        data_flagging.run()
    
    def test_plot_greenland_stations_map(self):
        import matplotlib.pyplot as plt
        import contextily as ctx
        import geopandas as gpd
        from shapely.geometry import Point

        def create_map(stations, output_path):
            """
            Creates a static map with a background and specified stations.
            
            :param stations: List of tuples [(lat, lon, "Label"), ...]
            :param output_file: Filename for the saved map
            """
            # Convert stations to a GeoDataFrame
            geometry = [Point(elem) for elem in list(stations.values())]
            gdf = gpd.GeoDataFrame(geometry=geometry, crs="EPSG:4326")
            
            fig, ax = plt.subplots(figsize=(6, 8))
            
            # Plot stations
            for index, (key, value) in enumerate(stations.items()):
                x, y = gdf.geometry[index].x, gdf.geometry[index].y
                ax.scatter(y, x, marker="o", s=45, label=key, color='black')
                ax.text(y - 0.5, x + 0.1, key, fontsize=18, ha='right', color='black')
            
            ax.set_ylabel("Longitude", fontsize=18)
            ax.set_xlabel("Latitude", fontsize=18)
            ax.tick_params(axis='both', which='major', labelsize=18) 
            ax.set_rasterization_zorder(0)
            
            # Add background map
            ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.OpenStreetMap.Mapnik, zoom=8)
            #ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.OpenStreetMap.HOT, zoom=8)

            # Save the map as an image file
            plt.savefig(os.path.join(output_path,'greenland_stations.png'), dpi=200, bbox_inches="tight")
            plt.close()
            print(f"Map saved as image")

        output_path = os.path.join(os.getcwd(),'output')
        #generate output folder for graphs and other docs
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        create_map(self.coord, output_path)

    def test_duration_measurements(self):
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from datetime import datetime
        import pandas as pd

        # Data for the timeline plot
        station_data = {
            "Pituffik": [("2005-09", "2024-10")],
            "Upernavik": [("2023-10", "2023-10"), ("2024-08", "2024-10")],
            "Nuuk": [("2014-07", "2023-08"), ("2022-11", "2024-10")],
            "Qaqortoq": [("2005-10", "2024-10")],
            "Ittoqqortoormiit": [("2007-10", "2024-10")],
        }

        # Convert string dates to datetime
        def parse_date(date_str):
            return datetime.strptime(date_str, "%Y-%m")

        # Prepare plotting data
        records = []
        for station, periods in station_data.items():
            for start, end in periods:
                records.append({
                    "Station": station,
                    "Start": parse_date(start),
                    "End": parse_date(end)
                })

        df = pd.DataFrame(records)

        station_order = list(station_data.keys())[::-1]
        y_pos = {name: idx for idx, name in enumerate(station_order)}
        df["y"] = df["Station"].map(y_pos)

        # Plot timeline
        fig, ax = plt.subplots(figsize=(9, 6))

        # Plot each bar
        for _, row in df.iterrows():
            ax.plot([row["Start"], row["End"]], [row["y"], row["y"]], color="#4682B4", linewidth=6, solid_capstyle="round")
            ax.plot(row["Start"], row["y"], 'o', color='firebrick')
            ax.plot(row["End"], row["y"], 'o', color='steelblue')

        # Formatting
        ax.set_yticks(range(len(station_order)))
        ax.set_yticklabels(station_order, fontsize=18)
        ax.xaxis.set_major_locator(mdates.YearLocator(3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.set_xlim([datetime(2005, 1, 1), datetime(2025, 1, 1)])
        #ax.set_title("Tide gauge station timelines in Greenland", fontsize=13)
        ax.grid(True, axis='x', linestyle='--', alpha=0.5)
        ax.tick_params(axis='x', labelsize=18)  # Set x-axis label font size to 18
        plt.tight_layout()

        # Save the figure
        plt.savefig(os.path.join(os.getcwd(),'output','greenland_station_timeline.png'), dpi=300,bbox_inches="tight")



if __name__ == '__main__':
    unittest.main()