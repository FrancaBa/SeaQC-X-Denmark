############################################################
## Written by frb for GronSL project (2024-2025)          ##
## Contains methods for more advanced analysis graphs     ##
############################################################

import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.io as pio
import matplotlib.font_manager as fm
import matplotlib.dates as mdates

from datetime import datetime

if os.name == 'nt':  # Windows
    font_path = "C:\\Windows\\Fonts\\TimesNewRoman.ttf"
else: #else Linux(Ubuntu)
    font_path = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf" 

prop = fm.FontProperties(fname=font_path)
#plt.rcParams["font.family"] = prop.get_name()
plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.size"] = 16  # Set global font size (adjust as needed)
pio.renderers.default = "browser"

class GraphMaker():
        
    def set_output_folder(self, folder_path):
        self.folder_path = os.path.join(folder_path,'0_final graphs')

        #generate output folder for graphs and other docs
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

    def set_station(self, station):
        self.station = station

    def run(self, df, time_column, data_column):
        """
        Calls a method for a manual defined period of interest based on station.

        Input:
        -df: whole dataframe [df]
        -time_column: name of column with timestamp [str]
        -data_column: name of column with data [str]
        """
       
        if self.station == 'Ittoqqortoormiit':
            self.run_graphs_zoom(df, time_column, data_column, '2011', '03', '03', '04', '05','06', '18')
            self.run_graphs_zoom(df, time_column, data_column, '2011', '05', '05', '01', '07')
            self.run_graphs_zoom(df, time_column, data_column, '2012', '06', '07', '15', '03')
            self.run_graphs_zoom(df, time_column, data_column, '2013', '08', '11', '10', '20')
            self.run_graphs_zoom(df, time_column, data_column, '2015', '07', '07', '01', '05')
            self.run_graphs_zoom(df, time_column, data_column, '2017', '03', '05', '10', '25')  
            self.run_graphs_zoom(df, time_column, data_column, '2019', '01', '01', '16', '23')      
            self.run_graphs_zoom(df, time_column, data_column, '2019', '05', '05', '11', '26') 
            self.run_graphs_zoom(df, time_column, data_column, '2019', '06', '06', '01', '20')
            self.run_graphs_zoom(df, time_column, data_column, '2020', '12', '12', '22', '31')
            self.run_graphs_zoom(df, time_column, data_column, '2022', '03', '03', '14', '18')
            self.run_graphs_zoom(df, time_column, data_column, '2022', '06', '07', '25', '05')
            self.run_graphs_zoom(df, time_column, data_column, '2023', '03', '04', '30', '05')
            self.run_graphs_zoom(df, time_column, data_column, '2023', '06', '07', '07', '05')
            self.run_graphs_zoom(df, time_column, data_column, '2023', '10', '10', '01', '10')
            self.run_graphs_zoom(df, time_column, data_column, '2023', '11', '11', '13', '30')
            self.run_graphs_zoom(df, time_column, data_column, '2023', '12', '12', '25', '31')
        elif self.station == 'Qaqortoq':
            self.run_graphs_zoom(df, time_column, data_column, '2006', '02', '03', '01', '10')
            self.run_graphs_zoom(df, time_column, data_column, '2008', '06', '07', '20', '05')
            self.run_graphs_zoom(df, time_column, data_column, '2008', '10', '10', '01', '20')
            self.run_graphs_zoom(df, time_column, data_column, '2011', '11', '11', '15', '24')
            self.run_graphs_zoom(df, time_column, data_column, '2014', '01', '01', '22', '30')
            self.run_graphs_zoom(df, time_column, data_column, '2015', '07', '07', '02', '23')  
            self.run_graphs_zoom(df, time_column, data_column, '2018', '05', '05', '03', '20')
            self.run_graphs_zoom(df, time_column, data_column, '2022', '10', '10', '01', '20')
            self.run_graphs_zoom(df, time_column, data_column, '2023', '10', '10', '20', '30')
            self.run_graphs_zoom(df, time_column, data_column, '2023', '07', '07', '01', '10')
            self.run_graphs_zoom(df, time_column, data_column, '2024', '06', '07', '25', '10')
        elif self.station == 'Nuuk':
            self.run_graphs_zoom(df, time_column, data_column, '2014', '11', '11', '20', '22')
            self.run_graphs_zoom(df, time_column, data_column, '2014', '12', '12', '27', '29')
            self.run_graphs_zoom(df, time_column, data_column, '2016', '09', '10', '01', '25') 
            self.run_graphs_zoom(df, time_column, data_column, '2017', '04', '04', '15', '25') 
            self.run_graphs_zoom(df, time_column, data_column, '2020', '11', '11', '15', '30') 
            self.run_graphs_zoom(df, time_column, data_column, '2020', '12', '12', '15', '30')
            self.run_graphs_zoom(df, time_column, data_column, '2022', '02', '02', '01', '10')
            self.run_graphs_zoom(df, time_column, data_column, '2023', '1', '1', '24', '25', '21', '18') 
            self.run_graphs_zoom(df, time_column, data_column, '2023', '1', '1', '25', '25', '15', '18')
            self.run_graphs_zoom(df, time_column, data_column, '2023', '1', '1', '25', '25', '12', '18')
            self.run_graphs_zoom(df, time_column, data_column, '2023', '1', '1', '25', '25', '0', '23')
        elif self.station == 'Nuuk1':
            self.run_graphs_zoom(df, time_column, data_column, '2022', '11', '12', '27', '12')
            self.run_graphs_zoom(df, time_column, data_column, '2023', '04', '04', '10', '12') 
            self.run_graphs_zoom(df, time_column, data_column, '2023', '11', '11', '15', '20') 
            self.run_graphs_zoom(df, time_column, data_column, '2024', '02', '02', '15', '20') 
            self.run_graphs_zoom(df, time_column, data_column, '2023', '1', '1', '24', '25', '21', '18') 
            self.run_graphs_zoom(df, time_column, data_column, '2023', '1', '1', '25', '25', '0', '23')
            self.run_graphs_zoom(df, time_column, data_column, '2023', '1', '1', '25', '25', '15', '18')
            self.run_graphs_zoom(df, time_column, data_column, '2023', '1', '1', '25', '25', '12', '18')
        elif self.station == 'Pituffik':
            self.run_graphs_zoom(df, time_column, data_column, '2007', '07', '07', '22', '25')
            self.run_graphs_zoom(df, time_column, data_column, '2007', '09', '09', '09', '21')
            self.run_graphs_zoom(df, time_column, data_column, '2008', '08', '08', '10', '13')        
            self.run_graphs_zoom(df, time_column, data_column, '2016', '02', '02', '01', '29')
            self.run_graphs_zoom(df, time_column, data_column, '2016', '03', '03', '01', '03')
            self.run_graphs_zoom(df, time_column, data_column, '2017', '12', '12', '14', '15', '10', '0')  
            self.run_graphs_zoom(df, time_column, data_column, '2017', '12', '12', '14', '14', '16', '19') 
            self.run_graphs_zoom(df, time_column, data_column, '2017', '12', '12', '14', '14', '17', '23')
        elif self.station == 'Upernavik1':
            self.run_graphs_zoom(df, time_column, data_column, '2023', '08', '08', '12', '12', '03', '22')
            self.run_graphs_zoom(df, time_column, data_column, '2023', '08', '08', '21', '31')
        elif self.station == 'Upernavik2':   
            self.run_graphs_zoom(df, time_column, data_column, '2024', '09', '09', '21', '22', '03', '15')
            self.run_graphs_zoom(df, time_column, data_column, '2024', '10', '10', '16', '16')  
            self.run_graphs_zoom(df, time_column, data_column, '2024', '10', '10', '20', '21', '19', '11')  
            self.run_graphs_zoom(df, time_column, data_column, '2024', '10', '10', '26', '27', '22', '13')   
        else:
            return 
        
    def run_graphs_zoom(self, df, time_column, data_column, year, start_month, end_month, start_day, end_day, start_hour='00', end_hour='23'):
        """
        Generates 2 graphs for a manual defined period of interest based on station.

        Input:
        -df: whole dataframe [df]
        -time_column: name of column with timestamp [str]
        -data_column: name of column with data [str]
        -year [str]
        -Start month of period of interest [str]
        -End month of period of interest [str]
        -Start day of period of interest [str]
        -End day of period of interest [str]
        -Start hour of period of interest [str] (default 00)
        -End hour of period of interest [str] (default 23)
        """
        
        self.extract_period(df, time_column, year, start_month, end_month, start_day, end_day, start_hour, end_hour)
        self.make_ts_plot(time_column, data_column)
        self.two_in_one_graph(time_column, data_column)

    def extract_period(self, data, time_column, year, start_month, end_month, start_day='1', end_day='31', start_hour='00', end_hour='23'):
        """
        Extract relevant periods based on inputs to a new shorter dataframe. 
        This new dataframe is saved as csv with only relevant columns and also plotted for visual analysis.

        Input:
        -Main dataframe [df]
        -Column name for timestamp [str]
        -Column name for relevant measurement series [str]
        -Year [str] (periods cannot go between years!)
        -Start month (possible: 01-12) [str]
        -End month (possible: 01-12) [str]
        -Start day (by default 1, but can be overwritten) (possible: 01-31) [str]
        -End day (by default 31, but can be overwritten) (possible: 01-31) [str]
        -Start hour (by default 00, but can be overwritten) (possible: 00-22) [str]
        -End hour (by default 23, but can be overwritten) (possible: 01-23) [str]
        """

        start_date = datetime(int(year), int(start_month), int(start_day), int(start_hour))
        end_date = datetime(int(year), int(end_month), int(end_day), int(end_hour))
        print(start_date, end_date)
        self.relev_df = data[(data[time_column] >= start_date) & (data[time_column]<= end_date)]

    def make_ts_plot(self, time_column, data_column):
        """
        Makes a plot marking the different spikes detected by different approaches for a period of interest.

        Input:
        -time_column: name of column with timestamp [str]
        -data_column: name of column with data [str]
        """
                
        print(self.relev_df)

        #Visualization of different spike detection methods
        self.relev_df['spike_value_statistical_improved'] = self.relev_df['outlier_change_rate'] | self.relev_df['noisy_period']
        pot_relev_columns = ['spike_value_statistical', 'spike_value_statistical_improved', 'cotede_spikes', 'cotede_improved_spikes', 'selene_spikes', 'selene_improved_spikes', 'ml_anomalies']
        lable_title = ['Implausible change rate', 'Implausible change rate (improved)', 'Neighbour Comparison', 'Neighbour Comparison (improved)', 'Spline fitting', 'Spline fitting (improved)', 'ML algorithm']
        markers = ["s" , "d", "o", "v", ">", "*", "P"]
        relev_columns = [col for col in pot_relev_columns if col in self.relev_df.columns]

        # Generate colors from a single-hue colormap
        color_map = plt.cm.plasma  # Choose a colormap (other good ones: Viridis, Cividis, Plasma)
        colors = [color_map(i /len(relev_columns)) for i in range(len(relev_columns))]
        #colors = ['red','green','blue','cyan','magenta', 'darkorange', 'lime']        
        offset_value = np.zeros(len(self.relev_df))+0.03
        title = 'Comparison of Spike Detection Methods'
        plt.figure(figsize=(14, 7), dpi=300)
        plt.plot(self.relev_df[time_column], self.relev_df[data_column], label='Measurements', color='black', marker='o',  markersize=0.6, linestyle='None')
        # Stack markers using offsets
        for i in range(len(relev_columns)):
            mask = self.relev_df[relev_columns[i]]
            scatter_marker = markers[i]
            scatter_colors = colors[i]
            plt.scatter(self.relev_df[time_column][mask], self.relev_df[data_column][mask] + offset_value[mask], color=scatter_colors, marker=scatter_marker, s=25, alpha=0.5, edgecolors='black', linewidth=0.5, label=lable_title[i])
            offset_value[mask] += 0.03

        plt.legend(title="Spike Detection Methods:", frameon=False).set_title("Spike Detection Methods:", prop={"weight": "bold"}) 
        plt.xlabel('Time')
        plt.ylabel('Water Level [m]')
        # Remove top and right spines
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder_path,f"{title}- Date: {self.relev_df[time_column].iloc[0]}.png"),  bbox_inches="tight")
        plt.close()

    def two_in_one_graph(self, time_column, data_column):
        """
        Makes a plot containing two subplots. One presents the period on interest and the table below indicates the performance of the different spike detection tools.

        Input:
        -time_column: name of column with timestamp [str]
        -data_column: name of column with data [str]
        """
        print(self.relev_df)

        #Visualization of different spike detection methods
        self.relev_df['spike_value_statistical_improved'] = self.relev_df['outlier_change_rate'] | self.relev_df['noisy_period']
        pot_relev_columns = ['spike_value_statistical', 'spike_value_statistical_improved', 'cotede_spikes', 'cotede_improved_spikes', 'selene_spikes', 'selene_improved_spikes', 'ml_anomalies']
        lable_title = ['Implausible rate of change', 'Implausible rate of change (improved)', 'Neighbour comparison', 'Neighbour comparison (improved)', 'Spline fitting', 'Spline fitting (improved)', 'ML algorithm']
        #mark ml detected spikes in ts graph
        highlight = self.relev_df[self.relev_df['ml_anomalies']]
        markers = ["s" , "d", "o", "v", ">", "*", "P"]
        relev_columns = [col for col in pot_relev_columns if col in self.relev_df.columns]
        count = [None] * len(lable_title)

        # Generate colors from a single-hue colormap
        #color_map = plt.cm.plasma  # Choose a colormap (other good ones: Viridis, Cividis, Plasma)
        #colors = [color_map(i /len(lable_title)) for i in range(len(lable_title))]
        colors = ['black','black','black','black','black', 'black', 'red'] 
        grid_heights = [0.28 - 0.16 * i for i in range(6)] 
        offset_value = np.zeros(len(self.relev_df))+0.35
        title = 'Comparison of Spike Detection Methods'
        # Create the figure and axes with shared x-axis
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8), gridspec_kw={'height_ratios': [5, 5]}, dpi=300)

        # Plot the first graph (Sea Level Measurements)
        ax1.plot(self.relev_df[time_column], self.relev_df[data_column], color='black', label='Measurement', marker='o',  markersize=1.2, linestyle='None')
        # Highlight where flag is True
        ax1.scatter(highlight[time_column], highlight[data_column], marker='o', s=3, color='red', label= 'ML detected spikes', zorder=2)
        ax1.set_ylabel('Water Level [m]')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.set_xlabel('Timestamp')
        ax1.tick_params(axis='x', labelbottom=True)
        ax1.legend(loc='upper right', frameon=False)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))  # Format: YYY-MM-DD HH:MM
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax1.set_xticks(ax1.get_xticks()[::3]) 

        # Plot the second graph (Icons)
        # Stack markers using offsets
        for i in range(len(relev_columns)):
            mask = self.relev_df[relev_columns[i]]
            scatter_marker = markers[i]
            scatter_colors = colors[i]
            #ax2.scatter(self.relev_df[time_column][mask], offset_value[mask], color=scatter_colors, marker=scatter_marker, s=25, alpha=0.5, edgecolors='black', linewidth=0.5, label=lable_title[i])
            ax2.scatter(self.relev_df[time_column][mask], offset_value[mask], color=scatter_colors, marker=scatter_marker, s=30, alpha=0.6, edgecolors=scatter_colors, linewidth=0.5, label=lable_title[i])
            count[i] = len(offset_value[mask])
            offset_value -= 0.16

        # Add gridlines at specific heights
        for height in grid_heights:
            ax2.axhline(y=height, color='gray', linewidth=0.6, alpha=0.5)

        ax2.axis('off')
        #ax2.legend(loc='lower right', frameon=False)
        #Add multi-line text
        text_box = f"""{lable_title[0]}\n\n{lable_title[1]}\n\n{lable_title[2]}\n\n{lable_title[3]}\n\n{lable_title[4]}\n\n{lable_title[5]}\n\n{lable_title[6]}"""
        ax2.figure.text(-0.24, 0.035, text_box, transform=ax2.figure.transFigure, fontsize=16, verticalalignment='bottom', horizontalalignment='left')
        text_box = f"""= {count[0]}\n\n= {count[1]}\n\n= {count[2]}\n\n= {count[3]}\n\n= {count[4]}\n\n= {count[5]}\n\n= {count[6]}"""
        ax2.figure.text(1, 0.035, text_box, transform=ax2.figure.transFigure, fontsize=16, verticalalignment='bottom', horizontalalignment='left')

        # Display the plot
        plt.subplots_adjust(hspace=0.01)
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder_path,f"{title}- Date: {self.relev_df[time_column].iloc[0]}-2in1.png"),  bbox_inches="tight")
        plt.close()