###################################################
## Written by frb for GronSL project (2024-2025) ##
## Contains methods for 4 different graphs       ##
###################################################

import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.dates as mdates
import matplotlib.font_manager as fm

mpl.rcdefaults()  # Reset everything

if os.name == 'nt':  # Windows
    font_path = "C:\\Windows\\Fonts\\TimesNewRoman.ttf"
else: #else Linux(Ubuntu)
    font_path = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf" 

#prop = fm.FontProperties(fname=font_path, size=12)
#prop = fm.FontProperties(fname=font_path)
#plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.size"] = 13  # Set global font size (adjust as needed)
plt.rcParams["axes.labelsize"] = 13  # Set x and y label size
plt.rcParams["axes.titlesize"] = 13  # Set title size
plt.rcParams["xtick.labelsize"] = 13  # Set x-axis tick labels size
plt.rcParams["ytick.labelsize"] = 13  # Set y-axis tick labels size
plt.rcParams["legend.fontsize"] = 13  # Set x-axis tick labels size
plt.rcParams["figure.titlesize"] = 13  # Set y-axis tick labels size

BIGGER_SIZE = 13
#plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
#plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
#plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
#plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
#plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
#plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
#plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#plt.rcParams.update({'font.size': BIGGER_SIZE})
#plt.rcParams.update({'axes.titlesize': BIGGER_SIZE})
#plt.rcParams.update({'axes.labelsize': BIGGER_SIZE})
#plt.rcParams.update({'xtick.labelsize': BIGGER_SIZE})
#plt.rcParams.update({'xtick.labelsize': BIGGER_SIZE})
#plt.rcParams.update({'legend.fontsize': BIGGER_SIZE})
#plt.rcParams.update({'figure.titlesize': BIGGER_SIZE})

pio.renderers.default = "browser"

class HelperMethods():

    def set_output_folder(self, folder_path):
        self.folder_path = folder_path


    def zoomable_plot_df(self, x_axis, data, y_title, x_title, title, legend_name = None, x_axis_2=None, data_2=None, y_title_2=None):
        """
        Generates a zoomable graph for one or two series. This is very slow and memory intensive. Only use it remotely or when really needed.)

        Input:
        -x_axis [Pandas Series] (here: timestamp)
        -data [Pandas Series] (here: measurement 1)
        -y_title [str]
        -x_title [str]
        -Graph title [str]
        -Legend [str] (to describe measurement 1)
        -x_title_2 [str] (used for measurement 2) (default is None, but could be added)
        -data_2 [Pandas Series] (here: measurement 2) (default is None, but could be added)
        -y_title_2 [str] (describes measurement 2) (default is None, but could be added)
        *If a second series is added, then all the corresponding values need to be defined.
        """

        # Create a figure with Plotly
        fig = go.Figure()

        # Add adjusted water level data
        fig.add_trace(go.Scatter(x=x_axis, y= data, mode='markers', marker=dict(
            size=5,          # Marker size
            color='rgb(255, 0, 0)',  # Marker color
            symbol='circle',   # Marker shape (e.g., 'circle', 'square', 'diamond')
            opacity=0.8       # Marker transparency
        ),name=legend_name))
        
        #Add additional ts (if wanted)
        if data_2 is not None:
            fig.add_trace(go.Scatter(x=x_axis_2, y=data_2, mode='lines', name=y_title_2, marker = {'color' : 'red'}))  
            
        # Update layout for a better view
        fig.update_layout(title=title,
                        xaxis_title= x_title,
                        yaxis_title= y_title,
                        legend_title='Legend',
                        hovermode='closest')

        # Enable zoom and pan
        fig.update_layout(xaxis=dict(rangeslider=dict(visible=True), type="date"))

        # Show the figure
        fig.show()

        # Write the figure to an HTML file
        #html_file_path = os.path.join(self.folder_path,f'{title.replace(" ", "")}.html')
        #fig.write_html(html_file_path)
        
        # To ensure the file is created
        #print(f"HTML file created at: {html_file_path}")


    def plot_df(self, x_axis, data, y_title, x_title, title = None):
        """
        Generates graph for a single series.

        Input:
        -x_axis [Pandas Series] (here: timestamp)
        -data [Pandas Series] (here: measurement)
        -y_title [str]
        -x_title [str]
        -title: default is None, but can be overwritten [str]
        """
                
        plt.figure(figsize=(18, 9))
        plt.plot(x_axis, data,  marker='o', markersize=1.2, label = 'Water Level', linestyle='None', color = 'black')
        #if title != None:
        #    plt.title(f"{title} - Date: {x_axis.iloc[0]}")
        plt.xlabel(x_title)
        plt.ylabel(y_title)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder_path,f"{title}- Date: {x_axis.iloc[0]}.png"), dpi=200, bbox_inches="tight")
        plt.close()  # Close the figure to release memory


    def plot_two_df(self, x_axis, data_1, y_title_1, data_2, y_title_2, x_title, title = None):
        """
        Generates graph for a two series on different y-axis.

        Input:
        -x_axis [Pandas Series] (here: timestamp)
        -data_1 [Pandas Series] (here: measurement 1)
        -y_title_1 [str]
        -data_2 [Pandas Series] (here: measurement 2)
        -y_title_2 [str]
        -x_title [str]
        -title: default is None, but can be overwritten [str]
        """
        #plt.rcParams["font.size"] = 12  # Set global font size (adjust as needed)
        #plt.rcParams["axes.labelsize"] = 12  # Set x and y label size
        #plt.rcParams["axes.titlesize"] = 12  # Set title size
        #plt.rcParams["xtick.labelsize"] = 12  # Set x-axis tick labels size
        #plt.rcParams["ytick.labelsize"] = 12  # Set y-axis tick labels size

        fig, ax1 = plt.subplots(figsize=(18, 10))

        #BIGGER_SIZE = 12
        #plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
        #plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
        #plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
        #plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
        #plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
        #plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
        #plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        #plt.rcParams.update({'font.size': BIGGER_SIZE})
        #plt.rcParams.update({'axes.titlesize': BIGGER_SIZE})
        #plt.rcParams.update({'axes.labelsize': BIGGER_SIZE})
        #plt.rcParams.update({'xtick.labelsize': BIGGER_SIZE})
        #plt.rcParams.update({'xtick.labelsize': BIGGER_SIZE})
        #plt.rcParams.update({'legend.fontsize': BIGGER_SIZE})
        #plt.rcParams.update({'figure.titlesize': BIGGER_SIZE})

        color = 'black'
        #ax1.set_xlabel(x_title, fontproperties=prop)
        #ax1.set_ylabel(y_title_1 , color=color, fontproperties=prop)
        ax1.set_xlabel(x_title)
        ax1.set_ylabel(y_title_1 , color=color)
        #ax1.plot(x_axis, data_1, marker='+', markersize=7, color=color, linestyle='None')
        ax1.plot(x_axis, data_1, marker='o', markersize=3, color=color, linestyle='None')
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

        color = 'tab:red'
        ax2.set_ylabel(y_title_2, color=color)  # we already handled the x-label with ax1
        ax2.plot(x_axis, data_2, marker='o', markersize=3, color=color, linestyle='None', alpha=0.6)
        ax2.tick_params(axis='y', labelcolor=color)

        # Format x-axis to show only numbers
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))  # Format: MM-DD HH:MM
        #ax2.xaxis.set_major_locator(mdates.HourLocator(interval=6))  # Tick every 6 hours
        #plt.xticks(rotation=45) 

        # Ensure all spines (box edges) are visible
        for spine in ax2.spines.values():
            spine.set_visible(True)

        fig.tight_layout() 
        #if title != None:
        #    plt.title(f"{title} - Date: {x_axis.iloc[0]}")
        fig.savefig(os.path.join(self.folder_path,f"{title}.png"),  bbox_inches="tight")
        plt.close()  # Close the figure to release memory


    def plot_two_df_same_axis(self, x_axis, data_1, y_title, legend_1, data_2, x_title, legend_2,title = None):
        """
        Generates graph for a two series on same y-axis.

        Input:
        -x_axis [Pandas Series] (here: timestamp)
        -data_1 [Pandas Series] (here: measurement 1)
        -y_title [str] (overall title for measurement 1 and 2)
        -legend_1 [str] (describes what measurement 1 is)
        -data_2 [Pandas Series] (here: measurement 2)
        -x_title [str]
        -legend_2 [str] (describes what measurement 2 is)
        -title: default is None, but can be overwritten [str]
        """
        #plt.rcParams["font.size"] = 12  # Set global font size (adjust as needed)
        #plt.rcParams["axes.labelsize"] = 12  # Set x and y label size
        #plt.rcParams["axes.titlesize"] = 12  # Set title size
        #plt.rcParams["xtick.labelsize"] = 12  # Set x-axis tick labels size
        #plt.rcParams["ytick.labelsize"] = 12  # Set y-axis tick labels size

        fig, ax1 = plt.subplots(figsize=(11, 6))
        #BIGGER_SIZE = 12
        #plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
        #plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
        #plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
        #plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
        #plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
        #plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
        #plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        #plt.rcParams.update({'font.size': BIGGER_SIZE})
        #plt.rcParams.update({'axes.titlesize': BIGGER_SIZE})
        #plt.rcParams.update({'axes.labelsize': BIGGER_SIZE})
        #plt.rcParams.update({'xtick.labelsize': BIGGER_SIZE})
        #plt.rcParams.update({'xtick.labelsize': BIGGER_SIZE})
        #plt.rcParams.update({'legend.fontsize': BIGGER_SIZE})
        #plt.rcParams.update({'figure.titlesize': BIGGER_SIZE})
        color = 'black'
        ax1.plot(x_axis, data_2, marker='o', markersize=3, color=color, linestyle='None', alpha=0.6, label= legend_2)
        
        color = 'tab:red'
        ax1.set_xlabel(x_title, fontsize=14)
        ax1.set_ylabel(y_title, fontsize=14)
        #ax1.set_xlabel(x_title, fontsize=14, fontproperties=prop)
        #ax1.set_ylabel(y_title, fontsize=14, fontproperties=prop)
        #ax1.set_xlabel(x_title)
        #ax1.set_ylabel(y_title)
        ax1.plot(x_axis, data_1, marker='o', markersize=3, color=color, linestyle='None', label= legend_1)
        
        ax1.tick_params(axis='y')
        ax1.legend(loc='upper right', frameon=False)  # Position the legend as desired   

        #Format x-axis to show only numbers
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))  # Format: YYY-MM-DD HH:MM
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax1.set_xticks(ax1.get_xticks()[::2]) 
        #ax1.xaxis.set_major_locator(mdates.HourLocator(interval=6))  # Tick every 6 hours
        #plt.xticks(rotation=45) 

        # Ensure all spines (box edges) are visible
        for spine in ax1.spines.values():
            spine.set_visible(True)

        fig.tight_layout()
        #if title != None:
        #    plt.title(f"{title} - Date: {x_axis.iloc[0]}")
        fig.savefig(os.path.join(self.folder_path,f"{title}.png"), dpi=300,  bbox_inches="tight")
        plt.close()  # Close the figure to release memory