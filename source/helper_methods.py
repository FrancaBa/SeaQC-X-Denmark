import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.renderers.default = "browser"

class HelperMethods():

    def set_output_folder(self, folder_path):
        self.folder_path = folder_path

    def zoomable_plot_df(self, x_axis, data, y_title, x_title, title, legend_name = None, x_axis_2=None, data_2=None, y_title_2=None):

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
        
        plt.figure(figsize=(18, 9))
        plt.plot(x_axis, data,  marker='o', markersize=1, linestyle='None')
        if title != None:
            plt.title(f"{title} - Date: {x_axis.iloc[0]}")
        plt.xlabel(x_title)
        plt.ylabel(y_title)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder_path,f"{title}- Date: {x_axis.iloc[0]}.png"),  bbox_inches="tight")
        plt.close()  # Close the figure to release memory

    def plot_two_df(self, x_axis, data_1, y_title_1, data_2, y_title_2, x_title, title = None):

        fig, ax1 = plt.subplots(figsize=(18, 10))

        color = 'tab:green'
        ax1.set_xlabel(x_title)
        ax1.set_ylabel(y_title_1 , color=color)
        ax1.plot(x_axis, data_1, marker='+', markersize=3, color=color, linestyle='None')
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

        color = 'tab:orange'
        ax2.set_ylabel(y_title_2, color=color)  # we already handled the x-label with ax1
        ax2.plot(x_axis, data_2, marker='o', markersize=1, color=color, linestyle='None', alpha=0.6)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout() 
        if title != None:
            plt.title(f"{title} - Date: {x_axis.iloc[0]}")
        plt.savefig(os.path.join(self.folder_path,f"{title}.png"),  bbox_inches="tight")
        plt.close()  # Close the figure to release memory

    def plot_two_df_same_axis(self, x_axis, data_1, y_title, legend_1, data_2, x_title, legend_2,title = None):

        fig, ax1 = plt.subplots(figsize=(18, 10))

        color = 'tab:green'
        ax1.set_xlabel(x_title)
        ax1.set_ylabel(y_title)
        ax1.plot(x_axis, data_1, marker='+', markersize=3, color=color, linestyle='None', label= legend_1)
        ax1.tick_params(axis='y')

        color = 'tab:orange'
        ax1.plot(x_axis, data_2, marker='o', markersize=1, color=color, linestyle='None', alpha=0.6, label= legend_2)
        ax1.legend(loc='upper right')  # Position the legend as desired

        fig.tight_layout() 
        if title != None:
            plt.title(f"{title} - Date: {x_axis.iloc[0]}")
        plt.savefig(os.path.join(self.folder_path,f"{title}.png"),  bbox_inches="tight")
        plt.close()  # Close the figure to release memory