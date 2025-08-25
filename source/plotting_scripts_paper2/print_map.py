import os
import numpy as np
import matplotlib.pyplot as plt
import contextily as ctx
import geopandas as gpd
import pandas as pd

import matplotlib.ticker as ticker

from shapely.geometry import Point
from matplotlib.patches import FancyArrowPatch

plt.rcParams.update({'font.size': 20})  # Set font size globally to 14

def create_map(stations, tidal_record, df1, df2, df3, output_path):
    """
    Creates a static map with a background and specified stations.
            
    :param stations: List of tuples [(lat, lon, "Label"), ...]
    :param output_file: Filename for the saved map
    """
    # Convert stations to a GeoDataFrame
    geometry = [Point(elem) for elem in list(stations.values())]
    gdf = gpd.GeoDataFrame(geometry=geometry, crs="EPSG:4326")

    # Normalize opacity between 0.35 and 1
    values = np.array(list(tidal_record.values()))
    print(values)
    alpha_values = 0.35 + (values - values.min()) / (values.max() - values.min()) * (1.0 - 0.35)
    print(alpha_values)
            
    fig = plt.figure(figsize=(20, 9))
    gs = fig.add_gridspec(3, 2, width_ratios=[4, 3], hspace=0.45)#, wspace=-0.02)  # 3 rows, 2 columns, left plot twice as wide

    # Plot stations
    ax_big = fig.add_subplot(gs[:, 0])
    for index, (key, value) in enumerate(stations.items()):
        x, y = gdf.geometry[index].x, gdf.geometry[index].y
        a = alpha_values[index]

        if key == 'Korsor':
            ax_big.scatter(y, x, marker="s", s=50, color='red', label='Spatial Testing')
            #ax_big.scatter(y, x, marker="s", alpha=a, s=50, color='red', label='Testing station')
        elif key == 'Køge' or key == 'Hirtshals':
            ax_big.scatter(y, x, marker="s", s=50, color='red')
        elif key == 'Frederikshavn':
            ax_big.scatter(y, x, marker="v", s=50, color='b', label='Training and Temporal Testing')
        elif key == 'Esbjerg':
            ax_big.scatter(y, x, marker="v", s=50, color='b')
        else:
            ax_big.scatter(y, x, marker="v", s=50, color='b')
        ax_big.legend(loc="best", fontsize=18)
        #ax.text(y - 0.1, x + 0.1, key, fontsize=18, ha='right', color='black')
            
    ax_big.set_ylabel("Longitude")
    ax_big.set_xlabel("Latitude")
    ax_big.tick_params(axis='both', which='major') 
    ax_big.set_rasterization_zorder(0)
    ax_big.set_xlim(5.0, 18)
    ax_big.set_ylim(53, 60)

    # Add background map
    #ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.OpenStreetMap.Mapnik, zoom=8)
    ctx.add_basemap(ax_big, crs="EPSG:4326", source=ctx.providers.CartoDB.Positron, zoom=8)

    #norm = plt.Normalize(np.min(alpha_values), np.max(alpha_values))
    #sm = plt.cm.ScalarMappable(cmap="Reds", norm=norm)
    #sm.set_array([])
    #cbar = plt.colorbar(sm, ax=ax_big, orientation="vertical", pad=0.01, anchor=(0, 1), shrink=0.4)
    #cbar.set_ticks([alpha_values.min(), alpha_values.max()])
    #cbar.set_ticklabels(['Low', 'High'])
    #cbar.set_label('Tide', rotation=-90, va="top", labelpad=-20)

    #Add text on map
    ax_big.text(7, 57.3, "North Sea", color="black", weight="bold", ha="center", va="center")
    ax_big.text(16.5, 55.6, "Baltic Sea", color="black", weight="bold", ha="center", va="center")

    # Three smaller stacked plots on the right
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(df1[time_column], df1[measurement_column], 'o', color='black', markersize=2)
    ax1.set_title("Hornbaek")
    ax1.set_ylabel("Water Level [m]")
    ticks = ax1.get_xticks()
    ax1.set_xticks(ticks[::2]) 
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(df2[time_column], df2[measurement_column], 'o', color='black', markersize=2)
    ax2.set_title("Tejn")
    ax2.set_ylabel("Water Level [m]")
    ticks = ax2.get_xticks()
    ax2.set_xticks(ticks[::2])
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

    ax3 = fig.add_subplot(gs[2, 1])
    ax3.plot(df3[time_column], df3[measurement_column], 'o', color='black', markersize=2)
    ax3.set_title("Esbjerg")
    ax3.set_ylabel("Water Level [m]")
    ax3.set_xlabel("Timestamp")
    ticks = ax3.get_xticks()
    ax3.set_xticks(ticks[::2])
    ax3.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

    #Arrow from Big Plot to Middle Right Plot
    arrow1 = FancyArrowPatch(
        #(0.17, 0.338),    # start point in figure fraction
        (0.232, 0.37),    # start point in figure fraction
        (0.55, 0.2),    # end point in figure fraction
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=15,
        color="black",
        linewidth=1,
        connectionstyle="arc3,rad=0.2"
    )
    fig.patches.append(arrow1)

    arrow2 = FancyArrowPatch(
        #(0.385, 0.51),    # start point in figure fraction
        (0.36, 0.46),    # start point in figure fraction
        (0.55, 0.77),    # end point in figure fraction
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=15,
        color="black",
        linewidth=1,
        connectionstyle="arc3,rad=-0.2"
    )
    fig.patches.append(arrow2)

    arrow3 = FancyArrowPatch(
        #(0.51, 0.312),    # start point in figure fraction
        (0.43, 0.37),    # start point in figure fraction
        (0.55, 0.5),    # end point in figure fraction
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=15,
        color="black",
        linewidth=1,
        connectionstyle="arc3,rad=-0.3"
    )
    fig.patches.append(arrow3)

    # Save the map as an image file
    plt.savefig(os.path.join(output_path,'dk_stations.png'), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Map saved as image")

output_path = os.path.join('/dmidata/users/frb/QC Work/2_Greenland&Denmark/output_paper')

#Coordinates of stations
coord = {'Frederikshavn': (57.4357340, 10.5478510), 'Esbjerg': (55.4601700, 8.4396940), 'Hornbaek': (56.0933900, 12.4571390), 'Kobenhavn': (55.704292, 12.598921), 'Gedser': (54.5721200, 11.9244830), 'Tejn': (55.2489970, 14.8367520),
         'Korsor': (55.3306500, 11.1422410), 'Køge': (55.455463, 12.196472), 'Hirtshals': (57.5950930, 9.9625200)}
#tidal_record = {'Frederikshavn': ((28.2+27.1)), 'Esbjerg': ((107.1+120.8)), 'Hornbaek': ((16.6+15.7)), 'Kobenhavn': ((13.7+15.2)), 'Gedser': ((8.8+9.7)), 'Tejn': (2.7+3.2),
#         'Korsor': ((22.7+23.7)), 'Køge': ((8.4+6.5)), 'Hirtshals': ((24.4+23.0))}
tidal_record = {'Frederikshavn': 55.3, 'Esbjerg': 55.3, 'Hornbaek': 32.3, 'Kobenhavn': 28.9, 'Gedser': 18.5, 'Tejn': 5.9, 'Korsor': 46.4, 'Køge': 14.9, 'Hirtshals': 47.4}

#Import ts data
time_column = 'Timestamp'
measurement_column = 'WaterLevel'
qc_column_seadb = 'QCFlag Seadb'

path_seadb = '/dmidata/users/frb/QC Work/2_Greenland&Denmark/Input_data/temporal/raw_data'

#Open .seadb_0 file and fix the column names
df_seadb_all = pd.read_csv(os.path.join(path_seadb,'30017.seadb_0'), sep=";", header=None)
df1 = pd.DataFrame(columns=[time_column, measurement_column, qc_column_seadb])
df1[time_column] = pd.to_datetime(df_seadb_all.iloc[:, 1])
df1[measurement_column] = df_seadb_all.iloc[:, 2]/100
filtered_df1 = df1[(df1[time_column] >= '2016-06-29 00:00:00') & (df1[time_column] <= '2016-07-04 00:00:00')]

df_seadb_all = pd.read_csv(os.path.join(path_seadb,'25149.seadb_0'), sep=";", header=None)
df3 = pd.DataFrame(columns=[time_column, measurement_column, qc_column_seadb])
df3[time_column] =  pd.to_datetime(df_seadb_all.iloc[:, 1])
df3[measurement_column] = df_seadb_all.iloc[:, 2]/100
filtered_df3 = df3[(df3[time_column] >= '2017-05-02 00:00:00') & (df3[time_column] <= '2017-05-07 00:00:00')]

df_seadb_all = pd.read_csv(os.path.join(path_seadb,'32048.seadb_0'), sep=";", header=None)
df2 = pd.DataFrame(columns=[time_column, measurement_column, qc_column_seadb])
df2[time_column] =  pd.to_datetime(df_seadb_all.iloc[:, 1])
df2[measurement_column] = df_seadb_all.iloc[:, 2]/100
filtered_df2 = df2[(df2[time_column] >= '2017-02-20 00:00:00') & (df2[time_column] <= '2017-02-25 00:00:00')]

#generate output folder for graphs and other docs
if not os.path.exists(output_path):
    os.makedirs(output_path)
create_map(coord, tidal_record, filtered_df1, filtered_df2, filtered_df3, output_path)