#########################################################
## Written by frb for DMI QC Comparison (2024-2025)    ##
## Contains a method for more advanced analysis graphs ##
#########################################################

import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams.update({'font.size': 22})  # Set font size globally to 18

#Run station of interest
file = '32048' #'20101', '32048', '30336', '31616', '30017', '25149', '29393', '20047', '30479' 

#Output table
output_path = f'/dmidata/users/frb/QC Work/2_Greenland&Denmark/output/{file}'
if not os.path.exists(output_path):
    os.makedirs(output_path)

#DF columns
time_column = 'Timestamp'
measurement_column = 'WaterLevel'
qc_column_mads = 'QCFlag Mads'
qc_column_seadb = 'QCFlag Seadb'
qc_column_gronsl = 'QCFlag Gronsl (non ML)'
qc_column_gronsl_ml = 'QCFlag Gronsl (ML)'
qc_column_ml = 'QCFlag(ML)'
qc_column_manual = 'QCFlag Manual'
qc_column_selene = 'QCFlag Selene'

#Paths to relevant files
path_selene  = '/dmidata/users/frb/QC Work/2_Greenland&Denmark/Input_data/temporal/data_qc_Selene'
path_ml  = '/dmidata/users/frb/QC Work/2_Greenland&Denmark/Input_data/temporal/data_ml_gr_dk_50'
path_seadb  = '/dmidata/users/frb/QC Work/2_Greenland&Denmark/Input_data/temporal/raw_data'
#path_selene  = '/dmidata/users/frb/QC Work/2_Greenland&Denmark/Input_data/spatial/data_qc_Selene'
#path_ml  = '/dmidata/users/frb/QC Work/2_Greenland&Denmark/Input_data/spatial/data_ml_gr_dk_50'
#path_seadb  = '/dmidata/users/frb/QC Work/2_Greenland&Denmark/Input_data/spatial/raw_data'

#Open the Selene QC tool from CMEMS as NetCDF file to df
ds = xr.open_dataset(os.path.join(path_selene,f'{file}.nc'))
df_selene = ds[['SLEV', 'SLEV_QC']].to_dataframe().reset_index()
if 'DEPTH' in df_selene.columns:
    df_selene = df_selene[df_selene['DEPTH'] != 1]
df_selene = df_selene.rename(columns={'SLEV': 'water_level', 'SLEV_QC': qc_column_selene, 'TIME': time_column})
df_selene = df_selene[[time_column, 'water_level', qc_column_selene]]
df_selene['water_level'] = df_selene['water_level'].astype(np.float64)
df_selene[time_column] = pd.to_datetime(df_selene[time_column], format='%Y-%m-%d %H:%M:%S').dt.round('min')
df_selene['QC_selene_mask'] = df_selene[qc_column_selene] != 1
print(df_selene)

#Open .seadb_0 file and fix the column names
df_seadb_all = pd.read_csv(os.path.join(path_seadb,f'{file}.seadb_0'), sep=";", header=None)
df_seadb = pd.DataFrame(columns=[time_column, measurement_column, qc_column_seadb])
df_seadb[time_column] = df_seadb_all.iloc[:, 1]
df_seadb[measurement_column] = df_seadb_all.iloc[:, 2]/100
df_seadb['raw_qc'] = df_seadb_all.iloc[:, 5]
df_seadb['raw_qc'] = df_seadb['raw_qc'].fillna(0.0)
df_seadb['manual_qc'] = df_seadb_all.iloc[:, 6]
df_seadb['manual_qc'] = df_seadb['manual_qc'].fillna(0.0)
df_seadb[qc_column_seadb] = df_seadb['raw_qc'].where(df_seadb['manual_qc'] != 2.0, 0)
df_seadb[qc_column_seadb] = df_seadb[qc_column_seadb].where(df_seadb['manual_qc'] != 1.0, 1)
df_seadb['QC_Seadb_manual_mask'] = df_seadb[qc_column_seadb] != 0 
df_seadb['QC_Seadb_mask'] = df_seadb['raw_qc'] != 0
df_seadb['cleaned_measurement'] = df_seadb[measurement_column].mask(df_seadb['QC_Seadb_mask'] == True, np.nan)
df_seadb[time_column] = pd.to_datetime(df_seadb[time_column], format='%Y-%m-%d %H:%M:%S')
print(df_seadb)
del df_seadb['manual_qc']

#Open .csv file and fix the column names
df_ml_all = pd.read_csv(os.path.join(path_ml,f'{file}.csv'), sep=",", header=0)
df_ml = pd.DataFrame(columns=[time_column, qc_column_ml])
df_ml[time_column] = df_ml_all.iloc[:, 0]
df_ml[measurement_column] = df_ml_all.iloc[:, 1]
df_ml[qc_column_ml] = df_ml_all.iloc[:, 3]
df_ml['QC_ml_mask'] = (df_ml[qc_column_ml] != 1.0).astype(bool)
df_ml[time_column] = pd.to_datetime(df_ml[time_column], format='%Y-%m-%d %H:%M:%S').dt.round('min')
print(df_ml)

#Merge the various dfs
#merged_df = pd.merge(df_seadb, df_mads, on=[time_column, measurement_column], how='outer')
#merged_df = pd.merge(merged_df, df_gronsl, on=[time_column, measurement_column], how='left')
#merged_df['QC_gronsl_mask'] = merged_df['QC_gronsl_mask'].fillna(False).astype(bool)
#merged_df = pd.merge(merged_df, df_gronsl_ml, on=[time_column, measurement_column], how='left')
#merged_df['QC_gronsl_ml_mask'] = merged_df['QC_gronsl_ml_mask'].fillna(False).astype(bool)
#merged_df = pd.merge(merged_df, df_manual, on=[time_column, measurement_column], how='left')
#merged_df['QC_manual_mask'] = merged_df['QC_manual_mask'].fillna(False).astype(bool)
merged_df = pd.merge(df_seadb, df_ml, on=[time_column, measurement_column], how='left')
merged_df['QC_ml_mask'] = merged_df['QC_ml_mask'].fillna(False).astype(bool)
merged_df = pd.merge(merged_df, df_selene, on=[time_column], how='left')
merged_df['QC_selene_mask'] = merged_df['QC_selene_mask'].fillna(False).astype(bool)

if file == '32048':
    short_df = merged_df[(merged_df[time_column] >= '2018-12-12 06:00:00') & (merged_df[time_column] <= '2018-12-12 15:00:00')]
    #short_df = merged_df[(merged_df[time_column] >= '2024-12-08 00:00:00') & (merged_df[time_column] <= '2024-12-11 00:00:00')]
    #short_df = merged_df[(merged_df[time_column] >= '2024-01-02 00:00:00') & (merged_df[time_column] <= '2024-01-06 00:00:00')]
    #short_df = merged_df[(merged_df[time_column] >= '2024-10-22 00:00:00') & (merged_df[time_column] <= '2024-10-23 00:00:00')]
    #short_df = merged_df[(merged_df[time_column] >= '2023-01-21 00:00:00') & (merged_df[time_column] <= '2023-01-23 00:00:00')]
    #short_df = merged_df[(merged_df[time_column] >= '2024-01-11 20:00:00') & (merged_df[time_column] <= '2024-01-14 00:00:00')]
    #short_df = merged_df[(merged_df[time_column] >= '2024-03-09 00:00:00') & (merged_df[time_column] <= '2024-03-13 00:00:00')]
    #short_df = merged_df[(merged_df[time_column] >= '2023-11-21 00:00:00') & (merged_df[time_column] <= '2023-11-24 00:00:00')]
    #short_df = merged_df[(merged_df[time_column] >= '2019-01-01 00:00:00') & (merged_df[time_column] <= '2019-01-03 00:00:00')]
elif file == '30017':
    #short_df = merged_df[(merged_df[time_column] >= '2016-12-25 00:00:00') & (merged_df[time_column] <= '2016-12-29 00:00:00')]
    #short_df = merged_df[(merged_df[time_column] >= '2024-09-26 00:00:00') & (merged_df[time_column] <= '2024-09-28 00:00:00')]
    #short_df = merged_df[(merged_df[time_column] >= '2023-06-01 00:00:00') & (merged_df[time_column] <= '2023-06-03 00:00:00')]
    #short_df = merged_df[(merged_df[time_column] >= '2024-06-04 00:00:00') & (merged_df[time_column] <= '2024-06-06 00:00:00')]
    short_df = merged_df[(merged_df[time_column] >= '2022-04-26 00:00:00') & (merged_df[time_column] <= '2022-04-30 00:00:00')]
    #short_df = merged_df[(merged_df[time_column] >= '2023-06-28 00:00:00') & (merged_df[time_column] <= '2023-06-30 00:00:00')]
    #short_df = merged_df[(merged_df[time_column] >= '2024-07-04 00:00:00') & (merged_df[time_column] <= '2024-07-08 00:00:00')]
    #short_df = merged_df[(merged_df[time_column] >= '2023-03-16 00:00:00') & (merged_df[time_column] <= '2023-03-18 00:00:00')]
    #short_df = merged_df[(merged_df[time_column] >= '2023-04-12 00:00:00') & (merged_df[time_column] <= '2023-04-14 00:00:00')]
    #short_df = merged_df[(merged_df[time_column] >= '2024-09-02 00:00:00') & (merged_df[time_column] <= '2024-09-05 00:00:00')]
    #short_df = merged_df[(merged_df[time_column] >= '2023-05-09 00:00:00') & (merged_df[time_column] <= '2023-05-09 12:00:00')]
    #short_df = merged_df[(merged_df[time_column] >= '2024-01-30 22:00:00') & (merged_df[time_column] <= '2024-02-01 00:00:00')]
elif file == '30336':
    short_df = merged_df[(merged_df[time_column] >= '2023-12-21 00:00:00') & (merged_df[time_column] <= '2023-12-27 00:00:00')]
    #short_df = merged_df[(merged_df[time_column] >= '2023-03-17 00:00:00') & (merged_df[time_column] <= '2023-03-19 00:00:00')]
    #short_df = merged_df[(merged_df[time_column] >= '2023-12-18 00:00:00') & (merged_df[time_column] <= '2023-12-20 00:00:00')]
    #short_df = merged_df[(merged_df[time_column] >= '2023-12-02 00:00:00') & (merged_df[time_column] <= '2023-12-05 00:00:00')]
    #short_df = merged_df[(merged_df[time_column] >= '2023-12-04 00:00:00') & (merged_df[time_column] <= '2023-12-08 00:00:00')]
    #short_df = merged_df[(merged_df[time_column] >= '2013-12-04 00:00:00') & (merged_df[time_column] <= '2013-12-08 00:00:00')]
    #short_df = merged_df[(merged_df[time_column] >= '2023-08-08 00:00:00') & (merged_df[time_column] <= '2023-08-10 00:00:00')]
    #short_df = merged_df[(merged_df[time_column] >= '2024-11-04 00:00:00') & (merged_df[time_column] <= '2024-11-07 00:00:00')]
    #short_df = merged_df[(merged_df[time_column] >= '2024-12-04 00:00:00') & (merged_df[time_column] <= '2024-12-04 18:00:00')]
    #short_df = merged_df[(merged_df[time_column] >= '2024-04-29 00:00:00') & (merged_df[time_column] <= '2024-05-07 00:00:00')]
    #short_df = merged_df[(merged_df[time_column] >= '2023-04-12 00:00:00') & (merged_df[time_column] <= '2023-04-17 00:00:00')]
    #short_df = merged_df[(merged_df[time_column] >= '2024-12-04 00:00:00') & (merged_df[time_column] <= '2024-12-04 18:00:00')]
elif file == '25149':
    short_df = merged_df[(merged_df[time_column] >= '2024-03-13 04:00:00') & (merged_df[time_column] <= '2024-03-13 14:00:00')]
    #short_df = merged_df[(merged_df[time_column] >= '2023-04-21 00:00:00') & (merged_df[time_column] <= '2023-04-24 00:00:00')]
    #short_df = merged_df[(merged_df[time_column] >= '2023-01-02 00:00:00') & (merged_df[time_column] <= '2023-01-04 00:00:00')]
    #short_df = merged_df[(merged_df[time_column] >= '2022-05-03 00:00:00') & (merged_df[time_column] <= '2022-05-06 12:00:00')]
    #short_df = merged_df[(merged_df[time_column] >= '2023-04-12 00:00:00') & (merged_df[time_column] <= '2023-04-13 10:00:00')]
    #short_df = merged_df[(merged_df[time_column] >= '2013-12-04 20:00:00') & (merged_df[time_column] <= '2013-12-06 20:00:00')]
elif file == '20101':
    #short_df = merged_df[(merged_df[time_column] >= '2024-04-29 00:00:00') & (merged_df[time_column] <= '2024-05-01 18:00:00')]
    #short_df = merged_df[(merged_df[time_column] >= '2024-05-14 00:00:00') & (merged_df[time_column] <= '2024-05-20 00:00:00')]
    short_df = merged_df[(merged_df[time_column] >= '2024-01-08 03:00:00') & (merged_df[time_column] <= '2024-01-09 13:00:00')]
    #short_df = merged_df[(merged_df[time_column] >= '2023-04-04 00:00:00') & (merged_df[time_column] <= '2023-04-07 00:00:00')]
    #short_df = merged_df[(merged_df[time_column] >= '2024-08-15 08:00:00') & (merged_df[time_column] <= '2024-08-15 18:00:00')]
    #short_df = merged_df[(merged_df[time_column] >= '2023-04-22 00:00:00') & (merged_df[time_column] <= '2023-04-24 00:00:00')]
    #short_df = merged_df[(merged_df[time_column] >= '2020-02-08 00:00:00') & (merged_df[time_column] <= '2020-02-12 00:00:00')]
elif file == '31616':
    #short_df = merged_df[(merged_df[time_column] >= '2024-10-22 00:00:00') & (merged_df[time_column] <= '2024-10-22 22:00:00')]
    #short_df = merged_df[(merged_df[time_column] >= '2023-01-02 00:00:00') & (merged_df[time_column] <= '2023-01-05 00:00:00')]
    short_df = merged_df[(merged_df[time_column] >= '2023-02-09 00:00:00') & (merged_df[time_column] <= '2023-02-10 10:00:00')]
    #short_df = merged_df[(merged_df[time_column] >= '2023-09-09 00:00:00') & (merged_df[time_column] <= '2023-09-12 00:00:00')]
    #short_df = merged_df[(merged_df[time_column] >= '2023-07-07 00:00:00') & (merged_df[time_column] <= '2023-07-08 00:00:00')]
    #short_df = merged_df[(merged_df[time_column] >= '2023-02-17 00:00:00') & (merged_df[time_column] <= '2023-02-19 00:00:00')]
    #short_df = merged_df[(merged_df[time_column] >= '2024-01-25 00:00:00') & (merged_df[time_column] <= '2024-01-28 00:00:00')]
    #short_df = merged_df[(merged_df[time_column] >= '2024-01-08 00:00:00') & (merged_df[time_column] <= '2024-01-12 00:00:00')]
    #short_df = merged_df[(merged_df[time_column] >= '2023-05-16 00:00:00') & (merged_df[time_column] <= '2023-05-18 18:00:00')]
    #short_df = merged_df[(merged_df[time_column] >= '2023-07-12 00:00:00') & (merged_df[time_column] <= '2023-07-15 18:00:00')]
elif file == '29393':
    #short_df = merged_df[(merged_df[time_column] >= '2022-01-28 00:00:00') & (merged_df[time_column] <= '2022-02-02 18:00:00')]
    short_df = merged_df[(merged_df[time_column] >= '2014-01-02 00:00:00') & (merged_df[time_column] <= '2014-01-11 04:00:00')]
elif file == '20047':
    #short_df = merged_df[(merged_df[time_column] >= '2020-02-09 00:00:00') & (merged_df[time_column] <= '2020-02-10 00:00:00')]
    #short_df = merged_df[(merged_df[time_column] >= '2024-01-16 00:00:00') & (merged_df[time_column] <= '2024-01-18 00:00:00')]
    #short_df = merged_df[(merged_df[time_column] >= '2011-04-04 00:00:00') & (merged_df[time_column] <= '2011-04-08 00:00:00')]
    #short_df = merged_df[(merged_df[time_column] >= '2011-04-01 00:00:00') & (merged_df[time_column] <= '2011-04-04 00:00:00')]
    #short_df = merged_df[(merged_df[time_column] >= '2011-04-14 00:00:00') & (merged_df[time_column] <= '2011-04-17 00:00:00')]
    short_df = merged_df[(merged_df[time_column] >= '2019-05-05 00:00:00') & (merged_df[time_column] <= '2019-05-26 00:00:00')]
elif file == '30479':
    #short_df = merged_df[(merged_df[time_column] >= '2023-10-20 00:00:00') & (merged_df[time_column] <= '2023-10-22 00:00:00')]
    #short_df = merged_df[(merged_df[time_column] >= '2024-09-08 12:00:00') & (merged_df[time_column] <= '2024-09-15 04:00:00')]
    short_df = merged_df[(merged_df[time_column] >= '2013-07-10 00:00:00') & (merged_df[time_column] <= '2013-07-15 00:00:00')]
    #short_df = merged_df[(merged_df[time_column] >= '2023-07-13 00:00:00') & (merged_df[time_column] <= '2023-07-25 00:00:00')]
    #short_df = merged_df[(merged_df[time_column] >= '2021-10-01 00:00:00') & (merged_df[time_column] <= '2021-10-15 00:00:00')]

def two_in_one_graph(df, time_column, measurement_column, output_path):
    """
    Makes a plot containing two subplots. One presents the period on interest and the table below indicates the performance of the different spike detection tools.

    Input:
    -dataframe
    -time_column: name of column with timestamp [str]
    -data_column: name of column with data [str]
    -output path
    """
    relev_columns = [ 'QC_Seadb_manual_mask', 'QC_selene_mask']
    lable_title = ['DMI QC', 'CMEMS QC']
    markers = ["s" , "d"]
    colors = ['orange', 'blue']
    count = [None] * len(relev_columns) 
    grid_heights = [0.08, 0.0] 
    offset_value = np.zeros(len(df))+0.12
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8), gridspec_kw={'height_ratios': [6, 1]}, dpi=300)
    #relev_columns = [ 'QC_Seadb_manual_mask', 'QC_selene_mask', 'QC_ml_mask']
    #lable_title = ['DMI QC', 'CMEMS QC', 'SeaQC-X']
    #markers = ["s" , "d", "*"]
    #colors = ['orange', 'blue', 'green']
    #count = [None] * len(relev_columns) 
    #grid_heights = [0.16, 0.08, 0.0] 
    #offset_value = np.zeros(len(df))+0.2
    title = 'Comparison of QC Methods'
    # Create the figure and axes with shared x-axis
    #fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8), gridspec_kw={'height_ratios': [8, 3]}, dpi=300)

    # Plot the first graph (Sea Level Measurements)
    ax1.plot(df[time_column], df[measurement_column], color='black', label='Measurement', marker='o',  markersize=6, linestyle='None')
    highlight = df[df['QC_Seadb_manual_mask']]
    ax1.scatter(highlight[time_column], highlight[measurement_column], marker='s', s=90, color='orange', label= 'DMI QC', zorder=2)
    highlight = df[df['QC_selene_mask']]
    ax1.scatter(highlight[time_column], highlight[measurement_column], marker='d', s=80, color='blue', label= 'CMEMS QC', zorder=2)
    #highlight = df[df['QC_ml_mask']]
    #ax1.scatter(highlight[time_column], highlight[measurement_column], marker='*', s=130, color='green', label= 'SeaQC-X', zorder=2)

    ax1.set_ylabel('Water Level [m]')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_xlabel('Timestamp')
    ax1.tick_params(axis='x', labelbottom=True)
    #leg = ax1.legend(loc='upper right', frameon=True)
    #leg.get_frame().set_edgecolor('black')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))  # Format: YYY-MM-DD HH:MM
    #ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.set_xticks(ax1.get_xticks()[::3]) 

    # Plot the second graph (Icons)
    # Stack markers using offsets
    for i in range(len(relev_columns)):
        mask = df[relev_columns[i]]
        scatter_marker = markers[i]
        scatter_colors = colors[i]
        count[i] = len(offset_value[mask])
        if len(offset_value[mask]) == 0:
            ax2.scatter(df[time_column].iloc[0], offset_value[1], color='white', marker=scatter_marker, s=60, alpha=0.6, edgecolors='white', linewidth=0.5, label=lable_title[i])
        else:   
            ax2.scatter(df.loc[mask, time_column], offset_value[mask], color=scatter_colors, marker=scatter_marker, s=130, alpha=0.6, edgecolors=scatter_colors, linewidth=0.5, label=lable_title[i])
        offset_value -= 0.08

    # Add gridlines at specific heights
    for height in grid_heights:
        ax2.axhline(y=height, color='gray', linewidth=0.6, alpha=0.5)

    ax2.axis('off')
    #ax2.legend(loc='lower right', frameon=False)
    #Add multi-line text
    text_box = '\n\n'.join(f"""{c}:""" for c in lable_title)
    ax2.figure.text(-0.01, 0.065, text_box, transform=ax2.figure.transFigure, verticalalignment='bottom', horizontalalignment='left')
    text_box = '\n\n'.join(f"""= {c}""" for c in count)
    ax2.figure.text(0.9, 0.065, text_box, transform=ax2.figure.transFigure, verticalalignment='bottom', horizontalalignment='left')

    # Display the plot
    plt.subplots_adjust(hspace=0.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path,f"{title}-{measurement_column}_final1.png"),  bbox_inches="tight")
    plt.savefig(os.path.join(output_path,f"{title}-{measurement_column}_final1.eps"), format='eps', bbox_inches="tight")
    plt.close()

two_in_one_graph(short_df, time_column, measurement_column, output_path)

print('Done!')