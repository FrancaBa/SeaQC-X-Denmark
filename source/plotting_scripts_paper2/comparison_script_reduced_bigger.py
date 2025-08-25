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

plt.rcParams.update({'font.size': 16})  # Set font size globally to 18

#Run station of interest
file = '29393' #'20101', '32048', '30336', '31616', '30017', '25149', '29393', '20047', '30479' 

#Output table
output_path = f'/dmidata/users/frb/QC Work/2_Greenland&Denmark/output/{file}'
if not os.path.exists(output_path):
    os.makedirs(output_path)

#DF columns
time_column = 'Timestamp'
measurement_column = 'WaterLevel'
qc_column_selene = 'QCFlag Selene'
qc_column_seadb = 'QCFlag Seadb'
qc_column_ml_50 = 'QCFlag(ML GR+DK 50)'
qc_column_ml_200 = 'QCFlag(ML GR+DK 200)'
qc_column_ml_GR = 'QCFlag(ML GR)'


#Paths to relevant files
#path_selene  = '/dmidata/users/frb/QC Work/2_Greenland&Denmark/Input_data/temporal/data_qc_Selene'
#path_ml_50  = '/dmidata/users/frb/QC Work/2_Greenland&Denmark/Input_data/temporal/data_ml_gr_dk_50'
#path_ml_200  = '/dmidata/users/frb/QC Work/2_Greenland&Denmark/Input_data/temporal/data_ml_gr_dk_200'
#path_ml_GR  = '/dmidata/users/frb/QC Work/2_Greenland&Denmark/Input_data/temporal/data_ml_gr'
#path_seadb  = '/dmidata/users/frb/QC Work/2_Greenland&Denmark/Input_data/temporal/raw_data'
path_selene  = '/dmidata/users/frb/QC Work/2_Greenland&Denmark/Input_data/spatial/data_qc_Selene'
path_ml_50  = '/dmidata/users/frb/QC Work/2_Greenland&Denmark/Input_data/spatial/data_ml_gr_dk_50'
path_ml_200  = '/dmidata/users/frb/QC Work/2_Greenland&Denmark/Input_data/spatial/data_ml_gr_dk_200'
path_ml_GR  = '/dmidata/users/frb/QC Work/2_Greenland&Denmark/Input_data/spatial/data_ml_gr'
path_seadb  = '/dmidata/users/frb/QC Work/2_Greenland&Denmark/Input_data/spatial/raw_data'

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
df_ml_all = pd.read_csv(os.path.join(path_ml_50,f'{file}.csv'), sep=",", header=0)
df_ml_50 = pd.DataFrame(columns=[time_column, qc_column_ml_50])
df_ml_50[time_column] = df_ml_all.iloc[:, 0]
df_ml_50[measurement_column] = df_ml_all.iloc[:, 1]
df_ml_50[qc_column_ml_50] = df_ml_all.iloc[:, 3]
df_ml_50['QC_ml_mask_50'] = (df_ml_50[qc_column_ml_50] != 1.0).astype(bool)
df_ml_50[time_column] = pd.to_datetime(df_ml_50[time_column], format='%Y-%m-%d %H:%M:%S').dt.round('min')
print(df_ml_50)

#Open .csv file and fix the column names
df_ml_all = pd.read_csv(os.path.join(path_ml_200,f'{file}.csv'), sep=",", header=0)
df_ml_200 = pd.DataFrame(columns=[time_column, qc_column_ml_200])
df_ml_200[time_column] = df_ml_all.iloc[:, 0]
df_ml_200[measurement_column] = df_ml_all.iloc[:, 1]
df_ml_200[qc_column_ml_200] = df_ml_all.iloc[:, 3]
df_ml_200['QC_ml_mask_200'] = (df_ml_200[qc_column_ml_200] != 1.0).astype(bool)
df_ml_200[time_column] = pd.to_datetime(df_ml_200[time_column], format='%Y-%m-%d %H:%M:%S').dt.round('min')
print(df_ml_200)

#Open .csv file and fix the column names
df_ml_all = pd.read_csv(os.path.join(path_ml_GR,f'{file}.csv'), sep=",", header=0)
df_ml_GR = pd.DataFrame(columns=[time_column, qc_column_ml_GR])
df_ml_GR[time_column] = df_ml_all.iloc[:, 0]
df_ml_GR[measurement_column] = df_ml_all.iloc[:, 1]
df_ml_GR[qc_column_ml_GR] = df_ml_all.iloc[:, 3]
df_ml_GR['QC_ml_mask_GR'] = (df_ml_GR[qc_column_ml_GR] != 1.0).astype(bool)
df_ml_GR[time_column] = pd.to_datetime(df_ml_GR[time_column], format='%Y-%m-%d %H:%M:%S').dt.round('min')
print(df_ml_GR)

#Merge the various dfs
#merged_df = pd.merge(df_seadb, df_mads, on=[time_column, measurement_column], how='outer')
#merged_df = pd.merge(merged_df, df_gronsl, on=[time_column, measurement_column], how='left')
#merged_df['QC_gronsl_mask'] = merged_df['QC_gronsl_mask'].fillna(False).astype(bool)
#merged_df = pd.merge(merged_df, df_gronsl_ml, on=[time_column, measurement_column], how='left')
#merged_df['QC_gronsl_ml_mask'] = merged_df['QC_gronsl_ml_mask'].fillna(False).astype(bool)
#merged_df = pd.merge(merged_df, df_manual, on=[time_column, measurement_column], how='left')
#merged_df['QC_manual_mask'] = merged_df['QC_manual_mask'].fillna(False).astype(bool)
merged_df = pd.merge(df_seadb, df_ml_50, on=[time_column, measurement_column], how='left')
merged_df['QC_ml_mask_50'] = merged_df['QC_ml_mask_50'].fillna(False).astype(bool)
merged_df = pd.merge(merged_df, df_ml_200, on=[time_column, measurement_column], how='left')
merged_df['QC_ml_mask_200'] = merged_df['QC_ml_mask_200'].fillna(False).astype(bool)
merged_df = pd.merge(merged_df, df_ml_GR, on=[time_column, measurement_column], how='left')
merged_df['QC_ml_mask_GR'] = merged_df['QC_ml_mask_GR'].fillna(False).astype(bool)
merged_df = pd.merge(merged_df, df_selene, on=[time_column], how='left')
merged_df['QC_selene_mask'] = merged_df['QC_selene_mask'].fillna(False).astype(bool)

#Run the graph script for each week
# Create a 'year-week' column to group by
merged_df['YearWeek'] = merged_df[time_column].dt.to_period('W')
# Group by week
weekly_dfs = {str(f'{week}'): group.drop(columns='YearWeek') for week, group in merged_df.groupby('YearWeek')}

def two_in_one_graph(df, time_column, measurement_column, output_path, week_number):
    """
    Makes a plot containing two subplots. One presents the period on interest and the table below indicates the performance of the different spike detection tools.

    Input:
    -dataframe
    -time_column: name of column with timestamp [str]
    -data_column: name of column with data [str]
    -output path
    """
    relev_columns = [ 'QC_Seadb_manual_mask', 'QC_selene_mask', 'QC_ml_mask_50', 'QC_ml_mask_200', 'QC_ml_mask_GR']
    lable_title = ['DMI QC', 'CMEMS QC', 'SeaQC-X (DK-GR 50)', 'SeaQC-X (DK-GR 200)', 'SeaQC-X (GR 200)']
    markers = ["s" , "d", "*", "o", "v",]
    colors = ['orange', 'blue', 'green', 'red', 'fuchsia']
    count = [None] * len(relev_columns) 
    grid_heights = [0.32, 0.24, 0.16, 0.08, 0.0] 
    offset_value = np.zeros(len(df))+0.36
    title = 'Comparison of QC Methods'
    # Create the figure and axes with shared x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8), gridspec_kw={'height_ratios': [8, 5]}, dpi=300)

    # Plot the first graph (Sea Level Measurements)
    ax1.plot(df[time_column], df[measurement_column], color='black', label='Measurement', marker='o',  markersize=3, linestyle='None')
    highlight = df[df['QC_Seadb_manual_mask']]
    ax1.scatter(highlight[time_column], highlight[measurement_column], marker='s', s=90, color='orange', label= 'Manual QC', zorder=2)
    highlight = df[df['QC_selene_mask']]
    ax1.scatter(highlight[time_column], highlight[measurement_column], marker='d', s=80, color='blue', label= 'CMEMS QC', zorder=2)
    highlight = df[df['QC_ml_mask_50']]
    ax1.scatter(highlight[time_column], highlight[measurement_column], marker='*', s=100, color='green', label= 'SeaQC-X (50)', zorder=2)
    highlight = df[df['QC_ml_mask_200']]
    ax1.scatter(highlight[time_column], highlight[measurement_column], marker='o', s=80, color='red', label= 'SeaQC-X (200)', zorder=2)

    ax1.set_ylabel('Water Level [m]')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_xlabel('Timestamp')
    ax1.tick_params(axis='x', labelbottom=True)
    leg = ax1.legend(loc='upper right', frameon=True)
    leg.get_frame().set_edgecolor('black')
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
            ax2.scatter(df.loc[mask, time_column], offset_value[mask], color=scatter_colors, marker=scatter_marker, s=110, alpha=0.6, edgecolors=scatter_colors, linewidth=0.5, label=lable_title[i])
        offset_value -= 0.08

    # Add gridlines at specific heights
    for height in grid_heights:
        ax2.axhline(y=height, color='gray', linewidth=0.6, alpha=0.5)

    ax2.axis('off')
    #ax2.legend(loc='lower right', frameon=False)
    #Add multi-line text
    text_box = '\n\n'.join(f"""{c}:""" for c in lable_title)
    ax2.figure.text(-0.1, 0.06, text_box, transform=ax2.figure.transFigure, verticalalignment='bottom', horizontalalignment='left')
    text_box = '\n\n'.join(f"""= {c}""" for c in count)
    ax2.figure.text(0.95, 0.06, text_box, transform=ax2.figure.transFigure, verticalalignment='bottom', horizontalalignment='left')

    # Display the plot
    plt.subplots_adjust(hspace=0.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path,f"{title}- {week_number} -{measurement_column}.png"),  bbox_inches="tight")
    #plt.savefig(os.path.join(output_path,f"{title}-{measurement_column}_final5.eps"), format='eps', bbox_inches="tight")
    plt.close()

for week, df_week in weekly_dfs.items():
    week_number = week.replace('/', '-')
    print(week_number)
    two_in_one_graph(df_week, time_column, measurement_column, output_path, week_number)

print('Done!')