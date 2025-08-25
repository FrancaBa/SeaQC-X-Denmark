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

plt.rcParams.update({'font.size': 10})  # Set font size globally to 14

#Run station of interest
file = '29393' #'20101', '32048', '30336', '31616', '30017', '25149'

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
#path_seadb = '/dmidata/users/frb/QC Work/2_Greenland&Denmark/old_results/2014-2018/data_qc_seadb'
#path_seadb = '/dmidata/users/frb/QC Work/2_Greenland&Denmark/old_results/2023-2024/raw_data'
#path_ml = '/dmidata/users/frb/QC Work/2_Greenland&Denmark/old_results/2023-2024/data_qc_ML'
#path_gronsl_ml = '/dmidata/users/frb/QC Work/2_Greenland&Denmark/old_results/2023-2024/data_qc_GronSL_ML'
#path_gronsl = '/dmidata/users/frb/QC Work/2_Greenland&Denmark/old_results/2023-2024/data_qc_GronSL_no_ML'
#path_manual = '/dmidata/users/frb/QC Work/2_Greenland&Denmark/old_results/2014-2018/data_qc_manual_labelling(final)'
#path_manual = '/dmidata/users/frb/QC Work/2_Greenland&Denmark/old_results/2023-2024/manual_controlled'
#path_mads = '/dmidata/users/frb/QC Work/2_Greenland&Denmark/old_results/2014-2018/data_qc_mads'
#path_selene = '/dmidata/users/frb/QC Work/2_Greenland&Denmark/old_results/2023-2024/data_qc_Selene'
path_selene  = '/dmidata/users/frb/QC Work/2_Greenland&Denmark/old_results/new_stations'
path_ml  = '/dmidata/users/frb/QC Work/2_Greenland&Denmark/old_results/new_stations'
path_seadb  = '/dmidata/users/frb/QC Work/2_Greenland&Denmark/old_results/new_stations'

#Open .csv file and fix the column names
#df_gronsl_all = pd.read_csv(os.path.join(path_gronsl,f'{file}.csv'), sep=",", header=0)
#df_gronsl = pd.DataFrame(columns=[time_column, qc_column_gronsl])
#df_gronsl[time_column] = df_gronsl_all.iloc[:, 0]
#df_gronsl[measurement_column] = df_gronsl_all.iloc[:, 1]
#df_gronsl[qc_column_gronsl] = df_gronsl_all.iloc[:, 3]
#df_gronsl['QC_gronsl_mask'] = (df_gronsl[qc_column_gronsl] != 1.0).astype(bool)
#df_gronsl[time_column] = pd.to_datetime(df_gronsl[time_column], format='%Y-%m-%d %H:%M:%S').dt.round('min')
#print(df_gronsl)

#Open .csv file and fix the column names
#df_gronsl_all = pd.read_csv(os.path.join(path_gronsl_ml,f'{file}.csv'), sep=",", header=0)
#df_gronsl_ml = pd.DataFrame(columns=[time_column, qc_column_gronsl_ml])
#df_gronsl_ml[time_column] = df_gronsl_all.iloc[:, 0]
#df_gronsl_ml[measurement_column] = df_gronsl_all.iloc[:, 1]
#df_gronsl_ml[qc_column_gronsl_ml] = df_gronsl_all.iloc[:, 3]
#df_gronsl_ml['QC_gronsl_ml_mask'] = (df_gronsl_ml[qc_column_gronsl_ml] != 1.0).astype(bool)
#df_gronsl_ml[time_column] = pd.to_datetime(df_gronsl_ml[time_column], format='%Y-%m-%d %H:%M:%S').dt.round('min')
#print(df_gronsl_ml)

#Open .csv file and fix the column names
df_ml_all = pd.read_csv(os.path.join(path_ml,f'{file}.csv'), sep=",", header=0)
df_ml = pd.DataFrame(columns=[time_column, qc_column_ml])
df_ml[time_column] = df_ml_all.iloc[:, 0]
df_ml[measurement_column] = df_ml_all.iloc[:, 1]
df_ml[qc_column_ml] = df_ml_all.iloc[:, 3]
#df_ml['QC_ml_mask'] = df_ml[qc_column_ml].astype(bool)
df_ml['QC_ml_mask'] = (df_ml[qc_column_ml] != 1.0).astype(bool)
df_ml[time_column] = pd.to_datetime(df_ml[time_column], format='%Y-%m-%d %H:%M:%S').dt.round('min')
#df_ml[time_column] = pd.to_datetime(df_ml[time_column]).dt.strftime('%Y-%m-%d %H:%M')
#df_ml[time_column] = pd.to_datetime(df_ml[time_column])
#print(df_ml)

#Open .csv file and fix the column names
#df_manual_all = pd.read_csv(os.path.join(path_manual,f'{file}-labeled.csv'), sep=",", header=0)
#print(df_manual_all)
#df_manual = pd.DataFrame(columns=[time_column, qc_column_manual])
#df_manual[time_column] = df_manual_all.iloc[:, 1]
#df_manual[measurement_column] = df_manual_all.iloc[:, 2]
#df_manual[qc_column_manual] = df_manual_all.iloc[:, 3]
#df_manual['QC_manual_mask'] = (df_manual[qc_column_manual] == 1.0).astype(bool)
#df_manual[time_column] = pd.to_datetime(df_manual[time_column], format='%Y-%m-%dT%H:%M:%S.%fZ').dt.round('min')
#print(df_manual)

#Open .txt file and fix the column names
#df_mads_all = pd.read_csv(os.path.join(path_mads,f'{file}_flagged.txt'), sep=",", header=0)
#df_mads = pd.DataFrame(columns=[time_column, measurement_column, qc_column_mads])
#df_mads[time_column] = df_mads_all.iloc[:, 0]
#df_mads[measurement_column] = df_mads_all.iloc[:, 1]/100
#df_mads[qc_column_mads] = df_mads_all.iloc[:, 2]
#df_mads['QC_mads_mask'] = df_mads[qc_column_mads] != 1
#df_mads[time_column] = pd.to_datetime(df_mads[time_column], format='%Y-%m-%d %H:%M:%S').dt.round('min')

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

#Merge the various dfs
#merged_df = pd.merge(df_seadb, df_mads, on=[time_column, measurement_column], how='outer')
#merged_df = pd.merge(df_seadb, df_gronsl, on=[time_column, measurement_column], how='left')
#merged_df['QC_gronsl_mask'] = merged_df['QC_gronsl_mask'].fillna(False).astype(bool)
#merged_df = pd.merge(merged_df, df_gronsl_ml, on=[time_column, measurement_column], how='left')
#merged_df['QC_gronsl_ml_mask'] = merged_df['QC_gronsl_ml_mask'].fillna(False).astype(bool)
#merged_df = pd.merge(merged_df, df_manual, on=[time_column, measurement_column], how='left')
#merged_df['QC_manual_mask'] = merged_df['QC_manual_mask'].fillna(False).astype(bool)
merged_df = pd.merge(df_seadb, df_ml, on=[time_column, measurement_column], how='left')
merged_df['QC_ml_mask'] = merged_df['QC_ml_mask'].fillna(False).astype(bool)
merged_df = pd.merge(merged_df, df_selene, on=[time_column], how='left')
merged_df['QC_selene_mask'] = merged_df['QC_selene_mask'].fillna(False).astype(bool)
print(merged_df)

#Run the graph script for each week
# Create a 'year-week' column to group by
#merged_df['YearWeek'] = merged_df[time_column].dt.to_period('W')
merged_df['YearWeek'] = merged_df[time_column].dt.to_period('M')
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
    #Visualization of different spike detection methods
    #relev_columns = ['QC_Seadb_mask', 'QC_Seadb_manual_mask', 'QC_manual_mask', 'QC_mads_mask', 'QC_gronsl_mask', 'QC_gronsl_ml_mask', 'QC_ml_mask', 'QC_selene_mask']
    #lable_title = ['SeaDB', 'Manual SeaDB', 'Manual Control', 'Mads QC', 'GronSL QC', 'GronSL ML QC', 'ML QC', 'Selene QC']
    #markers = ["s" , "d", "o", "v", ">", "*", "P", "X"]
    #colors = ['black', 'yellow', 'green', 'blue', 'red', 'fuchsia', 'grey', 'orange']
    #count = [None] * len(relev_columns) 
    #grid_heights = [1.19, 1.02, 0.85, 0.68, 0.51, 0.34, 0.17, 0.0] 
    #offset_value = np.zeros(len(df))+1.27
    #title = 'Comparison of DMI`s QC Methods'
    # Create the figure and axes with shared x-axis
    #fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]}, dpi=300)

    #relev_columns = [ 'QC_Seadb_manual_mask', 'QC_selene_mask']
    #lable_title = ['Manual QC', 'CMEMS QC']
    #markers = ["s" , "d"]
    #colors = ['green', 'blue']
    #count = [None] * len(relev_columns) 
    #grid_heights = [0.05, 0.0] 
    #offset_value = np.zeros(len(df))+0.075
    #fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8), gridspec_kw={'height_ratios': [6, 1]}, dpi=300)

    relev_columns = [ 'QC_Seadb_manual_mask', 'QC_selene_mask', 'QC_ml_mask']
    lable_title = ['Manual QC', 'CMEMS QC', 'RF QC']
    markers = ["s" , "d", "*"]
    colors = ['orange', 'blue', 'green']
    count = [None] * len(relev_columns) 
    grid_heights = [0.1, 0.05, 0.0] 
    offset_value = np.zeros(len(df))+0.125
    title = 'Comparison of QC Methods'
    # Create the figure and axes with shared x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8), gridspec_kw={'height_ratios': [6, 1]}, dpi=300)

    # Plot the first graph (Sea Level Measurements)
    ax1.plot(df[time_column], df[measurement_column], color='black', label='Measurement', marker='o',  markersize=1.2, linestyle='None')
    highlight = df[df['QC_Seadb_manual_mask']]
    ax1.scatter(highlight[time_column], highlight[measurement_column], marker='s', s=50, color='orange', label= 'Manual QC', zorder=2)
    highlight = df[df['QC_selene_mask']]
    ax1.scatter(highlight[time_column], highlight[measurement_column], marker='d', s=40, color='blue', label= 'CMEMS QC', zorder=2)
    highlight = df[df['QC_ml_mask']]
    ax1.scatter(highlight[time_column], highlight[measurement_column], marker='*', s=60, color='green', label= 'RF QC', zorder=2)
    ax1.set_ylabel('Water Level [m]')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_xlabel('Timestamp')
    ax1.tick_params(axis='x', labelbottom=True)
    ax1.legend(loc='best', frameon=False)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))  # Format: YYY-MM-DD HH:MM
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.set_xticks(ax1.get_xticks()[::3]) 

    # Plot the second graph (Icons)
    # Stack markers using offsets
    for i in range(len(relev_columns)):
        mask = df[relev_columns[i]]
        scatter_marker = markers[i]
        scatter_colors = colors[i]
        count[i] = len(offset_value[mask])
        if len(offset_value[mask]) == 0:
            ax2.scatter(df[time_column].iloc[0], offset_value[1], color='white', marker=scatter_marker, s=50, alpha=0.6, edgecolors='white', linewidth=0.5, label=lable_title[i])
        else:   
            ax2.scatter(df.loc[mask, time_column], offset_value[mask], color=scatter_colors, marker=scatter_marker, s=50, alpha=0.6, edgecolors=scatter_colors, linewidth=0.5, label=lable_title[i])
        offset_value -= 0.05
        #offset_value -= 0.17

    # Add gridlines at specific heights
    for height in grid_heights:
        ax2.axhline(y=height, color='gray', linewidth=0.6, alpha=0.5)

    ax2.axis('off')
    #ax2.legend(loc='lower right', frameon=False)
    #Add multi-line text
    text_box = '\n\n'.join(f"""= {c}""" for c in lable_title)
    ax2.figure.text(-0.03, 0.038, text_box, transform=ax2.figure.transFigure, fontsize=10, verticalalignment='bottom', horizontalalignment='left')
    #ax2.figure.text(-0.03, 0.05, text_box, transform=ax2.figure.transFigure, verticalalignment='bottom', horizontalalignment='left')
    text_box = '\n\n'.join(f"""= {c}""" for c in count)
    ax2.figure.text(1, 0.038, text_box, transform=ax2.figure.transFigure, fontsize=10, verticalalignment='bottom', horizontalalignment='left')
    #ax2.figure.text(1, 0.05, text_box, transform=ax2.figure.transFigure, verticalalignment='bottom', horizontalalignment='left')

    # Display the plot
    plt.subplots_adjust(hspace=0.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path,f"{title}- {week_number} -{measurement_column}.png"),  bbox_inches="tight")
    plt.close()

for week, df_week in weekly_dfs.items():
    week_number = week.replace('/', '-')
    print(week_number)
    two_in_one_graph(df_week, time_column, measurement_column, output_path, week_number)
    #two_in_one_graph(df_week, time_column, 'cleaned_measurement', output_path, week)

# Select only the columns you want to save
#df_labelling = merged_df.copy()
#df_labelling.loc[:, time_column] = df_labelling[time_column].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
#df_labelling['series'] = 'A'
#df_labelling['value'] = df_labelling[measurement_column]
#df_labelling['label'] = df_labelling[['QC_Seadb_manual_mask', 'QC_gronsl_mask', 'QC_selene_mask']].any(axis=1).astype(int)
#columns_to_save = ['series', time_column, 'value', 'label']  # Specify the desired columns
#df_labelling_final = df_labelling[columns_to_save]

# Save to a CSV file with comma-delimited format
#file_name = f"{file}-labelling.csv"
#df_labelling_final.to_csv(os.path.join(output_path, file_name), index=False)
    
print('Done!')