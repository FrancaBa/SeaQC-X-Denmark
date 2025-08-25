#########################################################
## Written by frb for DMI QC Comparison (2024-2025)    ##
## Contains a method for a heatmap                     ##
#########################################################

import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import matplotlib.patches as patches

plt.rcParams.update({'font.size': 40})  # Set font size globally 

#Run station of interest
file = '29393' #'20101', '32048', '30336', '31616', '30017', '25149', '29393', '20047', '30479' 

#Output folder
output_path = f'/dmidata/users/frb/QC Work/2_Greenland&Denmark/output_paper/heatmap/{file}'
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
path_selene  = '/dmidata/users/frb/QC Work/2_Greenland&Denmark/Input_data/spatial/data_qc_Selene'
path_ml  = '/dmidata/users/frb/QC Work/2_Greenland&Denmark/Input_data/spatial/data_ml_gr_dk_50'
path_seadb  = '/dmidata/users/frb/QC Work/2_Greenland&Denmark/Input_data/spatial/raw_data'
#path_selene  = '/dmidata/users/frb/QC Work/2_Greenland&Denmark/Input_data/temporal/data_qc_Selene'
#path_ml  = '/dmidata/users/frb/QC Work/2_Greenland&Denmark/Input_data/temporal/data_ml_gr_dk_50'
#path_seadb  = '/dmidata/users/frb/QC Work/2_Greenland&Denmark/Input_data/temporal/raw_data'

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

years = merged_df[time_column].dt.year.unique()


#######################
## Make the heat map ##
#######################

def make_heatmap(column_1, column_2, label_1, label_2, name_1, name_2):
    weekly_counts = pd.DataFrame()

    for year in sorted(years):
        yearly_df = merged_df[merged_df[time_column].dt.year == year].copy()
        yearly_df['week'] = yearly_df[time_column].dt.isocalendar().week
        weekly_sum = yearly_df.groupby('week')[['QC_ml_mask', 'QC_selene_mask', 'QC_Seadb_manual_mask']].sum()
        weekly_sum = weekly_sum.astype('Int64')
        weekly_sum[f'{year}'] = weekly_sum[column_1] - weekly_sum[column_2]
        if weekly_counts.empty:
            weekly_counts = weekly_sum[[f'{year}']]
        else:
            weekly_counts = weekly_counts.join(weekly_sum[[f'{year}']], how='outer')

    if 2025 in years:
        del weekly_counts['2025']

    weekly_counts = weekly_counts.iloc[:-1]
    print(weekly_counts)
    heatmap_data = weekly_counts.T

    # Create custom annotation DataFrame
    annot_data = heatmap_data.copy()

    # Apply conditions for extreme values
    def cap_label_str(x):
        if pd.isna(x):
            return ""  
        if x > 20:
            return ">20"
        elif x < -20:
            return "<-20"
        else:
            return str(int(x))

    annot_data = annot_data.applymap(cap_label_str)
    print(annot_data)

    # Create a copy for coloring, but set extremes to NaN (which will show as white)
    color_data = heatmap_data.mask((heatmap_data > 20) | (heatmap_data < -20))
    color_data = color_data.astype(float)

    # Plot heatmap
    plt.figure(figsize=(len(heatmap_data.columns) / 1.2, 16))
    ax = sns.heatmap(
        color_data,
        annot=False,
        fmt="",
        cmap="coolwarm",
        vmin=-20,           # clip color scale lower bound
        vmax=20,            # clip color scale upper bound
        linewidths=0.01,    
        linecolor='lightgrey',
        cbar_kws={'label': f'{label_1} - {label_2}', 'pad': 0.01}
    )

    # Manually add annotations for all cells
    for i in range(heatmap_data.shape[0]):
        for j in range(heatmap_data.shape[1]):
            text = annot_data.iloc[i, j]
            ax.text(j + 0.5, i + 0.5, text, ha='center', va='center', color='black', fontsize=20)

    # Create a red rectangle outline
    #rect = patches.Rectangle(
    #    (0, 4),               # (x,y) lower left corner in data coords
    #    52,                    # width
    #    5,                    # height
    #    linewidth=7,
    #    edgecolor='darkred',
    #    facecolor='none'      # no fill
    #)

    #ax.add_patch(rect)

    plt.xlabel("Week")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path,f"{file}-{name_1}vs{name_2}.png"),  bbox_inches="tight")
    plt.close()


###############################
## Call the heat map function##
###############################

make_heatmap('QC_ml_mask', 'QC_selene_mask', 'SeaQC-X', 'CMEMS QC', 'ML', 'CMEMS')
make_heatmap('QC_ml_mask', 'QC_Seadb_manual_mask', 'SeaQC-X', 'Manual QC', 'ML', 'Manual')
make_heatmap('QC_Seadb_manual_mask', 'QC_selene_mask', 'Manual QC', 'CMEMS QC', 'Manual', 'CMEMS')
