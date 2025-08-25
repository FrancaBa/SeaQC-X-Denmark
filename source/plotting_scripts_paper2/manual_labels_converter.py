#Convert manual labelled data to fix GRONSL QC format for ML training

import os
import numpy as np
import pandas as pd

#Run station of interest
file = '20101' #'20101', '32048', '30336', '31616', '30017', '25149'

#Input data
time_column = 'Timestamp' #'Timestamp'
measurement_column = 'value' #'WaterLevel'
qc_column_manual = 'label' #'QCFlag Manual'
path_manual = '/dmidata/users/frb/QC Work/2_Greenland&Denmark/old_results/2014-2018/data_qc_manual_labelling(final)'

# Open .csv file
df_manual_all = pd.read_csv(os.path.join(path_manual,f'{file}-labeled.csv'), sep=",", header=0)
#df_manual_all = pd.read_csv(os.path.join(path_manual,f'{file}.seadb_0'), sep=";", header=None)
print(df_manual_all)
df_manual =  pd.DataFrame(columns=['series', time_column, measurement_column, qc_column_manual])

# Convert labels to fit GronSL legend
#df_manual = df_manual.rename(columns={'flag': 'label'})
df_manual_all['label'] = df_manual_all['label'].replace(1, 3.0)
df_manual_all['label'] = df_manual_all['label'].replace(0, np.nan)
#df_manual['reg'] = df_manual['reg']/100
print(df_manual_all)

# Convert labels to fit Trainset
#print(df_manual_all[1])
#df_manual['series'] = 'A'
#df_manual[time_column] = df_manual_all.iloc[:, 1]
#df_manual[time_column] = pd.to_datetime(df_manual[time_column]).dt.tz_localize("UTC").dt.strftime('%Y-%m-%dT%H:%M:%SZ')
#df_manual[measurement_column] = df_manual_all.iloc[:, 2]/100
#df_manual[qc_column_manual] = df_manual_all.iloc[:, 5].astype(bool)
#print(df_manual)

# Save to a CSV file with comma-delimited format
file_name = f"{file}-labelling.csv"
df_manual_all.to_csv(os.path.join(path_manual, file_name), index=False)