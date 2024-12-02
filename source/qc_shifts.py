import numpy as np
import pandas as pandas
import ruptures as rpt
import matplotlib.pyplot as plt
import datetime
import os
import random 

from sklearn.ensemble import IsolationForest

import source.helper_methods as helper
import source.qc_spike as qc_spike

class ShiftDetector():

    def __init__(self):
        self.zscore_threshold = 3
        self.rolling_window = 725
        self.min_periods = 30
        self.helper = helper.HelperMethods()
    
        #Call method from detect spike values
        self.spike_detection = qc_spike.SpikeDetector()

    def set_output_folder(self, folder):
        folder_path = os.path.join(folder,'shifted periods')

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.helper.set_output_folder(folder_path)
        self.spike_detection.set_output_folder(folder_path)

    def detect_shifts_statistical(self, df, data_column_name, time_column, measurement_column):
        
        df['shifted_period'] = False
        shift_points = (df['segments'] != df['segments'].shift())

        #test_df = df[(df[time_column].dt.year == 2014) & (df[time_column].dt.month == 1) & (df[time_column].dt.day == 24)]

        for i in range(0,len(df['segments'][shift_points]), 1):
            start_index = df['segments'][shift_points].index[i]
            if i == len(df['segments'][shift_points])-1:
                end_index = len(df)
            else:
                end_index = df['segments'][shift_points].index[i+1]
            if df['segments'][start_index] == 0:
                relev_df = df[start_index:end_index]
                #Get shifted points based on strong gradient
                relev_df['change'] = np.abs(np.diff(relev_df[data_column_name], append=np.nan))
                change_points= relev_df[relev_df['change'] > 0.05].index
                if change_points.any():
                    mask = np.diff(change_points, append=np.inf) > 1
                    filtered_changepoints = change_points[mask]

                    #Remove spikes from changepoints and only mark shifted periods
                    too_close_indices = np.where((np.diff(filtered_changepoints) < 20) & (np.diff(filtered_changepoints) > 200))[0]
                    remove_indices = set(too_close_indices).union(set(too_close_indices + 1))
                    mask = np.array([i not in remove_indices for i in range(len(filtered_changepoints))])
                    result = np.array(filtered_changepoints[mask])

                    # Get indices of non-NaN values in column 'a'
                    non_nan_indices = relev_df[~relev_df[measurement_column].isna()].index.to_numpy()
                    # Compute the absolute difference between target indices and non-NaN indices using broadcasting
                    distances = np.abs(result[:, np.newaxis] - non_nan_indices)
                    # Find the index of the minimum distance for each target index
                    closest_indices = non_nan_indices[np.argmin(distances, axis=1)]
                    print(closest_indices)

                    if closest_indices.any():
                        for start, end in zip(closest_indices[::2], closest_indices[1::2]):
                            df['shifted_period'][start:end-1] = True
                            df[measurement_column][start:end-1] = np.nan
                    
                            self.helper.plot_df(relev_df.loc[start-1000:end+1000,time_column], relev_df.loc[start-1000:end+1000, measurement_column],'Water Level','Timestamp ',f'testing - outlier {start}')

        #print details on the small distribution check
        if df['shifted_period'].any():
            ratio = (df['shifted_period'].sum()/len(df))*100
            print(f"There are {df['shifted_period'].sum()} shifted values in periods. This is {ratio}% of the overall dataset.")

        return df

    def detect_shifts_ruptures(self, df, data_column_name, interpolated_data_colum):

        shift_points = (df['segments'] != df['segments'].shift())

        for i in range(0,len(df['segments'][shift_points]), 1):
            start_index = df['segments'][shift_points].index[i]
            if i == len(df['segments'][shift_points])-1:
                end_index = len(df)
            else:
                end_index = df['segments'][shift_points].index[i+1]
            if df['segments'][start_index] == 0:
                if not np.isnan(df[data_column_name][start_index]):
                    # Detect shifts in mean using Pruned Exact Linear Time approach
                    #l2: For detecting shifts in the mean.
                    #l1: For detecting changes in the median.
                    #rbf: For non-linear changes.
                    #normal: For normal-distributed data.
                    print(datetime.datetime.now())
                    print(f'Segment is {end_index - start_index} entries long.')
                    print(f'This segments')
                    #Search method for changepoint
                    #algo = rpt.Binseg(model="l2").fit(np.array(df[interpolated_data_colum][start_index:end_index]))
                    #algo = rpt.Window(width=40, model="l2").fit(df[interpolated_data_colum][start_index:end_index].values)
                    algo = rpt.BottomUp(model="l2").fit(df[interpolated_data_colum][start_index:end_index].values)

                    #They are slow
                    #algo = rpt.Pelt(model="l2").fit(df[interpolated_data_colum][start_index:end_index].values)
                    #algo = rpt.KernelCPD(kernel="linear", min_size=2).fit(df[interpolated_data_colum][start_index:end_index].values)
                    # Detect change points (tune 'pen' to control sensitivity)
                    #change_points = algo.predict(n_bkps=test)
                    #print(datetime.datetime.now())
                    change_points = algo.predict(pen=500)
                    print(datetime.datetime.now())
                    # Plot the results
                    # Step 3: Visualize the detected change points
                    plt.figure(figsize=(10, 6))
                    rpt.display(df[interpolated_data_colum][start_index:end_index].values, change_points)
                    plt.title(f"Change Point Detection in Time Series - Graph {i}")
                    plt.savefig(f"change_point_detection_plot{i}.png") 
                    plt.close()

                    #for z in range(0, len(change_points)-1):
                    #    plt.figure(figsize=(10, 6))
                    #    rpt.display(df[interpolated_data_colum][start_index:end_index].values, change_points)
                    #    plt.xlim(change_points[z]-2500, change_points[z]+2500)  # Limit the x-axis to focus on  certain indices
                    #    plt.title(f"Change Point Detection in Time Series - Graph {i, z}")
                    #    plt.savefig(f"change_point_detection_plot{i, z}.png") 
                    #    plt.close()

                    print(datetime.datetime.now())
                    print('hey 3.0')

        return df
    
    def unsupervised_outlier_detection(self, df, data_column_name, interpolated_data_colum, time_column):

        shift_points = (df['segments'] != df['segments'].shift())

        for i in range(0,len(df['segments'][shift_points]), 1):
            start_index = df['segments'][shift_points].index[i]
            if i == len(df['segments'][shift_points])-1:
                end_index = len(df)
            else:
                end_index = df['segments'][shift_points].index[i+1]
            if df['segments'][start_index] == 0:
                if not np.isnan(df[data_column_name][start_index]):
        
                    # Prepare data for Isolation Forest
                    # Isolation Forest expects input in a 2D array form
                    X = df[interpolated_data_colum][start_index:end_index].values.reshape(-1, 1)

                    # Fit Isolation Forest
                    model = IsolationForest(contamination=0.15, random_state=42)  # Adjust contamination as needed
                    anomaly = model.fit_predict(X)

                    # Add anomaly scores
                    score = model.decision_function(X)

                    # Identify anomalies
                    anomalies = df[anomaly == -1]

                    # Plot the results
                    plt.figure(figsize=(12, 6))
                    plt.plot(df[time_column], df[interpolated_data_colum][start_index:end_index], label='Time Series')
                    plt.scatter(anomalies[time_column], anomalies[interpolated_data_colum], color='red', label='Anomalies', zorder=5)
                    plt.title('Anomaly Detection with Isolation Forest')
                    plt.xlabel('Time')
                    plt.ylabel('Value')
                    plt.legend()
                    plt.show()