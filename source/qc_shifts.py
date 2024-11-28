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
        
        test_df = df[(df[time_column].dt.year == 2014) & (df[time_column].dt.month == 1) & (df[time_column].dt.day == 24)]
        test_df['change'] = np.abs(np.diff(test_df[data_column_name], append=np.nan))
        change_points= test_df[test_df['change'] > 0.04].index
        mask = np.diff(change_points, append=np.inf) > 1
        filtered_changepoints = change_points[mask]

        #Remove spikes from changepoints and only mark shifted periods
        too_close_indices = np.where(np.diff(filtered_changepoints) < 20)[0]
        remove_indices = set(too_close_indices).union(set(too_close_indices + 1))
        mask = np.array([i not in remove_indices for i in range(len(filtered_changepoints))])
        result = filtered_changepoints[mask]
        
        df['shifted_period'] = False
        shift_points = (df['segments'] != df['segments'].shift())

        for i in range(0,len(df['segments'][shift_points]), 1):
            start_index = df['segments'][shift_points].index[i]
            if i == len(df['segments'][shift_points])-1:
                end_index = len(df)
            else:
                end_index = df['segments'][shift_points].index[i+1]
            if df['segments'][start_index] == 0:

                df_relev = df[start_index:end_index]
                df_relev['change'] = np.abs(np.diff(df_relev[data_column_name], append=np.nan))
                change_points= df_relev[df_relev['change'] > 0.04].index
                #This is to removes spikes - here I want to detect shifted periods
                diffs_prev = np.abs(np.diff(change_points, prepend=np.nan))  # Difference with previous element
                diffs_next = np.abs(np.diff(change_points, append=np.nan))  # Difference with next element
                #Keep elements where the differences with both neighbors are >= 10
                mask = (diffs_prev >= 10) & (diffs_next >= 10)
                #self.helper.plot_df(df[time_column][change_points[0]-500:change_points[0]+500], df[data_column_name][change_points[0]-500:change_points[0]+500],'Water Level','Timestamp ','testing - outlier 0')

                # Step 4: Mark Period Between Changepoints
                mask = np.zeros_like(data, dtype=bool)
                if len(changepoints) >= 2:
                    mask[changepoints[0]:changepoints[-1]] = True
                    #print details on the small distribution check
                    if df['statistical_outlier'][start_index:end_index].any():
                        ratio = (df['statistical_outlier'][start_index:end_index].sum()/len(df))*100
                        print(f"There are {df['statistical_outlier'][start_index:end_index].sum()} shifted values according to moving windows in this timeseries. This is {ratio}% of the overall dataset.")
                        self.helper.plot_two_df_same_axis(df[time_column][start_index:end_index], df[data_column_name][start_index:end_index],'Water Level', 'Water Level (corrected)', df[measurement_column][start_index:end_index], 'Timestamp', 'Water Level (measured)',f'Graph {i}- Shifted periods based on moving window')
                        for i in range(1, 41):
                            min = (random.choice(change_index))-500
                            max = min + 500
                            self.helper.plot_two_df_same_axis(df[time_column][min:max], df[data_column_name][min:max],'Water Level', 'Water Level (corrected)', df[measurement_column][min:max], 'Timestamp', 'Water Level (measured)',f'Graph {i, min}- Shifted periods based on z-score')

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