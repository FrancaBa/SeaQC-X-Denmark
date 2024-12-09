import numpy as np
import pandas as pandas
import ruptures as rpt
import matplotlib.pyplot as plt
import datetime
import os

import source.helper_methods as helper
import source.qc_spike as qc_spike

class ShiftDetector():

    def __init__(self):
        self.zscore_threshold = 3
        self.rolling_window = 725
        self.min_periods = 30
        self.helper = helper.HelperMethods()

    def set_output_folder(self, folder):
        folder_path = os.path.join(folder,'shifted periods')
        self.folder_path_ruptures = os.path.join(folder,'shifted periods','ruptures')

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        if not os.path.exists(self.folder_path_ruptures):
            os.makedirs(self.folder_path_ruptures)

        self.helper.set_output_folder(folder_path)

    def detect_shifts_statistical(self, df, data_column_name, time_column, measurement_column):
        
        df['shifted_period'] = False
        df['remove_shifted_period'] = df[measurement_column].copy()
        shift_points = (df['segments'] != df['segments'].shift())

        test_df = df[(df[time_column].dt.year == 2014) & (df[time_column].dt.month == 1) & (df[time_column].dt.day == 24)]
        #test_df = df[(df[time_column].dt.year == 2008) & (df[time_column].dt.month == 12) & (df[time_column].dt.day > 21) & (df[time_column].dt.day < 24)]
        #test_df = df[(df[time_column].dt.year == 2010) & (df[time_column].dt.month == 3) & (df[time_column].dt.day > 17) & (df[time_column].dt.day < 20)]
        #test_df = df[(df[time_column].dt.year == 2007) & (df[time_column].dt.month == 9) & (df[time_column].dt.day > 17) & (df[time_column].dt.day < 20)]
        self.helper.plot_df(test_df[time_column], test_df[measurement_column],'Water Level', 'Timestamp', 'shifted period test')
        test_df['change'] = np.abs(np.diff(test_df[data_column_name], append=np.nan))
        #change_points= test_df[test_df['change'] > 0.045].index
        change_points= test_df[test_df['change'] > 0.01].index
        if change_points.any():
                    mask = np.diff(change_points, append=np.inf) > 1
                    filtered_changepoints = change_points[mask]
                    
                    # Get indices of non-NaN values in measurement column
                    non_nan_indices = test_df[~test_df[measurement_column].isna()].index.to_numpy()
                    # Compute the absolute difference between target indices and non-NaN indices using broadcasting
                    distances = np.abs(np.array(filtered_changepoints)[:, np.newaxis] - non_nan_indices)
                    # Find the index of the minimum distance for each target index
                    numbers = np.unique(non_nan_indices[np.argmin(distances, axis=1)])

                    # Set to keep track of numbers to remove
                    #to_remove = set()

                    # Check for differences of 5 between consecutive numbers
                    #for i in range(1, len(numbers)):
                    #    if abs(numbers[i] - numbers[i-1]) <= 15:
                    #        to_remove.add(numbers[i])
                    #        to_remove.add(numbers[i-1])

                    # Filter out the numbers to remove
                    #filtered_numbers = [num for num in numbers if num not in to_remove]
                    #print(filtered_numbers)

                    test = []
                    for elem in numbers:
                    #for elem in filtered_numbers:
                        previous_data = test_df[measurement_column].loc[:elem - 1].dropna()
                        if len(previous_data) >= 2:
                            prev2_wl = previous_data.iloc[-2]
                            prev_wl = previous_data.iloc[-1]
                        else:
                            prev2_wl = 100
                            prev_wl = previous_data.iloc[-1]
                        now_wl = test_df[measurement_column].loc[elem]
                        next_wl = test_df[measurement_column].loc[elem + 1:].dropna().iloc[0]
                        next2_wl = test_df[measurement_column].loc[elem + 1:].dropna().iloc[1]
                        print(prev2_wl, prev_wl,now_wl,next_wl, next2_wl)
                        #if abs(next2_wl-next_wl) < 0.02 and abs(now_wl-next_wl) > 0.05 and abs(prev_wl-now_wl) < 0.02:
                        if abs(next2_wl-next_wl) < 0.05 and abs(now_wl-next_wl) > 0.1 and abs(prev_wl-now_wl) < 0.05:
                            test.append(elem)
                        #elif abs(next_wl-now_wl) < 0.02 and abs(prev_wl-now_wl) > 0.05 and abs(prev2_wl-prev_wl) < 0.02:
                        elif abs(next_wl-now_wl) < 0.06 and abs(prev_wl-now_wl) > 0.1 and abs(prev2_wl-prev_wl) < 0.06:
                            test.append(elem)
                    print(test)
        
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
                self.helper.plot_df(relev_df[time_column], relev_df[measurement_column],'Water Level','Timestamp ', f'Measured water level-{start_index}')
                change_points= relev_df[relev_df['change'] > 0.01].index
                if change_points.any():
                    mask = np.diff(change_points, append=np.inf) > 1
                    filtered_changepoints = change_points[mask]
                        
                    # Get indices of non-NaN values in measurement column
                    non_nan_indices = relev_df[~relev_df[measurement_column].isna()].index.to_numpy()
                    # Compute the absolute difference between target indices and non-NaN indices using broadcasting
                    distances = np.abs(np.array(filtered_changepoints)[:, np.newaxis] - non_nan_indices)
                    # Find the index of the minimum distance for each target index
                    numbers = np.unique(non_nan_indices[np.argmin(distances, axis=1)])

                    test = []
                    for elem in numbers:
                        previous_data = relev_df[measurement_column].loc[:elem - 1].dropna()
                        if len(previous_data) >= 2:
                            prev2_wl = previous_data.iloc[-2]
                            prev_wl = previous_data.iloc[-1]
                        else:
                            prev2_wl = 100
                            prev_wl = previous_data.iloc[-1]
                        now_wl = relev_df[measurement_column].loc[elem]
                        next_wl = relev_df[measurement_column].loc[elem + 1:].dropna().iloc[0]
                        next2_wl = relev_df[measurement_column].loc[elem + 1:].dropna().iloc[1]
                        if abs(next2_wl-next_wl) < 0.06 and abs(now_wl-next_wl) > 0.1 and abs(prev_wl-now_wl) < 0.06:
                            test.append(elem)
                        elif abs(next_wl-now_wl) < 0.06 and abs(prev_wl-now_wl) > 0.1 and abs(prev2_wl-prev_wl) < 0.06:
                            test.append(elem)

                    print(test)
                    if test:
                        for start, end in zip(test[::2], test[1::2]):
                            df['shifted_period'][start:end-1] = True
                            df['remove_shifted_period'][start:end-1] = np.nan
                            self.helper.plot_two_df_same_axis(df.loc[start-1000:end+1000,time_column], df.loc[start-1000:end+1000, measurement_column],'Water Level', 'Water Level (original)', df.loc[start-1000:end+1000, 'remove_shifted_period'], 'Timestamp ', 'Water Level (correct)', f'Measured water level at shifted periods - {start}')
       
        #print details on the small distribution check
        if df['shifted_period'].any():
            ratio = (df['shifted_period'].sum()/len(df))*100
            print(f"There are {df['shifted_period'].sum()} shifted values in periods. This is {ratio}% of the overall dataset.")

        del df['remove_shifted_period']

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
                    #Search method for changepoint
                    algo = rpt.Binseg(model="l2").fit(np.array(df[interpolated_data_colum][start_index:end_index]))
                    #algo = rpt.Window(width=40, model="l2").fit(df[interpolated_data_colum][start_index:end_index].values)
                    #algo = rpt.BottomUp(model="l2").fit(df[interpolated_data_colum][start_index:end_index].values)

                    #They are slow and need more memory
                    #algo = rpt.Pelt(model="l2").fit(df[interpolated_data_colum][start_index:end_index].values)
                    #algo = rpt.KernelCPD(kernel="linear", min_size=2).fit(df[interpolated_data_colum][start_index:end_index].values)
                    # Detect change points (tune 'pen' to control sensitivity)
                    #change_points = algo.predict(n_bkps=test)
                    #print(datetime.datetime.now())
                    #change_points = algo.predict(pen=500)
                    change_points = algo.predict(pen=1000)
                    print(datetime.datetime.now())

                    # Plot the results
                    if change_points:
                        for z in range(0, len(change_points)-1):
                            plt.figure(figsize=(10, 6))
                            rpt.display(df[interpolated_data_colum][start_index:end_index].values, change_points)
                            plt.xlim(change_points[z]-3500, change_points[z]+3500)  # Limit the x-axis to focus on  certain indices
                            plt.title(f"Change Point Detection in Time Series - Graph {i, z}")
                            plt.savefig(os.path.join(self.folder_path_ruptures,f"change_point_detection_plot{i, z}.png"))
                            plt.close()

                    # Step 3: Visualize the detected change points
                    plt.figure(figsize=(10, 6))
                    rpt.display(df[interpolated_data_colum][start_index:end_index].values, change_points)
                    plt.title(f"Change Point Detection in Time Series - Graph {i}")
                    plt.savefig(os.path.join(self.folder_path_ruptures,f"change_point_detection_plot{i}.png")) 
                    plt.close()

        return df