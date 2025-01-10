import numpy as np
import pandas as pandas
import ruptures as rpt
import matplotlib.pyplot as plt
import datetime
import os
import random
import builtins

import source.helper_methods as helper

os.environ["PYDEVD_WARN_EVALUATION_TIMEOUT"] = "10"

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

    def filter_elements_lazy(self, test):
        """Yields filtered elements lazily to save memory."""
        it = iter(test)  # Turn the input into an iterator
        try:
            previous = next(it)  # Get the first element
            yield previous  # Always include the first element
            for current in it:  # Iterate through the rest
                if current - previous <= 60:
                    yield current
                previous = current
        except StopIteration:
            pass

    def detect_shifts_statistical(self, df, data_column_name, time_column, measurement_column, segment_column):
        
        df['shifted_period'] = False
        df['remove_shifted_period'] = df[measurement_column].copy()
        segment_points = (df[segment_column] != df[segment_column].shift())

        #test_df = df[(df[time_column].dt.year == 2014) & (df[time_column].dt.month == 1) & (df[time_column].dt.day == 24)]
                   
        for i in range(0,len(df[segment_column][segment_points]), 1):
            start_index = df[segment_column][segment_points].index[i]
            if i == len(df[segment_column][segment_points])-1:
                end_index = len(df)
            else:
                end_index = df[segment_column][segment_points].index[i+1]
            if df[segment_column][start_index] == 0:
                relev_df = df[start_index:end_index]
                #Get shifted points based on strong gradient
                relev_df['change'] = np.abs(np.diff(relev_df[data_column_name], append=np.nan))
                change_points= relev_df[relev_df['change'] > 0.04].index
                if change_points.any():
                    mask = np.diff(change_points, append=np.inf) > 1
                    filtered_changepoints = np.array(change_points[mask])
                        
                    # Get indices of non-NaN values in measurement column
                    non_nan_indices = relev_df[~relev_df[measurement_column].isna()].index.to_numpy()

                    relev_indices = []

                    for elem in filtered_changepoints:
                        closest_index = np.argmin(np.abs(non_nan_indices - elem))
                        closest_value = non_nan_indices[closest_index]
                        relev_indices.append(closest_value)

                    #for i in range(0, len(non_nan_indices), chunk_size): #This loop is only needed because of memory issues.
                    #    chunk = non_nan_indices[i:i+chunk_size]
                    #    chunk_distances = np.abs(filtered_changepoints[:, np.newaxis] - chunk)
                    #    distances.append(chunk_distances)

                    # Concatenate the chunks into the final distances array
                    #distances = np.concatenate(distances, axis=1)
                    # Compute the absolute difference between target indices and non-NaN indices using broadcasting
                    #Below line causes memory issues, thus use code above
                    #distances = np.abs(np.array(filtered_changepoints)[:, np.newaxis] - non_nan_indices)

                    # Find the index of the minimum distance for each target index
                    #numbers = np.unique(non_nan_indices[np.argmin(distances, axis=1)])
                    relev_indices = np.unique(relev_indices)
                    print(relev_indices)

                    shift_points = []
                    for elem in relev_indices:
                        previous_data = relev_df[measurement_column].loc[:elem - 1].dropna()
                        if len(previous_data) >= 2:
                            prev2_wl = previous_data.iloc[-2]
                            prev_wl = previous_data.iloc[-1]
                        elif len(previous_data) == 1:
                            prev2_wl = 100
                            prev_wl = previous_data.iloc[-1]
                        else:
                            prev2_wl = 0
                            prev_wl = 0
                        now_wl = relev_df[measurement_column].loc[elem]
                        if (elem) == relev_df.index[-1]:
                            next_wl = 100 #dummy values for beginning of segment
                            next2_wl = 200 #dummy values for beginning of segment
                        else:
                            next_wl = relev_df[measurement_column].loc[elem + 1:].dropna().iloc[0]
                            #print(relev_df.loc[elem + 1:].dropna(), relev_df.index[-1])
                            if relev_df.loc[elem + 1:].dropna().empty:
                                next2_wl = next_wl
                            else:
                                next2_wl = relev_df[measurement_column].loc[elem + 1:].dropna().iloc[1]
                        if abs(next2_wl-next_wl) < 0.06 and abs(now_wl-next_wl) > 0.3 and abs(prev_wl-now_wl) < 0.06:
                            shift_points.append(elem)
                        elif abs(next_wl-now_wl) < 0.06 and abs(prev_wl-now_wl) > 0.3 and abs(prev2_wl-prev_wl) < 0.06:
                            shift_points.append(elem)

                    if shift_points:
                        filtered_elements = [shift_points[0]] + [current for previous, current in zip(shift_points, shift_points[1:]) if current - previous <= 60]
                        for elem in filtered_elements:
                            df['shifted_period'][elem-60:elem+60] = True
                            df['remove_shifted_period'][elem-60:elem+60] = np.nan
                    
        true_indices = df['shifted_period'][df['shifted_period']].index
        for i in range(0, 41):
            min = builtins.max(0,(random.choice(true_indices))-2000)
            max = min + 2000                
            self.helper.plot_two_df_same_axis(df.loc[min:max,time_column], df.loc[min:max, 'remove_shifted_period'],'Water Level', 'Water Level (corrected)', df.loc[min:max, measurement_column], 'Timestamp ', 'Values removed', f'Statistical Shift {i}')
        
        #print details on the small distribution check
        if df['shifted_period'].any():
            ratio = (df['shifted_period'].sum()/len(df))*100
            print(f"There are {df['shifted_period'].sum()} shifted values in periods. This is {ratio}% of the overall dataset.")

        del df['remove_shifted_period']

        return df


    def detect_shifts_ruptures(self, df, data_column_name, interpolated_data_colum, segment_column):

        shift_points = (df[segment_column] != df[segment_column].shift())

        for i in range(0,len(df[segment_column][shift_points]), 1):
            start_index = df[segment_column][shift_points].index[i]
            if i == len(df[segment_column][shift_points])-1:
                end_index = len(df)
            else:
                end_index = df[segment_column][shift_points].index[i+1]
            if df[segment_column][start_index] == 0:
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
                    change_points = algo.predict(pen=2500)
                    print(datetime.datetime.now())

                    # Plot the results
                    if change_points:
                        for z in range(0, len(change_points)-1):
                            plt.figure(figsize=(10, 6))
                            rpt.display(df[interpolated_data_colum][start_index:end_index].values, change_points)
                            plt.xlim(change_points[z]-1500, change_points[z]+1500)  # Limit the x-axis to focus on  certain indices
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