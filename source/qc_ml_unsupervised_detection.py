#######################################################################################################################
## Written by frb for GronSL project (2024-2025)                                                                     ##
## This is the main ML-plug-in script for the QC. It splits the ts in 3 different classes (good, probably bad, bad). ##
#######################################################################################################################

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import utide
import random
import matplotlib.dates as mdates 
import pickle
import csv

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import imblearn
print(imblearn.__version__)

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.renderers.default = "browser"

import source.helper_methods as helper
import source.qc_wavelet_analysis as wavelet_analysis

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, average_precision_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

class MLOutlierDetectionUNSU(): 

    def __init__(self):

        self.helper = helper.HelperMethods()

        #Initialize encoder
        self.le = LabelEncoder()


        #Empty element to store different texts about QC tests and save as a summary document
        self.information = []
        self.scores = []
        self.scores.append("PR-AUC")
        self.scores.append("Precision")
        self.scores.append("Recall")

    def set_station(self, station):
        self.station = station
        self.information.append([f'The following text summarizes the supervised ML spike detection performed on the manual labelled data.', '\n', f'{station} has been used for training:', '\n'])
        
    def set_tidal_components_file(self, folder_path):
        self.tidal_infos = []
        #Open .csv file and fix the column names
        for file in os.listdir(folder_path):
            self.tidal_infos.append(os.path.join(folder_path, file))

    #Create output folder to save results
    def set_output_folder(self, folder_path):
        self.folder_path = folder_path

        #generate output folder for graphs and other docs
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

        self.helper.set_output_folder(self.folder_path)

    #set input table to correct heading
    def set_column_names(self, time_column, measurement_column, qc_column):
        self.time_column = time_column
        self.measurement_column = measurement_column
        self.qc_column = qc_column

    def import_data(self, folder_path):

        self.dfs_training = {}
        self.dfs_testing = {}
        #Open .csv file and fix the column names
        for file in os.listdir(folder_path):  # Loops through files in the current directory
            print(file)
            df_name = file.split("-")[0].strip("0123456789")
            df = pd.read_csv(os.path.join(folder_path,file), sep=",", header=0)
            df = df[[self.time_column, self.measurement_column, self.qc_column]]
            df[self.measurement_column] = df[self.measurement_column] - df[self.measurement_column].mean()
            df[self.qc_column] = df[self.qc_column].fillna(1).astype(int)
            #FOR NOW: Only a binary issue of spikes/non-spikes
            df[self.qc_column] = df[self.qc_column].replace(4, 3)
            df[self.time_column] = pd.to_datetime(df[self.time_column]).dt.round('min')
            #Add features
            tidal_signal= [path for path in self.tidal_infos if df_name in path]
            df = self.add_features(df, tidal_signal[0], df_name)
            #Analyse imported data
            anomaly = np.isin(df[self.qc_column], [3,4])
            self.basic_plotter_no_xaxis(df, f'Manual QC for {df_name} station',anomaly)
            #Merge dataset according to station for training
            if df_name in self.dfs_training.keys(): 
                merged_df = pd.concat([self.dfs_training[f"{df_name}"], df])
                merged_df = merged_df.sort_values(by='Timestamp').reset_index(drop=True)
                self.dfs_training[f"{df_name}"] = merged_df.copy()
            else:
                self.dfs_training[f"{df_name}"] = df.copy()
            anomaly = np.isin(self.dfs_training[f"{df_name}"][self.qc_column], [3,4])
            self.basic_plotter_no_xaxis(self.dfs_training[f"{df_name}"], f'Manual QC for {df_name} station combined',anomaly)
            #Keep dataset according to station for testing
            if file.split("-")[0] in self.dfs_testing.keys(): 
                merged_df = pd.concat([self.dfs_testing[f"{file.split("-")[0]}"], df])
                merged_df = merged_df.sort_values(by='Timestamp').reset_index(drop=True)
                self.dfs_testing[f"{file.split("-")[0]}"] = merged_df.copy()
            else:
                self.dfs_testing[f"{file.split("-")[0]}"] = df.copy()
            if (df[self.qc_column] == 4).any():
                raise Exception('Code is build for binary classification. QC flags can currently only be 1 and 3!')
    
    def run_unsupervised(self):
        
        print(self.dfs_testing.keys())
        print(self.dfs_training.keys())
        dfs_testing_new = {}
        dfs_train = {}
        for elem in self.dfs_training:
            #Decied if whole station df is used for training or only subset. If subset, split here
            df_train = self.dfs_training[elem].copy()
            anomalies = np.isin(df_train[self.qc_column], [3])
            self.basic_plotter_no_xaxis(df_train, f'QC classes (for training data) - Subset {elem}', anomalies)
            #Preprocess the training data if needed
            df_train = self.preprocessing_data(df_train)
            #Add features to the original data series and the training data
            tidal_signal = [path for path in self.tidal_infos if elem in path]
            df_train = self.add_features(df_train, tidal_signal[0], elem)
            dfs_train[f"{elem}"] = df_train
            dfs_testing_new[f"{elem}"] = self.dfs_training[elem].copy()

        # Merge all DataFrames in the dictionary for training_df
        merged_df = pd.concat(dfs_train.values())
        df_train = merged_df.sort_values(by='Timestamp').reset_index(drop=True)
        unique_values, counts = np.unique(df_train[self.qc_column], return_counts=True)
        for value, count in zip(unique_values, counts):
            self.information.append(f"Class {value} exists {count} times in training dataset. This is {count/len(df_train)}.")
            print(f"Class {value} exists {count} times in training dataset. This is {count/len(df_train)}.")
            self.information.append('\n')

        X_train = df_train[self.features].values
        y_train = self.le.fit_transform(df_train[self.qc_column].values)

        #Visualize training data
        anomalies = np.isin(df_train[self.qc_column], [3])
        self.basic_plotter_no_xaxis(df_train, 'QC classes (full dataset)', anomalies)

        #Fit ML model to data
        self.get_model_unsupervised(X_train, y_train)

        #Runs model again for manual labelled dataset and check the score
        for elem in dfs_testing_new:
                self.run_testing_model(dfs_testing_new[elem], elem)
        
        self.save_to_txt()
        self.save_to_csv()


    def preprocessing_data(self, df):
        #Undersampling: Mark 10% of random good rows to be deleted
        #rows_to_drop = df[df[self.qc_column]==1].sample(frac=0.1, random_state=42).index
        #Drop selected rows
        #df = df.drop(rows_to_drop).reset_index(drop=True).copy()
        
        #Oversampling:Filter rows which bad qc flag and multiply them and assign them randomly 
        #bad_rows = df_res[df_res[self.qc_column].isin([3,4])].copy()
        #bad_rows[self.time_column] = bad_rows[self.time_column]  + pd.to_timedelta(np.random.randint(1, 60, size=len(bad_rows)), unit='m')
        #df_res_new = pd.concat([df_res, bad_rows]).sort_values(by=self.time_column).drop_duplicates(subset=self.time_column, keep='last').reset_index(drop=True)
        #df_res_new = pd.concat([bad_rows, df_res]).sort_values(by=self.time_column).drop_duplicates(subset=self.time_column, keep='first').reset_index(drop=True)
        
        # Oversampling:Select good rows and shift them up and down
        rows_shift_neg = df[df[self.qc_column]==1].sample(frac=0.05, random_state=42)
        shift_array = np.random.uniform(0.04, 0.2, len(rows_shift_neg))
        rows_shift_neg[self.measurement_column] = rows_shift_neg[self.measurement_column] - shift_array
        rows_shift_neg[self.qc_column] = 3
        df = pd.concat([df, rows_shift_neg]).sort_values(by=self.time_column).drop_duplicates(subset=self.time_column, keep='last').reset_index(drop=True) 
        rows_shift_pos = df[df[self.qc_column]==1].sample(frac=0.05, random_state=42)
        shift_array = np.random.uniform(0.04, 0.2, len(rows_shift_pos))
        rows_shift_pos[self.measurement_column] = rows_shift_pos[self.measurement_column] + shift_array
        rows_shift_pos[self.qc_column] = 3
        df = pd.concat([df, rows_shift_pos]).sort_values(by=self.time_column).drop_duplicates(subset=self.time_column, keep='last').reset_index(drop=True)

        #Visualization resampled data
        anomalies = np.isin(df[self.qc_column], [3])
        self.basic_plotter_no_xaxis(df, 'Resampled QC classes (small distribution)', anomalies)
        self.zoomable_plot_df(df, 'QC classes - Resampled', anomalies)

        #Oversampling:Add some extreme values
        rows_shift_neg = df[df[self.qc_column]==1].sample(frac=0.005, random_state=42)
        shift_array = np.random.uniform(0.1, 1.5, len(rows_shift_neg))
        rows_shift_neg[self.measurement_column] = rows_shift_neg[self.measurement_column] - shift_array
        rows_shift_neg[self.qc_column] = 3
        df = pd.concat([df, rows_shift_neg]).sort_values(by=self.time_column).drop_duplicates(subset=self.time_column, keep='last').reset_index(drop=True) 
        rows_shift_pos = df[df[self.qc_column]==1].sample(frac=0.005, random_state=42)
        shift_array = np.random.uniform(0.1, 1.5, len(rows_shift_pos))
        rows_shift_pos[self.measurement_column] = rows_shift_pos[self.measurement_column] + shift_array
        rows_shift_pos[self.qc_column] = 3
        df = pd.concat([df, rows_shift_pos]).sort_values(by=self.time_column).drop_duplicates(subset=self.time_column, keep='last').reset_index(drop=True)

        #Analysis
        unique_values, counts = np.unique(df[self.qc_column], return_counts=True)
        for value, count in zip(unique_values, counts):
            self.information.append(f"Class {value} exists {count} times in the resampled dataset. This is {count/len(df)}.")
            print(f"Class {value} exists {count} times in the resampled dataset. This is {count/len(df)}.")
        self.information.append('\n')

        #Visualization resampled data
        anomalies = np.isin(df[self.qc_column], [3])
        self.basic_plotter_no_xaxis(df, 'Resampled QC classes (large distribution)', anomalies)
        self.zoomable_plot_df(df, 'QC classes - Resampled', anomalies)

        return df

    def add_features(self, df, tidal_signal, station):

        # Set it as index
        df.set_index(self.time_column, inplace=True)  
        
        #simple features like lag time and gradient
        df['lag_1'] = df['value'].shift(1)
        df['lag_2'] = df['value'].shift(2)
        df['lag_-1'] = df['value'].shift(-1)
        df['lag_-2'] = df['value'].shift(-2)
        df['gradient_1'] = df['value'] - df['value'].shift(1)
        df['gradient_-1'] = df['value'].shift(-1) - df['value']
        df['gradient_2'] = df['value'] - df['value'].shift(2)
        df['gradient_-2'] = df['value'].shift(-2) - df['value']
        #df['gradient_3'] = df['value'] - df['value'].shift(3)
        #df['gradient_-3'] = df['value'].shift(-3) - df['value']

        #Advanced features:
        #Generate wavelets
        #freq_hours = pd.Timedelta('1min').total_seconds() / 3600
        #self.wavelet_analyzer = wavelet_analysis.TidalWaveletAnalysis(target_sampling_rate_hours=freq_hours)
        #wavelet_outcomes = self.wavelet_analyzer.analyze(df, column='value', detrend='False', max_gap_hours=24)
        #df_wavelet = pd.DataFrame(index=range(len(wavelet_outcomes.times)))
        #test = wavelet_outcomes.power_spectrum.T.tolist()
        #df_wavelet = pd.DataFrame(test)
        #df_wavelet = pd.DataFrame({'wavelet_coef': df_wavelet.values.tolist()})
        #df_wavelet['timestep'] = df.index.min() + pd.to_timedelta(wavelet_outcomes.times, unit='h')
        #df_wavelet['timestep'] = df_wavelet['timestep'].dt.round('min')
        #merged_df = pd.merge(df, df_wavelet, left_index=True, right_on='timestep', how='left')
        #df['wavelet'] = merged_df['wavelet_coef'].values

        #Add tidal signal
        tidal_signal_series = self.reinitialize_utide_from_txt(df, tidal_signal, station)
        df['tidal_signal'] = tidal_signal_series
        rolling_corr = df['value'].rolling(10).corr(pd.Series(tidal_signal_series))
        df['tidal_corr'] = rolling_corr
        #self.features = ['lag_1', 'lag_2', 'lag_-1', 'lag_-2', 'tidal_signal', 'wavelet', 'gradient_1', 'gradient_-1', 'gradient_2', 'gradient_-2', 'gradient_3', 'gradient_-3']
        self.features = [self.measurement_column, 'tidal_signal', 'gradient_1', 'gradient_-1', 'gradient_2', 'gradient_-2']

        #reset index and timestep back to column
        df = df.reset_index().rename(columns={'index': self.time_column})
        
        return df
        
    def reinitialize_utide_from_txt(self,df, filename, station):
        # Load coef from utide
        with open(filename, "rb") as f:
            coef_loaded = pickle.load(f)
        
        # Load the new timeseries data
        new_time = pd.to_datetime(df.index)

        # Perform U-tide reconstruction with the new timeseries and parameters
        new_tide_results = utide.reconstruct(new_time, coef_loaded, verbose=False)

        fig, ax1 = plt.subplots(figsize=(18, 10))
        ax1.set_xlabel('Time')
        ax1.set_ylabel('WaterLevel')
        ax1.plot(new_time, df['value'], marker='o', markersize=3, color='black', linestyle='None', label = 'WaterLevel')
        ax1.plot(new_time, new_tide_results['h'], marker='o', markersize=3, color='blue', linestyle='None', alpha=0.6, label = 'TidalSignal')
        ax1.legend(loc='upper right')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d %H:%M'))
        # Ensure all spines (box edges) are visible
        for spine in ax1.spines.values():
            spine.set_visible(True)
        fig.tight_layout() 
        fig.savefig(os.path.join(self.folder_path,f"{station}-tidal_signal.png"),  bbox_inches="tight")
        plt.close()  # Close the figure to release memory

        # Return the reinitialized tide results
        return new_tide_results['h']
    
    def get_model_unsupervised(self, X, y):

        # Define the hyperparameter grid
        param_dist = {
            'n_estimators': np.arange(50, 501, 50),
            'max_samples': np.linspace(0.1, 1.0, 10),
            'contamination': [0.01, 0.05, 0.1, 'auto'],
            'max_features': np.linspace(0.5, 1.0, 5),
            'bootstrap': [True, False]
        }

        # Initialize the model
        iso_forest = IsolationForest(random_state=42)

        # Custom scorer for F1-score
        def f1_scorer(estimator, X, y):
            y_pred = estimator.predict(X)
            y_pred = (y_pred == -1).astype(int)  # Convert -1 (anomaly) to 1, 1 (normal) to 0
            return f1_score(y, y_pred)  # Compare with true labels

        # Wrap the function in make_scorer
        scorer = make_scorer(f1_scorer, greater_is_better=True, needs_proba=False)

        # Initialize RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=iso_forest,
            param_distributions=param_dist,
            n_iter=20,
            cv=3,
            scoring=scorer,
            random_state=42,
            n_jobs=-1
        )

        # Fit the model
        random_search.fit(X,  y)
        # Print the best parameters
        print("Best Parameters:", random_search.best_params_)
        # Evaluate the best model
        self.model = random_search.best_estimator_

        # Fit Isolation Forest
        #self.model = IsolationForest(contamination=0.1, n_estimators=200, random_state=42)  # Adjust contamination as needed
        #self.model.fit(X)

    def run_testing_model(self, df_test, station):
        #Print details on how testing dataset is expected to look like  
        unique_values, counts = np.unique(df_test[self.qc_column], return_counts=True)
        for value, count in zip(unique_values, counts):
            self.information.append(f"Class {value} exists {count} times in testing dataset. This is {count/len(df_test)}.")
            #print(f"Class {value} exists {count} times in testing dataset. This is {count/len(df_test)}.")

        title = f'QC classes (for testing data) for {station}'
        anomalies = np.isin(df_test[self.qc_column], [3])
        self.basic_plotter_no_xaxis(df_test, title, anomalies)

        #Extract test data
        X_test = df_test[self.features].values
        #X_test = df_test[[self.measurement_column, 'lag_1', 'lag_2', 'lag_-1', 'lag_-2', 'gradient_1', 'gradient_-1']].values
        #X_test = df_test.drop(columns=["Timestamp", "label"]).values
        y_test = self.le.fit_transform(df_test[self.qc_column].values)

        # Predictions
        y_pred = self.model.predict(X_test)
        # Convert IsolationForest output (-1 → 1, 1 → 0)
        y_pred = (y_pred == -1).astype(int)

        # Visualization predictions vs testing label   
        # Create confusion matix
        cm = confusion_matrix(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label=1)
        recall = recall_score(y_test, y_pred, pos_label=1)
        pr_auc = average_precision_score(y_test, y_pred)  # Precision-Recall AUC
        #print(f"Confusion Matrix for testing data for {station}:")
        #print(cm)
        #print("Proportion of the predicted minority class labels are actually correct:")
        #print(precision)
        #print("Proportion of actual minority class instances that are correctly predicted:")
        #print(recall)
        print(f"{station} PR-AUC: {pr_auc:.3f}")
        self.information.append(f"Confusion Matrix for testing data for {station}:")
        self.information.append(str(cm.tolist()))
        self.information.append("Precision (Proportion of the predicted minority class labels are actually correct)")
        self.information.append(str(precision))
        self.information.append("Recall (Proportion of actual minority class instances that are correctly predicted)")
        self.information.append(str(recall))
        self.information.append(f"PR-AUC: {pr_auc:.3f}")
        self.information.append('\n')
        self.scores.append(station)
        self.scores.append(str(pr_auc))
        self.scores.append(str(precision))
        self.scores.append(str(recall))

        # Identify anomalies: Visualization
        title = f"ML predicted QC classes (for testing data)- Binary -{station}"
        anomalies_pred = np.isin(y_pred, [1])
        #anomalies_pred = np.isin(y_pred, [3])
        self.basic_plotter_no_xaxis(df_test, title, anomalies_pred)
        self.zoomable_plot_df(df_test, title, anomalies, anomalies_pred)

    def basic_plotter_no_xaxis(self, df, title, anomalies):
        #Visualization of station data
        plt.figure(figsize=(12, 6))
        plt.plot(df[self.measurement_column], label='Time Series', color='black', marker='o',  markersize=0.5, linestyle='None')
        plt.scatter(df.index[anomalies], df[self.measurement_column][anomalies], color='red', label='QC Classes', zorder=0.5)
        plt.legend()
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Water Level [m]')
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder_path,f"{title}- Date: {df[self.time_column].iloc[0]}.png"),  bbox_inches="tight")
        plt.close() 

    def zoomable_plot_df(self, df, title, anomaly1, anomaly2=np.array([])):
        """
        Generates a zoomable graph for one series. This is very slow and memory intensive. (Only use it remotely or when really needed.)

        Input:
        -df [Pandas df]
        -Graph title [str]
        -Legend [str] (to describe measurement 1)
        """
        #Zoomable plot
        fig = go.Figure()
        # Add water level data
        fig.add_trace(go.Scatter(x=df[self.time_column], y= df[self.measurement_column], mode='markers', marker=dict(
            size=5,          # Marker size
            color='black',  # Marker color
            symbol='circle',   # Marker shape (e.g., 'circle', 'square', 'diamond')
            opacity=0.8       # Marker transparency
        )))
        # Add orig QC flags 
        fig.add_trace(go.Scatter(x=df[self.time_column][anomaly1], y= df[self.measurement_column][anomaly1], mode='markers', marker = {'color' : 'green'}))
        # Add QC flags
        if not anomaly2.size == 0:
            fig.add_trace(go.Scatter(x=df[self.time_column][anomaly2], y= df[self.measurement_column][anomaly2], mode='markers', marker = {'color' : 'red'}))
        # Update layout for a better view
        fig.update_layout(title=title, xaxis_title= 'Time', yaxis_title= 'Water Levels [m]', legend_title='Legend', hovermode='closest')
        # Enable zoom and pan
        fig.update_layout(xaxis=dict(rangeslider=dict(visible=True), type="date"))
        # Show the figure
        fig.show()

    def save_to_txt(self):
        """
        Save the print statements from each QC test to a common txt-file. This can be used later on to compare between stations and set-ups.
        """
        # Filepath to save the text file
        filename = f"ML_spike_summary.txt"
        file_path = os.path.join(self.folder_path, filename)

        # Save the list to a .txt file
        with open(file_path, "w") as file:
            for list in self.information:
                # Write each elem as a row
                file.write("".join(list) + "\n")

        print(f"ML QC test statements have been saved to {file_path}")

    def save_to_csv(self):

        file_path = os.path.join(self.folder_path, "scores.csv")
        with open(file_path, mode='w', newline='') as file:
            for list in self.scores:
                # Write each elem as a row
                file.write("".join(list) + "\n")

        print("CSV file 'scores.csv' created successfully.")
        