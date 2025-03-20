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
from sklearn.model_selection import GridSearchCV
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

class MLOutlierDetection(): 

    def __init__(self):

        self.helper = helper.HelperMethods()

        #Empty element to store different texts about QC tests and save as a summary document
        self.information = []

        #Define percentage of data used for training and testing (here: 80/20%)
        self.train_size = 0.90
        self.val_size = 0.10

        # Initialize encoder
        self.le = LabelEncoder()

    def set_station(self, station):
        self.station = station
        self.information.append(['The following text summarizes the supervised ML QC perfromed on measurements from ', station,':'])
    
    def set_tidal_components_file(self, filename):
        self.filename = filename
    #load config.json file to generate bitmask and flags after QC tests and to load threshold/parameters used for the QC tests
    def load_config_json(self, json_path):

        # Open and load JSON file containing the quality flag classification
        with open(json_path, 'r') as file:
            config_data = json.load(file)

        #Defines if ML tests should be carried out as well
        self.detide_mode = config_data['ML_mode']

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

    def set_missing_value_filler(self, missing_meas_value):
        #Dummy value for NaN-values in measurement series
        self.missing_meas_value = missing_meas_value 

    def import_data(self, folder_path, ending):

        combined_df = []
        combined_station_df = []
        self.dfs = {}
        #Open .csv file and fix the column names
        for file in os.listdir(folder_path):  # Loops through files in the current directory
            print(file)
            if file.endswith(ending) and self.station in file:
                print(self.station)
                station_df = pd.read_csv(os.path.join(folder_path,file), sep=",", header=0)
                station_df = station_df[[self.time_column, self.measurement_column, self.qc_column]]
                station_df[self.measurement_column] = station_df[self.measurement_column] - station_df[self.measurement_column].mean()
                station_df[self.qc_column] = station_df[self.qc_column].fillna(1).astype(int)
                station_df[self.qc_column] = station_df[self.qc_column].replace(4, 3)
                station_df[self.time_column] = pd.to_datetime(station_df[self.time_column]).dt.round('min')
                feature_df = station_df.copy()
                feature_df.set_index(self.time_column, inplace=True)  # Set it as index
                df = self.add_features(feature_df, True)
                station_df = df.reset_index()
                #merge features to correct df
                self.dfs[f"df_{file.split("-")[0]}"] = station_df.copy()
                combined_station_df.append(station_df)
                self.basic_plotter(station_df, f'Manual QC for {file} station - Main Analysis')
            elif file.endswith(ending):
                df = pd.read_csv(os.path.join(folder_path,file), sep=",", header=0)
                df = df[[self.time_column, self.measurement_column, self.qc_column]]
                df[self.measurement_column] = df[self.measurement_column] - df[self.measurement_column].mean()
                df[self.qc_column] = df[self.qc_column].fillna(1).astype(int)
                df[self.time_column] = pd.to_datetime(df[self.time_column]).dt.round('min')
                feature_df = df.copy()
                feature_df.set_index(self.time_column, inplace=True)  # Set it as index
                df_new = self.add_features(feature_df, False)
                #merge features to correct df
                df_big = df_new.reset_index()
                self.dfs[f"df_{file.split("-")[0]}"] = df_big.copy()
                self.basic_plotter(df_big, f'Manual QC for {file} station')
                combined_df.append(df_big)

        labelled_df = pd.concat(combined_df, ignore_index=True)
        labelled_df = labelled_df.sort_values(by=self.time_column).reset_index(drop=True)
        combined_station_df = pd.concat(combined_station_df, ignore_index=True)
        combined_station_df = combined_station_df.sort_values(by=self.time_column).reset_index(drop=True)
        self.basic_plotter_no_xaxis(combined_station_df, 'Manual QC for station')

        return labelled_df, combined_station_df
    
    def add_features(self, df, tidal_signal):
        
        #simple features like lag time and gradient
        df['lag_1'] = df['value'].shift(1)
        df['lag_2'] = df['value'].shift(2)
        df['lag_-1'] = df['value'].shift(-1)
        df['lag_-2'] = df['value'].shift(-2)
        df['gradient_1'] = df['value'] - df['value'].shift(1)
        df['gradient_-1'] = df['value'].shift(-1) - df['value']

        #Advanced features:
        #Generate wavelets
        freq_hours = pd.Timedelta('1min').total_seconds() / 3600
        self.wavelet_analyzer = wavelet_analysis.TidalWaveletAnalysis(target_sampling_rate_hours=freq_hours)
        wavelet_outcomes = self.wavelet_analyzer.analyze(df, column='value', detrend='False', max_gap_hours=24)
        df_wavelet = pd.DataFrame(index=range(len(wavelet_outcomes.times)))
        test = wavelet_outcomes.power_spectrum.T.tolist()
        df_wavelet = pd.DataFrame(test)
        df_wavelet = pd.DataFrame({'wavelet_coef': df_wavelet.values.tolist()})
        df_wavelet['timestep'] = df.index.min() + pd.to_timedelta(wavelet_outcomes.times, unit='h')
        df_wavelet['timestep'] = df_wavelet['timestep'].dt.round('min')
        merged_df = pd.merge(df, df_wavelet, left_index=True, right_on='timestep', how='left')
        #df['wavelet'] = merged_df['wavelet_coef'].values
        #Add tidal signal
        if tidal_signal:
            tidal_signal_series = self.reinitialize_utide_from_txt(df, self.filename)
            df['tidal_signal'] = tidal_signal_series
            self.features = ['lag_1', 'lag_2', 'lag_-1', 'lag_-2', 'tidal_signal', 'wavelet', 'gradient_1', 'gradient_-1']
        else:
            #all developed features
            self.features = ['lag_1', 'lag_2', 'lag_-1', 'lag_-2', 'wavelet', 'gradient_1', 'gradient_-1']
        
        return df

    def reinitialize_utide_from_txt(self,df, filename):
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
        fig.savefig(os.path.join(self.folder_path,f"tidal_signal.png"),  bbox_inches="tight")
        plt.close()  # Close the figure to release memory

        # Return the reinitialized tide results
        return new_tide_results['h']

    def basic_plotter_no_xaxis(self, df, title):
        #Visualization of station data
        anomalies = df[self.qc_column].isin([3, 4])
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

    def basic_plotter(self, df, title):
        #Visualization of station data
        anomalies = df[self.qc_column].isin([3, 4])
        plt.figure(figsize=(12, 6))
        plt.plot(df[self.time_column], df[self.measurement_column], label='Time Series', color='black', marker='o',  markersize=0.5, linestyle='None')
        plt.scatter(df[self.time_column][anomalies], df[self.measurement_column][anomalies], color='red', label='QC Classes', zorder=1)
        plt.legend()
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Water Level [m]')
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder_path,f"{title}- Date: {df[self.time_column].iloc[0]}.png"),  bbox_inches="tight")
        plt.close() 

    def zoomable_plot_df(self, df, title, legend_name = None):
        """
        Generates a zoomable graph for one series. This is very slow and memory intensive. (Only use it remotely or when really needed.)

        Input:
        -df [Pandas df]
        -Graph title [str]
        -Legend [str] (to describe measurement 1)
        """
        anomaly = df[self.qc_column].isin([3, 4])
        # Create a figure with Plotly
        fig = go.Figure()

        # Add water level data
        fig.add_trace(go.Scatter(x=df[self.time_column], y= df[self.measurement_column], mode='markers', marker=dict(
            size=5,          # Marker size
            color='rgb(255, 0, 0)',  # Marker color
            symbol='circle',   # Marker shape (e.g., 'circle', 'square', 'diamond')
            opacity=0.8       # Marker transparency
        ),name=legend_name))
        
        # Add QC flags
        fig.add_trace(go.Scatter(x=df[self.time_column][anomaly], y= df[self.measurement_column][anomaly], mode='markers', marker = {'color' : 'black'}))

        # Update layout for a better view
        fig.update_layout(title=title, xaxis_title= 'Time', yaxis_title= 'Water Levels', legend_title='Legend', hovermode='closest')

        # Enable zoom and pan
        fig.update_layout(xaxis=dict(rangeslider=dict(visible=True), type="date"))

        # Show the figure
        fig.show()

    def run_new(self, df, station_df):

        #df_train, df_test = self.split_dataset(station_df)
        df.loc[df[self.qc_column] == 4, self.qc_column] = 3
        station_df.loc[df[self.qc_column] == 4, self.qc_column] = 3
        
        # Train-test split (keep sequential order)
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        df_train = station_df

         #Analyse existing dataset
        unique_values, counts = np.unique(station_df[self.qc_column], return_counts=True)
        for value, count in zip(unique_values, counts):
            print(f"Class {value} exists {count} times in whole dataset. This is {count/len(station_df)}.")

        unique_values, counts = np.unique(df_train[self.qc_column], return_counts=True)
        for value, count in zip(unique_values, counts):
            print(f"Class {value} exists {count} times in training dataset. This is {count/len(df_train)}.")


        #X_train = df_train.drop(columns=["Timestamp", "label"]).values
        #X_test = df_test.drop(columns=["Timestamp", "label"]).values
        #X_train = df_train[[self.measurement_column, 'lag_1', 'lag_2', 'lag_-1', 'lag_-2', 'gradient_1', 'gradient_-1']].values
        X_train = df_train[[self.measurement_column, 'gradient_1', 'gradient_-1']].values
        y_train = df_train[self.qc_column].values

        title = 'QC classes (for training data)'
        anomalies = np.isin(df_train[self.qc_column], [3,4])
        plt.figure(figsize=(12, 6))
        plt.plot(df_train[self.measurement_column], label='Time Series', color='black', marker='o',  markersize=0.5)
        plt.scatter(df_train.index[anomalies], df_train[self.measurement_column][anomalies], color='red', label='QC Classes', zorder=1)
        plt.legend()
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Water Level')
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder_path,f"{title}.png"),  bbox_inches="tight")
        plt.close()

        # Train RandomForestClassifier
        self.model = RandomForestClassifier(n_estimators=200, random_state=42)
        self.model.fit(X_train, y_train)

        # Make predictions (train)
        y_pred_train = self.model.predict(X_train)
        #y_pred_train_prob = model.predict_proba(X_train)
        #y_pred_train = (y_pred_train_prob[:, 1] > 0.5).astype(int)

        # Create confusion matix
        cm = confusion_matrix(y_train, y_pred_train)
        print("Confusion Matrix for training data:")
        print(cm)

        # Identify anomalies
        anomalies_train = np.isin(y_pred_train, [3])
        title = 'ML predicted QC classes (for training data)- Binary'
        plt.figure(figsize=(12, 6))
        plt.plot(df_train[self.measurement_column], label='Time Series', color='black', marker='o',  markersize=0.5)
        plt.scatter(df_train.index[anomalies_train], df_train[self.measurement_column][anomalies_train], color='red', label='QC Classes', zorder=1)
        plt.legend()
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Water Level')
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder_path,f"{title}-pred-train.png"),  bbox_inches="tight")
        plt.close()

        for elem in self.dfs:
            self.run_testing(self.dfs[elem], elem)

    def run_testing(self, df_test, station):

        unique_values, counts = np.unique(df_test[self.qc_column], return_counts=True)
        for value, count in zip(unique_values, counts):
            print(f"Class {value} exists {count} times in testing dataset. This is {count/len(df_test)}.")

        X_test = df_test[[self.measurement_column, 'gradient_1', 'gradient_-1']].values
        #X_test = df_test[[self.measurement_column, 'lag_1', 'lag_2', 'lag_-1', 'lag_-2', 'gradient_1', 'gradient_-1']].values
        y_test = df_test[self.qc_column].values

        title = f'QC classes (for testing data) for {station}'
        anomalies = np.isin(df_test[self.qc_column], [3,4])
        plt.figure(figsize=(12, 6))
        plt.plot(df_test[self.measurement_column], label='Time Series', color='black', marker='o',  markersize=0.5)
        plt.scatter(df_test.index[anomalies], df_test[self.measurement_column][anomalies], color='red', label='QC Classes', zorder=1)
        plt.legend()
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Water Level')
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder_path,f"{title}.png"),  bbox_inches="tight")
        plt.close()

        # Predictions
        y_pred = self.model.predict(X_test)
        #y_pred_prob = model.predict_proba(X_test)
        #y_pred = (y_pred_prob[:, 1] > 0.5).astype(int)

        # Visualization       
        # Create confusion matix
        cm = confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix for testing data for {station}:")
        print(cm)

        # Identify anomalies
        anomalies_pred = np.isin(y_pred, [3])

        #1.Visualization
        title = f"ML predicted QC classes (for testing data)- Binary -{station}"
        plt.figure(figsize=(12, 6))
        plt.plot(df_test[self.measurement_column], label='Time Series', color='black', marker='o',  markersize=0.5)
        plt.scatter(df_test.index[anomalies_pred], df_test[self.measurement_column][anomalies_pred], color='red', label='QC Classes', zorder=1)
        plt.legend()
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Water Level')
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder_path,f"{title}-pred-test.png"),  bbox_inches="tight")
        plt.close()

        #Zoomable plot
        fig = go.Figure()
        # Add water level data
        fig.add_trace(go.Scatter(x=df_test[self.time_column], y= df_test[self.measurement_column], mode='markers', marker=dict(
            size=5,          # Marker size
            color='black',  # Marker color
            symbol='circle',   # Marker shape (e.g., 'circle', 'square', 'diamond')
            opacity=0.8       # Marker transparency
        )))
        # Add QC flags
        fig.add_trace(go.Scatter(x=df_test[self.time_column][anomalies], y= df_test[self.measurement_column][anomalies], mode='markers', marker = {'color' : 'green'}))
        # Add QC flags
        fig.add_trace(go.Scatter(x=df_test[self.time_column][anomalies_pred], y= df_test[self.measurement_column][anomalies_pred], mode='markers', marker = {'color' : 'red'}))
        # Update layout for a better view
        fig.update_layout(title=title, xaxis_title= 'Time', yaxis_title= 'Water Levels [m]', legend_title='Legend', hovermode='closest')
        # Enable zoom and pan
        fig.update_layout(xaxis=dict(rangeslider=dict(visible=True), type="date"))
        # Show the figure
        fig.show()


    def run(self, df, station_df):

        #df_train, df_test = self.split_dataset(station_df)
        df.loc[df[self.qc_column] == 4, self.qc_column] = 3
        station_df.loc[df[self.qc_column] == 4, self.qc_column] = 3
        
        # Define window size
        past_window = 5  # Use past 5 values
        future_window = 5  # Use next 5 values

        # Create feature matrix (X) and target labels (y)
        #input_df = station_df.drop(columns=["Timestamp", "label"])
        #X, y = [], []
        #for i in range(past_window, len(input_df) - future_window):
        #    past_features = input_df[i - past_window:i]  # Past 5
        #    future_features = input_df[i:i + 1 + future_window]  # current + next 5
        #    features = np.concatenate([past_features, future_features]).flatten()
        #    X.append(features)  # Combine past & future
        #    y.append(station_df[self.qc_column][i])  

        #X = np.array(X)
        #y = self.le.fit_transform(np.array(y))
        #X = station_df.drop(columns=["Timestamp", "label"]).values 
        y = station_df[self.qc_column].values 

        # Train-test split (keep sequential order)
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        df_train, df_test = train_test_split(station_df, test_size=0.2, shuffle=False)

         #Analyse existing dataset
        unique_values, counts = np.unique(station_df[self.qc_column], return_counts=True)
        for value, count in zip(unique_values, counts):
            print(f"Class {value} exists {count} times in whole dataset. This is {count/len(station_df)}.")

        unique_values, counts = np.unique(df_train[self.qc_column], return_counts=True)
        for value, count in zip(unique_values, counts):
            print(f"Class {value} exists {count} times in training dataset. This is {count/len(df_train)}.")

        unique_values, counts = np.unique(df_test[self.qc_column], return_counts=True)
        for value, count in zip(unique_values, counts):
            print(f"Class {value} exists {count} times in testing dataset. This is {count/len(df_test)}.")

        #X_train = df_train.drop(columns=["Timestamp", "label"]).values
        #X_test = df_test.drop(columns=["Timestamp", "label"]).values
        X_train = df_train[[self.measurement_column, 'gradient_1', 'gradient_-1']].values
        X_test = df_test[[self.measurement_column, 'gradient_1', 'gradient_-1']].values
        #X_train = df_train[[self.measurement_column]].values
        #X_test = df_test[[self.measurement_column]].values
        y_train = df_train[self.qc_column].values
        y_test = df_test[self.qc_column].values

        title = 'QC classes (for testing data)'
        anomalies = np.isin(df_test[self.qc_column], [3,4])
        plt.figure(figsize=(12, 6))
        plt.plot(df_test[self.measurement_column], label='Time Series', color='black', marker='o',  markersize=0.5)
        plt.scatter(df_test.index[anomalies], df_test[self.measurement_column][anomalies], color='red', label='QC Classes', zorder=1)
        plt.legend()
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Water Level')
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder_path,f"{title}.png"),  bbox_inches="tight")
        plt.close()

        title = 'QC classes (for training data)'
        anomalies = np.isin(df_train[self.qc_column], [3,4])
        plt.figure(figsize=(12, 6))
        plt.plot(df_train[self.measurement_column], label='Time Series', color='black', marker='o',  markersize=0.5)
        plt.scatter(df_train.index[anomalies], df_train[self.measurement_column][anomalies], color='red', label='QC Classes', zorder=1)
        plt.legend()
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Water Level')
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder_path,f"{title}.png"),  bbox_inches="tight")
        plt.close()

        # Train RandomForestClassifier
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        #y_pred_prob = model.predict_proba(X_test)
        #y_pred = (y_pred_prob[:, 1] > 0.5).astype(int)

        # Make predictions (train)
        y_pred_train = model.predict(X_train)
        #y_pred_train_prob = model.predict_proba(X_train)
        #y_pred_train = (y_pred_train_prob[:, 1] > 0.5).astype(int)
 
        # Visualization       
        # Create confusion matix
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix for testing data:")
        print(cm)

        # Create confusion matix
        cm = confusion_matrix(y_train, y_pred_train)
        print("Confusion Matrix for training data:")
        print(cm)

        # Identify anomalies
        anomalies_pred = np.isin(y_pred, [3,4])
        anomalies_train = np.isin(y_pred_train, [3,4])

        #1.Visualization
        title = 'ML predicted QC classes (for testing data)- Binary'
        plt.figure(figsize=(12, 6))
        plt.plot(df_test[self.measurement_column], label='Time Series', color='black', marker='o',  markersize=0.5)
        plt.scatter(df_test.index[anomalies_pred], df_test[self.measurement_column][anomalies_pred], color='red', label='QC Classes', zorder=1)
        plt.legend()
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Water Level')
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder_path,f"{title}-pred-test.png"),  bbox_inches="tight")
        plt.close()

        title = 'ML predicted QC classes (for training data)- Binary'
        plt.figure(figsize=(12, 6))
        plt.plot(df_train[self.measurement_column], label='Time Series', color='black', marker='o',  markersize=0.5)
        plt.scatter(df_train.index[anomalies_train], df_train[self.measurement_column][anomalies_train], color='red', label='QC Classes', zorder=1)
        plt.legend()
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Water Level')
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder_path,f"{title}-pred-train.png"),  bbox_inches="tight")
        plt.close()
        
        #df_train, df_test = self.split_dataset(station_df)

        #Alter imbalanced dataset in order to even out the minority class
        #self.preprocessing_data(df_train)

        # Fit and transform labels
        #self.y_train_transformed = self.le.fit_transform(self.y_train)
        #self.y_test_transformed = self.le.fit_transform(self.y_test)
        #self.y_val_transformed = self.le.fit_transform(self.y_val)
        #self.y_train_res = self.le.fit_transform(self.y_train_res)
        # Print mapping
        #print(dict(zip(self.le.classes_, self.le.transform(self.le.classes_))))

        #self.preprocessing_data_imb_learn(df_train)
        
        #self.run_binary(df_train, df_test)
        #self.run_nn_torch(df_train, df_test)
        #self.run_multi(station_df, df)

    def split_dataset(self, df):

        #Split into train and test
        df_test = df[int((self.train_size*len(df))):]
        df_train = df[:int((self.train_size*len(df)))]
        #Split training dataset again into validation and training data
        #df_val = df_train[:int((self.val_size*len(df_train)))]
        #df_train = df_train[int((self.val_size*len(df_train))):]
        #Generate correct data subsets
        self.X_train = df_train[self.measurement_column].values.reshape(-1, 1)
        self.y_train = df_train[self.qc_column]
        self.X_test = df_test[self.measurement_column].values.reshape(-1, 1)  
        self.y_test = df_test[self.qc_column]
        #self.X_val = df_val[self.measurement_column]
        #self.y_val = df_val[self.qc_column]
        self.X_train_orig = self.X_train.copy()
        self.y_train_orig = self.y_train.copy()

        #Visualization training data
        self.basic_plotter_no_xaxis(df_train, 'QC classes for Training')
        self.zoomable_plot_df(df_test, 'QC classes - Testing')
        
        #Analyse existing dataset
        unique_values, counts = np.unique(df[self.qc_column], return_counts=True)
        for value, count in zip(unique_values, counts):
            print(f"Class {value} exists {count} times in whole dataset. This is {count/len(df)}.")

        unique_values, counts = np.unique(df_train[self.qc_column], return_counts=True)
        for value, count in zip(unique_values, counts):
            print(f"Class {value} exists {count} times in training dataset. This is {count/len(df_train)}.")

        unique_values, counts = np.unique(df_test[self.qc_column], return_counts=True)
        for value, count in zip(unique_values, counts):
            print(f"Class {value} exists {count} times in testing dataset. This is {count/len(df_test)}.")

        return df_train, df_test

    def preprocessing_data(self, df):
        #Undersampling: Mark 30% of random good rows to be deleted
        rows_to_drop = df[df[self.qc_column]==1].sample(frac=0.3, random_state=42).index
        #Drop selected rows
        df_res = df.drop(rows_to_drop).reset_index(drop=True).copy()
        
        #Oversampling:Filter rows which bad qc flag and multiply them and assign them randomly 
        #bad_rows = df_res[df_res[self.qc_column].isin([3,4])].copy()
        #bad_rows[self.time_column] = bad_rows[self.time_column]  + pd.to_timedelta(np.random.randint(1, 60, size=len(bad_rows)), unit='m')
        #df_res_new = pd.concat([df_res, bad_rows]).sort_values(by=self.time_column).drop_duplicates(subset=self.time_column, keep='last').reset_index(drop=True)
        #df_res_new = pd.concat([bad_rows, df_res]).sort_values(by=self.time_column).drop_duplicates(subset=self.time_column, keep='first').reset_index(drop=True)
        
        # Oversampling:Select good rows and shift them up and down
        rows_shift_neg = df[df[self.qc_column]==1].sample(frac=0.1, random_state=42)
        rows_shift_neg[self.measurement_column] = rows_shift_neg[self.measurement_column] -random.randint(5, 30)
        df_res = pd.concat([df_res, rows_shift_neg]).sort_values(by=self.time_column).drop_duplicates(subset=self.time_column, keep='last').reset_index(drop=True) 
        rows_shift_pos = df[df[self.qc_column]==1].sample(frac=0.1, random_state=42)
        rows_shift_pos[self.measurement_column] = rows_shift_pos[self.measurement_column] + random.randint(5, 30) 
        df_res = pd.concat([df_res, rows_shift_pos]).sort_values(by=self.time_column).drop_duplicates(subset=self.time_column, keep='last').reset_index(drop=True)


        #Analysis
        unique_values, counts = np.unique(df_res[self.qc_column], return_counts=True)
        for value, count in zip(unique_values, counts):
            print(f"Class {value} exists {count} times in the resampled dataset. This is {count/len(df_res)}.")

        #Visualization resampled data
        self.basic_plotter_no_xaxis(df_res, 'Resampled QC classes')
        self.zoomable_plot_df(df_res, 'QC classes - Resampled')

        #Need balanced output
        self.X_train = df_res[self.measurement_column].values.reshape(-1, 1)  # Convert Series to 2D array
        self.y_train = df_res[self.qc_column].values.reshape(-1, 1)  # Convert Series to 2D array

    def preprocessing_data_imb_learn(self,df):

        # Define XGBoost model
        # Apply undersampling to reduce the majority class in the training data
        undersampler = RandomUnderSampler(sampling_strategy = 0.01, random_state=42)
        #self.X_train_res, self.y_train_res = undersampler.fit_resample(self.X_train, self.y_train_transformed)

        #Creating an instance of SMOTE
        smote = SMOTE(random_state=42, sampling_strategy=0.1)
        #smote = RandomOverSampler(random_state=42, sampling_strategy=0.1)
        self.X_train_res, self.y_train_res = smote.fit_resample(self.X_train, self.y_train_transformed)
        #Balancing the data
        #pipeline = Pipeline([('under', undersampler), ('over', smote)])
        #smotetomek = SMOTETomek(random_state=42, sampling_strategy=0.2)
        #self.X_train_res, self.y_train_res = smotetomek.fit_resample(self.X_train_2d, self.y_train_transformed)

        unique_values, counts = np.unique(self.y_train_res, return_counts=True)
        for value, count in zip(unique_values, counts):
            print(f"Class {value} exists {count} times in the resampled dataset. This is {count/len(self.y_train_res)}.")
        
        #0.Visualization - resampled data
        anomalies_resampled =  np.isin(self.y_train_res,[3, 4])
        title = 'ML predicted QC classes - Resampled training'
        plt.figure(figsize=(12, 6))
        plt.plot(self.X_train_res, label='Time Series', color='black', marker='o',  markersize=0.5, linestyle='None')
        #plt.scatter(np.arange(len(y_train_res))[anomalies_resampled], y_train_res[anomalies_resampled], color='red', label='QC Classes', zorder=1)
        plt.legend()
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Water Level [m]')
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder_path,f"{title}.png"),  bbox_inches="tight")
        plt.close()
        plt.close()

        #0.Visualization - resampled data - zoomed
        anomalies_resampled =  np.isin(self.y_train_res,[3, 4])
        title = 'ML predicted QC classes - Resampled training - Zoom'
        plt.figure(figsize=(12, 6))
        plt.plot(self.X_train_res[14000:], label='Time Series', color='black', marker='o',  markersize=0.5, linestyle='None')
        #plt.scatter(np.arange(len(y_train_res))[anomalies_resampled], y_train_res[anomalies_resampled], color='red', label='QC Classes', zorder=1)
        plt.legend()
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Water Level [m]')
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder_path,f"{title}.png"),  bbox_inches="tight")
        plt.close()

    def run_binary(self, df_train, df_test):

        #Get best XGBClassifier model based on hyperparameters
        param_grid = {
            'max_depth': [3, 5, 7],
            'min_child_weight': [0, 0.1, 1],
            'learning_rate': [0.001, 0.005, 0.01, 0.05],
            'n_estimators': [50, 100, 200],
            'gamma': [0, 0.001]
        }

        #Generate the model
        grid_search = GridSearchCV(XGBClassifier(), param_grid, cv=3, verbose=1)
        grid_search.fit(self.X_train_res, self.y_train_res)
        grid_search.fit(self.X_train_2d, self.y_train_transformed)
        best_params = grid_search.best_params_ 
        print(f"Best parameters: {best_params}")

        scale_pos_weight = sum(self.y_train_transformed == 0) / sum(self.y_train_transformed == 1)
        #scale_pos_weight = sum(y_train_res == 0) / sum(y_train_res == 1)
        #model = XGBClassifier(objective="binary:logistic", scale_pos_weight=scale_pos_weight, eval_metric="logloss", max_depth=4, learning_rate=0.01, n_estimators=200, random_state=42)
        #model = XGBClassifier(objective="binary:logistic", eval_metric="logloss", max_depth=4, learning_rate=0.01, n_estimators=200, random_state=42)
        model = XGBClassifier(objective="binary:logistic", eval_metric="aucpr", random_state=42, **best_params)
        #model = RandomForestClassifier(n_estimators=300, random_state=42)
        #model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
        #Adaboost
        #sample_weights = compute_sample_weight(class_weight='balanced', y=self.y_train_transformed)
        #base_estimator = DecisionTreeClassifier(max_depth=1)
        #model = AdaBoostClassifier(estimator=base_estimator)
        #model = BalancedRandomForestClassifier(n_estimators=100, random_state=42)


        # Train the model
        #model.fit(self.X_train_res, self.y_train_res, eval_set=[(self.X_train_res, self.y_train_res), (self.X_val, self.y_val_transformed)], verbose=True)
        #model.fit(self.X_train_res, self.y_train_res) #, sample_weight=sample_weights)
        #model.fit(self.X_train, self.y_train_transformed, eval_set=[(X_train, y_train_transformed), (X_val, y_val_transformed)], verbose=True)
        #model.fit(self.X_train, self.y_train_transformed) #, sample_weight=sample_weights)
        model.fit(self.X_train_window, self.y_train_transformed)
        # Make predictions (test)
        # # Get predicted probabilities
        y_pred_prob = model.predict_proba(self.X_test)[:, 1]

        # Set a custom threshold (e.g., 0.3 instead of 0.5)
        #y_pred = (y_prob > 0.3).astype(int) 
        #y_pred_prob = model.predict(self.X_test_2d)
        y_pred_transformed = (y_pred_prob > 0.5).astype(int)
        y_pred = self.le.inverse_transform(y_pred_transformed)
        
        # Make predictions (train)
        y_pred_train_prob = model.predict(self.X_train_orig)
        y_pred_train_transformed = (y_pred_train_prob > 0.5).astype(int)
        y_pred_train = self.le.inverse_transform(y_pred_train_transformed)

        # Visualization
        #Print-statement
        unique_values, counts = np.unique(y_pred, return_counts=True)
        for value, count in zip(unique_values, counts):
            print(f"Class {value} was predicted {count} times.")
        
        # Create confusion matix
        cm = confusion_matrix(self.y_test, y_pred)
        print("Confusion Matrix for testing data:")
        print(cm)

        # Create confusion matix
        cm = confusion_matrix(self.y_train_orig, y_pred_train)
        print("Confusion Matrix for training data:")
        print(cm)

        # Identify anomalies
        anomalies_pred = np.isin(y_pred, [3, 4])
        anomalies_train = np.isin(y_pred_train, [3, 4])

        #1.Visualization
        self.basic_plotter_no_xaxis(df_test, 'QC classes for Testing data')

        #2.Visualization
        title = 'ML predicted QC classes (for testing data)- Binary'
        plt.figure(figsize=(12, 6))
        plt.plot(df_test[self.measurement_column], label='Time Series', color='black', marker='o',  markersize=0.5)
        plt.scatter(df_test.index[anomalies_pred], df_test[self.measurement_column][anomalies_pred], color='red', label='QC Classes', zorder=1)
        plt.legend()
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Water Level')
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder_path,f"{title}-pred-test.png"),  bbox_inches="tight")
        plt.close()

        #Zoomable plot
        fig = go.Figure()
        # Add water level data
        fig.add_trace(go.Scatter(x=df_test[self.time_column], y= df_test[self.measurement_column], mode='markers', marker=dict(
            size=5,          # Marker size
            color='rgb(255, 0, 0)',  # Marker color
            symbol='circle',   # Marker shape (e.g., 'circle', 'square', 'diamond')
            opacity=0.8       # Marker transparency
        )))
        # Add QC flags
        fig.add_trace(go.Scatter(x=df_test[self.time_column][anomalies_pred], y= df_test[self.measurement_column][anomalies_pred], mode='markers', marker = {'color' : 'black'}))
        # Update layout for a better view
        fig.update_layout(title=title, xaxis_title= 'Time', yaxis_title= 'Water Levels [m]', legend_title='Legend', hovermode='closest')
        # Enable zoom and pan
        fig.update_layout(xaxis=dict(rangeslider=dict(visible=True), type="date"))
        # Show the figure
        fig.show()

        title = 'ML predicted QC classes (for training data)- Binary'
        plt.figure(figsize=(12, 6))
        plt.plot(df_train[self.measurement_column], label='Time Series', color='black', marker='o',  markersize=0.5)
        plt.scatter(df_train.index[anomalies_train], df_train[self.measurement_column][anomalies_train], color='red', label='QC Classes', zorder=1)
        plt.legend()
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Water Level')
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder_path,f"{title}-pred-training.png"),  bbox_inches="tight")
        plt.close()

    
    def run_nn_torch(self, df_train, df_test):

        sequences = []
        sequence_labels = []
        window_size = 29

        for i in range(0, len(self.X_train) - window_size + 1, 1):
            sequence = self.X_train[i:i+window_size]
            # Get the label for this sequence (e.g., you could take the label of the last time step)
            sequence_label = 1 if 1 in self.y_train_transformed[i : i + window_size] else 0  # Label entire window
            sequences.append(sequence)
            sequence_labels.append(sequence_label)

        sequences = torch.tensor(sequences, dtype=torch.float32)
        sequence_labels = torch.tensor(sequence_labels, dtype=torch.float32)
        
        train_dataset = TensorDataset(sequences, sequence_labels)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

        #Convert the data to PyTorch tensors
        #X_train_tensor = torch.tensor(self.X_train, dtype=torch.float32)
        #y_train_tensor = torch.tensor(self.y_train_transformed, dtype=torch.float32).view(-1, 1)  # Reshaping to 2D for binary classification

        X_test_tensor = torch.tensor(self.X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(self.y_test_transformed, dtype=torch.float32).view(-1, 1)

        # 6. Create DataLoader for batching
        #train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        #train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=7500, shuffle=False)

        # Loop through batches
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader): 
            if batch_idx == 30:
                break
            else:     
                # Create a figure for the batch
                plt.figure(figsize=(10, 6))
                
                # Loop through sequences within the batch
                for seq_idx in range(X_batch.shape[0]):  # Assuming X_batch is (batch_size, sequence_length)
                    plt.plot(X_batch[seq_idx].numpy(), marker='o', markersize=2, linestyle='-', label=f"X_seq {seq_idx}")
                    plt.plot(y_batch[seq_idx].numpy(), marker='x', markersize=2, linestyle='-', label=f"y_seq {seq_idx}")

                # Title and labels
                plt.title(f"Batch {batch_idx + 1} - Sequences Visualization")
                plt.xlabel("Timesteps")
                plt.ylabel("Value")
                plt.legend(loc="upper right", fontsize="small", ncol=2)  # Adjust legend for readability
                plt.grid(True, linestyle="--", alpha=0.6)

                # Save the figure
                save_path = os.path.join(self.folder_path, f"Batch_{batch_idx + 1}.png")
                plt.savefig(save_path, bbox_inches="tight")
                plt.close()  # Close figure to prevent memory issues
                
        # 2. Define the model
        class TimeSeriesModel(nn.Module):
            def __init__(self, input_size, window_size):
                super().__init__()
                
                # A simple feed-forward neural network
                self.fc1 = nn.Linear(input_size * window_size, 128)  # First fully connected layer
                self.fc2 = nn.Linear(128, 64)  # Second fully connected layer
                self.fc3 = nn.Linear(64, 1)  # Output layer (1 unit for binary classification)
                self.sigmoid = nn.Sigmoid()  # Sigmoid activation for binary classification

            def forward(self, x):
                x = x.view(x.shape[0], -1) 

                x = torch.relu(self.fc1(x))  # Apply ReLU activation
                x = torch.relu(self.fc2(x))  # Apply ReLU activation
                x = self.fc3(x)  # Output layer
                x = self.sigmoid(x)  # Sigmoid to get probabilities (0 or 1)
                return x
            
        class SpikeDetectionLSTM(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers):
                super(SpikeDetectionLSTM, self).__init__()
                
                # LSTM Layer(s)
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                
                # Fully connected layer for classification
                self.fc = nn.Linear(hidden_size, 1)  # Output layer (1 output for spike detection)
                self.sigmoid = nn.Sigmoid()  # Sigmoid to classify as 0 or 1 (good or bad)
            
            def forward(self, x):
                # Pass through LSTM
                lstm_out, _ = self.lstm(x)  # lstm_out is the output of the LSTM for each time step
                
                # Use the last LSTM output for classification (can also try using all outputs)
                x = self.fc(lstm_out[:, -1, :])  # Output of last time step (if it's one-step classification)
                x = self.sigmoid(x)  # Sigmoid activation for binary classification (spike vs non-spike)
                
                return x
            
        #Example for time series with 1 feature
        input_size = 1 # Total input size to the network

        #Instantiate the model
        #model = TimeSeriesModel(input_size, window_size).to('cpu')

        #Model configuration
        input_size = 1  # Single feature per timestep
        hidden_size = 256  # Maximum number of neurons per layer
        num_layers = 3  # Number of LSTM layers

        # Instantiate the model
        model = SpikeDetectionLSTM(input_size, hidden_size, num_layers).to('cpu')

        class FocalLoss(torch.nn.Module):
            def __init__(self, alpha=0.25, gamma=2):
                super(FocalLoss, self).__init__()
                self.alpha = alpha
                self.gamma = gamma

            def forward(self, inputs, targets):
                bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
                pt = torch.exp(-bce_loss)
                focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
                return focal_loss.mean()
        
        #criterion = nn.BCELoss()
        criterion = FocalLoss(alpha=0.75, gamma=3)  # Example: balanced alpha and high gamma for focus on hard samples
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        # 7. Training loop
        num_epochs = 3
        for epoch in range(num_epochs):
            model.train()  # Set the model to training mode
            for X_batch, y_batch in train_loader:
                # Zero the gradients
                optimizer.zero_grad()
                # Forward pass
                outputs = model(X_batch)
                # Compute the loss
                #loss = criterion(outputs.squeeze(), y_batch)  # Squeeze the output to match the labels
                loss = criterion(outputs, y_batch.view(-1, 1))  # Squeeze the output to match the labels
                # Backward pass
                loss.backward()
                optimizer.step()

            # Print the loss for this epoch
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss}')

        # 8. Model evaluation (after training)
        model.eval()  # Set the model to evaluation mode
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to('cpu'), y_batch.to('cpu')
                outputs = model(X_batch)
                
                # Convert probabilities to binary predictions (0 or 1)
                predicted = (outputs > 0.2).float()

                # Collect the true labels and predictions
                all_labels.extend(y_batch.numpy())
                all_predictions.extend(predicted.numpy())

        # Convert to numpy arrays for easier manipulation
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        anomalies_pred = all_predictions.astype(bool).flatten()  

        precision = precision_score(all_labels, all_predictions, average='binary')
        recall = recall_score(all_labels, all_predictions, average='binary')
        f1 = f1_score(all_labels, all_predictions, average='binary')
        auc = roc_auc_score(all_labels, all_predictions)
        
        print(f'Precision: {precision}, Recall: {recall}, F1 Score: {f1}, AUC-ROC: {auc}')

        # Compute confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)

        # Print the confusion matrix
        print("Confusion Matrix:")
        print(cm)

        # 13. Print evaluation metrics
        title = 'NN - ML predicted QC classes (Pred)'
        plt.figure(figsize=(12, 6))
        plt.plot(df_test[self.measurement_column], label='Time Series', color='black', marker='o',  markersize=0.5)
        plt.plot(df_test[self.measurement_column][anomalies_pred], color='red', label='QC Classes', zorder=1)
        plt.legend()
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Water Level')
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder_path,f"{title}.png"),  bbox_inches="tight")
        plt.close()
