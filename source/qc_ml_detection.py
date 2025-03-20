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

    def set_station(self, station):
        self.station = station
        self.information.append(['The following text summarizes the supervised ML QC perfromed on measurements from ', station,':'])
    
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

        self.dfs = {}
        #Open .csv file and fix the column names
        for file in os.listdir(folder_path):  # Loops through files in the current directory
            print(file)
            df = pd.read_csv(os.path.join(folder_path,file), sep=",", header=0)
            df = df[[self.time_column, self.measurement_column, self.qc_column]]
            df[self.measurement_column] = df[self.measurement_column] - df[self.measurement_column].mean()
            df[self.qc_column] = df[self.qc_column].fillna(1).astype(int)
            #FOR NOW: Only a binary issue of spikes/non-spikes
            df[self.qc_column] = df[self.qc_column].replace(4, 3)
            df[self.time_column] = pd.to_datetime(df[self.time_column]).dt.round('min')
            self.dfs[f"{file.split("-")[0]}"] = df.copy()
            #self.dfs[f"df_{file.split("-")[0]}"] = df.copy()
            anomaly = np.isin(df[self.qc_column], [3,4])
            self.basic_plotter_no_xaxis(df, f'Manual QC for {file} station',anomaly)
            if (df[self.qc_column] == 4).any():
                raise Exception('Code is build for binary classification. QC flags can currently only be 1 and 3!')
    
    def run(self):
        
        print(self.dfs.keys())
        for elem in self.dfs:
            if self.station in elem:
                #duplicate df of station for training data and analyse it
                df_train = self.dfs[elem]
                unique_values, counts = np.unique(df_train[self.qc_column], return_counts=True)
                for value, count in zip(unique_values, counts):
                    print(f"Class {value} exists {count} times in training dataset. This is {count/len(df_train)}.")
                #Preprocess the training data if needed
                #df_train = self.preprocessing_data(df_train)
                #Decied if whole station df is used for training or only subset. If subset, split here
                #df_train, df_test = train_test_split(self.dfs[elem], test_size=0.2, shuffle=False)
                #Add features to the original data series and the training data
                tidal_signal = [path for path in self.tidal_infos if elem.strip("0123456789") in path]
                df_train = self.add_features(df_train, tidal_signal[0], elem)
                self.dfs[elem] = self.add_features(self.dfs[elem], tidal_signal[0], elem)
            else:
                tidal_signal= [path for path in self.tidal_infos if elem.strip("0123456789") in path]
                self.dfs[elem] = self.add_features(self.dfs[elem], tidal_signal[0], elem)

        X_train = df_train.drop(columns=["Timestamp", "label"]).values
        #X_train = df_train[[self.measurement_column, 'lag_1', 'lag_2', 'lag_-1', 'lag_-2', 'gradient_1', 'gradient_-1']].values
        #X_train = df_train[[self.measurement_column, 'gradient_1', 'gradient_-1']].values
        y_train = df_train[self.qc_column].values

        #Visualize training data
        anomalies = np.isin(df_train[self.qc_column], [3])
        self.basic_plotter_no_xaxis(df_train, 'QC classes (for training data)', anomalies)

        #Fit ML model to data
        self.run_model(X_train, y_train)

        #Runs all dataset again for testing (also training set)
        for elem in self.dfs:
            self.run_testing_model(self.dfs[elem], elem)

    def preprocessing_data(self, df):
        #Undersampling: Mark 10% of random good rows to be deleted
        rows_to_drop = df[df[self.qc_column]==1].sample(frac=0.1, random_state=42).index
        #Drop selected rows
        #df = df.drop(rows_to_drop).reset_index(drop=True).copy()
        
        #Oversampling:Filter rows which bad qc flag and multiply them and assign them randomly 
        #bad_rows = df_res[df_res[self.qc_column].isin([3,4])].copy()
        #bad_rows[self.time_column] = bad_rows[self.time_column]  + pd.to_timedelta(np.random.randint(1, 60, size=len(bad_rows)), unit='m')
        #df_res_new = pd.concat([df_res, bad_rows]).sort_values(by=self.time_column).drop_duplicates(subset=self.time_column, keep='last').reset_index(drop=True)
        #df_res_new = pd.concat([bad_rows, df_res]).sort_values(by=self.time_column).drop_duplicates(subset=self.time_column, keep='first').reset_index(drop=True)
        
        # Oversampling:Select good rows and shift them up and down
        rows_shift_neg = df[df[self.qc_column]==1].sample(frac=0.1, random_state=42)
        shift_array = np.random.uniform(0.02, 0.12, len(rows_shift_neg))
        rows_shift_neg[self.measurement_column] = rows_shift_neg[self.measurement_column] - shift_array
        rows_shift_neg[self.qc_column] = 3
        df = pd.concat([df, rows_shift_neg]).sort_values(by=self.time_column).drop_duplicates(subset=self.time_column, keep='last').reset_index(drop=True) 
        rows_shift_pos = df[df[self.qc_column]==1].sample(frac=0.1, random_state=42)
        shift_array = np.random.uniform(0.05, 0.12, len(rows_shift_pos))
        rows_shift_pos[self.measurement_column] = rows_shift_pos[self.measurement_column] + shift_array
        rows_shift_pos[self.qc_column] = 3
        df = pd.concat([df, rows_shift_pos]).sort_values(by=self.time_column).drop_duplicates(subset=self.time_column, keep='last').reset_index(drop=True)

        #Visualization resampled data
        anomalies = np.isin(df[self.qc_column], [3])
        self.basic_plotter_no_xaxis(df, 'Resampled QC classes (small distribution)', anomalies)
        self.zoomable_plot_df(df, 'QC classes - Resampled', anomalies)

        #Oversampling:Add some extreme values
        rows_shift_neg = df[df[self.qc_column]==1].sample(frac=0.01, random_state=42)
        shift_array = np.random.uniform(0.1, 1.5, len(rows_shift_neg))
        rows_shift_neg[self.measurement_column] = rows_shift_neg[self.measurement_column] - shift_array
        rows_shift_neg[self.qc_column] = 3
        df = pd.concat([df, rows_shift_neg]).sort_values(by=self.time_column).drop_duplicates(subset=self.time_column, keep='last').reset_index(drop=True) 
        rows_shift_pos = df[df[self.qc_column]==1].sample(frac=0.01, random_state=42)
        shift_array = np.random.uniform(0.1, 1.5, len(rows_shift_pos))
        rows_shift_pos[self.measurement_column] = rows_shift_pos[self.measurement_column] + shift_array
        rows_shift_pos[self.qc_column] = 3
        df = pd.concat([df, rows_shift_pos]).sort_values(by=self.time_column).drop_duplicates(subset=self.time_column, keep='last').reset_index(drop=True)

        #Analysis
        unique_values, counts = np.unique(df[self.qc_column], return_counts=True)
        for value, count in zip(unique_values, counts):
            print(f"Class {value} exists {count} times in the resampled dataset. This is {count/len(df)}.")

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
        tidal_signal_series = self.reinitialize_utide_from_txt(df, tidal_signal, station)
        df['tidal_signal'] = tidal_signal_series
        self.features = ['lag_1', 'lag_2', 'lag_-1', 'lag_-2', 'tidal_signal', 'wavelet', 'gradient_1', 'gradient_-1']

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
    
    def run_model(self, X_train, y_train):

        #Get best XGBClassifier model based on hyperparameters
        param_grid = {
            'max_depth': [3, 5, 7],
            'min_child_weight': [0, 0.1, 1],
            'learning_rate': [0.001, 0.005, 0.01, 0.05],
            'n_estimators': [50, 100, 200],
            'gamma': [0, 0.001]
        }

        #Generate the model
        #grid_search = GridSearchCV(XGBClassifier(), param_grid, cv=3, verbose=1)
        #grid_search.fit(X_train, y_train)
        #best_params = grid_search.best_params_ 
        #print(f"Best parameters: {best_params}")

        #scale_pos_weight = sum(y_train == 0) / sum(y_train== 1)
        #model = XGBClassifier(objective="binary:logistic", scale_pos_weight=scale_pos_weight, eval_metric="logloss", max_depth=4, learning_rate=0.01, n_estimators=200, random_state=42)
        #model = XGBClassifier(objective="binary:logistic", eval_metric="logloss", max_depth=4, learning_rate=0.01, n_estimators=200, random_state=42)
        #model = XGBClassifier(objective="binary:logistic", eval_metric="aucpr", random_state=42, **best_params)
        self.model = RandomForestClassifier(n_estimators=200, random_state=42)
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
        self.model.fit(X_train, y_train)

    def run_testing_model(self, df_test, station):
        #Print details on how testing dataset is expected to look like  
        unique_values, counts = np.unique(df_test[self.qc_column], return_counts=True)
        for value, count in zip(unique_values, counts):
            print(f"Class {value} exists {count} times in testing dataset. This is {count/len(df_test)}.")

        title = f'QC classes (for testing data) for {station}'
        anomalies = np.isin(df_test[self.qc_column], [3])
        self.basic_plotter_no_xaxis(df_test, title, anomalies)

        #Extract test data
        #X_test = df_test[[self.measurement_column, 'gradient_1', 'gradient_-1']].values
        #X_test = df_test[[self.measurement_column, 'lag_1', 'lag_2', 'lag_-1', 'lag_-2', 'gradient_1', 'gradient_-1']].values
        X_test = df_test.drop(columns=["Timestamp", "label"]).values
        y_test = df_test[self.qc_column].values

        # Predictions
        y_pred = self.model.predict(X_test)

        # Visualization predictions vs testing label   
        # Create confusion matix
        cm = confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix for testing data for {station}:")
        print(cm)
        # Identify anomalies: Visualization
        title = f"ML predicted QC classes (for testing data)- Binary -{station}"
        anomalies_pred = np.isin(y_pred, [3])
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


    def run_unsupervised(self, df):
        #one high-high or low-low cycle in tide for slicing ts for unsupervised ML test
        self.window_size = 30
                
        # Prepare data for Isolation Forest including reelv features
        X = []
        for i in range(len(df)):
            content = padded_values[i:i + self.window_size]
            # Extract features (mean, std, min, max)
            #mean = np.array([np.mean(content)])
            #std = np.array([np.std(content)])
            #min_val = np.array([np.min(content)])
            #max_val = np.array([np.max(content)])
            #features =  np.append(content, [mean, std, min_val, max_val])
            X.append(content)
        X = np.array(X)

        # Fit Isolation Forest
        model = IsolationForest(contamination=0.05, n_estimators=200, random_state=42)  # Adjust contamination as needed
        anomaly = model.fit_predict(X)

        #Identify anomalies
        anomalies = relev_df[anomaly == -1]

        #Drop non-nan indices
        relev_index = anomalies.index
        non_nan_indices = relev_df[~relev_df[data_column_name].isna()].index.to_numpy()
        common_indices = relev_index.intersection(non_nan_indices)
        df.loc[common_indices, f'unsupervised_ml_outliers{suffix}'] = True

        # Visualization
        title = 'Anomaly Detection with Isolation Forest (Using Interpolation)'
        plt.figure(figsize=(12, 6))
        plt.plot(relev_df[time_column], relev_df[data_column_name], label='Time Series', color='blue', marker='o',  markersize=1)
        plt.scatter(anomalies[time_column], anomalies[data_column_name], color='red', label='Anomalies', zorder=1)
        plt.legend()
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Water Level')
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder_path,f"{title}- Date: {relev_df[time_column].iloc[0]} -{suffix}.png"),  bbox_inches="tight")
        plt.close()

