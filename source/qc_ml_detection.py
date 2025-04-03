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
from datetime import datetime

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

plt.rcParams["font.size"] = 13  # Set global font size (adjust as needed)
plt.rcParams["axes.labelsize"] = 13  # Set x and y label size
plt.rcParams["axes.titlesize"] = 13  # Set title size
plt.rcParams["xtick.labelsize"] = 13  # Set x-axis tick labels size
plt.rcParams["ytick.labelsize"] = 13  # Set y-axis tick labels size
plt.rcParams["legend.fontsize"] = 13  # Set x-axis tick labels size
plt.rcParams["figure.titlesize"] = 13  # Set y-axis tick labels size

BIGGER_SIZE = 13

class MLOutlierDetection(): 

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
        self.dfs_station_subsets = {}
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
            if file.split("-")[0] in self.dfs_station_subsets.keys(): 
                merged_df = pd.concat([self.dfs_station_subsets[f"{file.split("-")[0]}"], df])
                merged_df = merged_df.sort_values(by='Timestamp').reset_index(drop=True)
                self.dfs_station_subsets[f"{file.split("-")[0]}"] = merged_df.copy()
            else:
                self.dfs_station_subsets[f"{file.split("-")[0]}"] = df.copy()
            if (df[self.qc_column] == 4).any():
                raise Exception('Code is build for binary classification. QC flags can currently only be 1 and 3!')

        return self.dfs_station_subsets, self.dfs_training

    def run(self, df_dict):

        print(df_dict.keys())
        print(df_dict.keys())
        
        for elem in df_dict:
            if self.station in elem:
                ##Use subset of one station for training and test on other stations and leftover data from this station. 
                df_train, df_test = train_test_split(df_dict[elem], test_size=0.15, shuffle=False)
                unique_values, counts = np.unique(df_train[self.qc_column], return_counts=True)
                for value, count in zip(unique_values, counts):
                    self.information.append(f"Class {value} exists {count} times in training dataset. This is {count/len(df_train)}.")
                    print(f"Class {value} exists {count} times in training dataset. This is {count/len(df_train)}.")
                    self.information.append('\n')
                #Preprocess the training data if needed
                df_train = self.preprocessing_data(df_train)
                #Add features to the original data series and the training data
                tidal_signal = [path for path in self.tidal_infos if elem.strip("0123456789") in path]
                df_train = self.add_features(df_train, tidal_signal[0], elem)

        #Visualize training data
        anomalies = np.isin(df_train[self.qc_column], [3])
        self.basic_plotter_no_xaxis(df_train, 'QC classes (for training data)', anomalies)

        X_train = df_train[self.features].values
        y_train = self.le.fit_transform(np.array(df_train[self.qc_column].values))
        print(dict(zip(self.le.classes_, self.le.transform(self.le.classes_))))

        #Fit ML model to data
        self.run_model(X_train, y_train)

        return df_dict, df_test

    def run_combined_training(self, df_dict):
        
        print(df_dict.keys())
        print(df_dict.keys())
        dfs_testing_new = {}
        dfs_train = {}
        for elem in df_dict:
            #Decied if whole station df is used for training or only subset. If subset, split here
            df_train, df_test = train_test_split(df_dict[elem], test_size=0.15, shuffle=False)
            anomalies = np.isin(df_train[self.qc_column], [3])
            self.basic_plotter_no_xaxis(df_train, f'QC classes (for training data) - Subset {elem}', anomalies)
            #Preprocess the training data if needed
            df_train = self.preprocessing_data(df_train)
            #Add features to the original data series and the training data
            tidal_signal = [path for path in self.tidal_infos if elem in path]
            df_train = self.add_features(df_train, tidal_signal[0], elem)
            dfs_train[f"{elem}"] = df_train
            dfs_testing_new[f"{elem}"] = df_test

        # Merge all DataFrames in the dictionary for training_df
        merged_df = pd.concat(dfs_train.values())
        df_train = merged_df.sort_values(by='Timestamp').reset_index(drop=True)
        unique_values, counts = np.unique(df_train[self.qc_column], return_counts=True)
        for value, count in zip(unique_values, counts):
            self.information.append(f"Class {value} exists {count} times in training dataset. This is {count/len(df_train)}.")
            print(f"Class {value} exists {count} times in training dataset. This is {count/len(df_train)}.")
            self.information.append('\n')

        #Visualize training data
        anomalies = np.isin(df_train[self.qc_column], [3])
        self.basic_plotter_no_xaxis(df_train, 'QC classes (for training data)', anomalies)

        X_train = df_train[self.features].values
        y_train = self.le.fit_transform(np.array(df_train[self.qc_column].values))
        print(dict(zip(self.le.classes_, self.le.transform(self.le.classes_))))

        #Fit ML model to data
        self.run_model(X_train, y_train)

        return dfs_testing_new

    def run_testing(self, df_dict, df_test=pd.DataFrame()):

        #Runs all dataset again for testing (also training set)
        outcomes = {}
        for elem in df_dict:
            if not df_test.empty:
                if self.station in elem:
                    df_outcome = self.run_testing_model(df_test, elem)
                else:
                    df_outcome = self.run_testing_model(df_dict[elem], elem)
            else:
                df_outcome = self.run_testing_model(df_dict[elem], elem)
            outcomes[elem]= df_outcome

        self.save_to_txt()
        self.save_to_csv()

        return outcomes

    def preprocessing_data(self, df):
        #Undersampling: Mark 10% of random good rows to be deleted
        #rows_to_drop = df[df[self.qc_column]==1].sample(frac=0.1, random_state=42).index
        #Drop selected rows
        #df = df.drop(rows_to_drop).reset_index(drop=True).copy()
        
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
        #self.zoomable_plot_df(df, 'QC classes - Resampled', anomalies)

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
        #self.zoomable_plot_df(df, 'QC classes - Resampled', anomalies)

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
        df['gradient_3'] = df['value'] - df['value'].shift(3)
        df['gradient_-3'] = df['value'].shift(-3) - df['value']

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
        rolling_corr = df['value'].rolling(window=10).corr(df['tidal_signal'])
        df['tidal_corr'] = rolling_corr

        #Define relevant features
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
    
    def run_model(self, X_train, y_train):

        #Get best XGBClassifier model based on hyperparameters
        param_dist = {
            'n_estimators': np.arange(100, 1001, 100),
            'max_depth': np.arange(3, 11, 2),
            'learning_rate': np.linspace(0.01, 0.3, 10),
            'min_child_weight': np.arange(1, 11, 2),
            'gamma': np.linspace(0, 5, 10),
            'subsample': np.linspace(0.5, 1.0, 10),
            'colsample_bytree': np.linspace(0.3, 1.0, 10),
            'scale_pos_weight': [1, 10, 25, 50, 75, 100]
        }

        # Initialize the model
        xgb = XGBClassifier(objective='binary:logistic', random_state=42, use_label_encoder=False, eval_metric='aucpr')

        # Initialize RandomizedSearchCV
        #random_search = RandomizedSearchCV(
        #    estimator=xgb,
        #    param_distributions=param_dist,
        #    n_iter=20,  # Number of different combinations to try
        #    cv=5,
        #    scoring='f1',  # Use F1-score for imbalanced data
        #    random_state=42,
        #    n_jobs=-1
        #)

        #scale_pos_weight = sum(y_train == 0) / sum(y_train== 1)
        #model = XGBClassifier(objective="binary:logistic", scale_pos_weight=scale_pos_weight, eval_metric="logloss", max_depth=4, learning_rate=0.01, n_estimators=200, random_state=42)
        #self.model = XGBClassifier(objective="binary:logistic", eval_metric="aucpr", learning_rate=0.01, n_estimators=200, random_state=42)
        #model = XGBClassifier(objective="binary:logistic", eval_metric="aucpr", random_state=42, **best_params)

        #param_dist = {
        #    'n_estimators': np.arange(10, 501, 100),
        #    'max_depth': np.arange(3, 21, 3),
        #    'min_samples_split': np.arange(2, 21, 2),
        #    'min_samples_leaf': np.arange(1, 11, 1),
        #    'max_features': ['sqrt', 'log2', None],
        #    'bootstrap': [True, False],
        #    'class_weight': [None, 'balanced']
        #}

        # Initialize the model
        #rf = RandomForestClassifier(random_state=42)

        # Initialize RandomizedSearchCV
        #random_search = RandomizedSearchCV(
        #    estimator=rf,
        #    param_distributions=param_dist,
        #    n_iter=20,  # Number of different combinations to try
        #    cv=5,       # 5-fold cross-validation
        #    scoring='f1',
        #    random_state=42
        #)

        #self.model = RandomForestClassifier(n_estimators=50, n_jobs=1, random_state=42)
        self.model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
        self.model.fit(X_train, y_train)
        
        #Adaboost: Define the hyperparameter grid
        #param_dist = {
        #    'n_estimators': np.arange(50, 1001, 50),
        #    'learning_rate': np.linspace(0.01, 1.0, 10),
        #    'estimator': [DecisionTreeClassifier(max_depth=d) for d in range(1, 6)],
        #    'algorithm': ['SAMME', 'SAMME.R']
        #}

        #Initialize the model
        #adaboost = AdaBoostClassifier(random_state=42)

        # Initialize RandomizedSearchCV
        #random_search = RandomizedSearchCV(
        #    estimator=adaboost,
        #    param_distributions=param_dist,
        #    n_iter=20,
        #    cv=5,
        #    scoring='f1',
        #    random_state=42,
        #    n_jobs=-1
        #)

        # Fit the model
        #random_search.fit(X_train, y_train)

        # Print the best parameters
        #print("Best Parameters:", random_search.best_params_)

        # Evaluate the best model
        #self.model = random_search.best_estimator_

        # Train the model
        #model.fit(self.X_train_res, self.y_train_res, eval_set=[(self.X_train_res, self.y_train_res), (self.X_val, self.y_val_transformed)], verbose=True)
        #model.fit(self.X_train, self.y_train_transformed, eval_set=[(X_train, y_train_transformed), (X_val, y_val_transformed)], verbose=True)

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
        #y_pred = self.model.predict(X_test)
        y_probs = self.model.predict_proba(X_test)[:, 1] 
        # Find the best threshold for F1-score
        thresholds = np.linspace(0.2, 0.8, 20)
        y_test_binary = (y_test == 1).astype(int)
        f1_scores = [f1_score(y_test_binary, (y_probs >= t).astype(int)) for t in thresholds]
        best_threshold = thresholds[np.argmax(f1_scores)]
        print(f"Best Threshold for F1-score: {best_threshold:.2f}")
        y_pred = (y_probs >= best_threshold).astype(int)
        y_test = y_test_binary

        # Visualization predictions vs testing label   
        # Create confusion matix
        cm = confusion_matrix(y_test, y_pred)
        #precision = precision_score(y_test, y_pred, pos_label=3)
        #recall = recall_score(y_test, y_pred, pos_label=3)
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
        self.basic_plotter_no_xaxis(df_test, title, anomalies, anomalies_pred)
        self.basic_plotter(df_test, title, anomalies, anomalies_pred)
        #self.zoomable_plot_df(df_test, title, anomalies, anomalies_pred)

        #Plot specific times
        df_test['Timestamp_helper'] = df_test['Timestamp'].dt.tz_convert(None)
        if station == 'Nuuk':
            start_date = datetime(int(2023), int(1), int(24), int(21))
            end_date = datetime(int(2023), int(1), int(25), int(18))
            start_date1 = datetime(int(2023), int(1), int(25), int(0))
            end_date1 = datetime(int(2023), int(1), int(25), int(1))
            relev_df = df_test[(df_test['Timestamp_helper'] >= start_date) & (df_test['Timestamp_helper']<= end_date)]
            relev_df1 = df_test[(df_test['Timestamp_helper'] >= start_date1) & (df_test['Timestamp_helper']<= end_date1)]
        elif station == 'Upernavik':
            start_date = datetime(int(2024), int(10), int(20), int(19))
            end_date = datetime(int(2024), int(10), int(21), int(11))
            start_date1 = datetime(int(2024), int(10), int(26), int(22))
            end_date1 = datetime(int(2024), int(10), int(27), int(13))
            relev_df = df_test[(df_test['Timestamp_helper'] >= start_date) & (df_test['Timestamp_helper']<= end_date)]
            relev_df1 = df_test[(df_test['Timestamp_helper'] >= start_date1) & (df_test['Timestamp_helper']<= end_date1)]
        elif station == 'Pituffik':
            start_date = datetime(int(2017), int(12), int(14), int(10))
            end_date = datetime(int(2017), int(12), int(15), int(0))
            start_date1 = datetime(int(2017), int(12), int(14), int(16))
            end_date1 = datetime(int(2017), int(12), int(14), int(19))
            relev_df = df_test[(df_test['Timestamp_helper'] >= start_date) & (df_test['Timestamp_helper']<= end_date)]
            relev_df1 = df_test[(df_test['Timestamp_helper'] >= start_date1) & (df_test['Timestamp_helper']<= end_date1)]
        else:
            relev_df = pd.DataFrame()
            relev_df1 = pd.DataFrame()

        if not relev_df.empty:
            anomalies = np.isin(relev_df[self.qc_column], [3])
            start = relev_df.index[0] - df_test.index[0]
            end = relev_df.index[-1] - df_test.index[0] + 1
            anomalies_pred_short = anomalies_pred[start:end]
            # Identify False Positives (FP) and False Negatives (FN)
            fp = (anomalies_pred_short == True) & (anomalies == False)  # False Positives (FP)
            fn = (anomalies_pred_short == False) & (anomalies == True)  # False Negatives (FN)
            tp = (anomalies_pred_short == True) & (anomalies == True)  # True Positives (TP)
            self.basic_plotter(relev_df, title, anomalies, anomalies_pred_short, fn, fp, tp)
        
        if not relev_df1.empty:
            anomalies = np.isin(relev_df1[self.qc_column], [3])
            start = relev_df1.index[0] - df_test.index[0]
            end = relev_df1.index[-1] - df_test.index[0] + 1
            anomalies_pred1 = anomalies_pred[start:end]
            # Identify False Positives (FP) and False Negatives (FN)
            fp = (anomalies_pred1 == True) & (anomalies == False)  # False Positives (FP)
            fn = (anomalies_pred1 == False) & (anomalies == True)  # False Negatives (FN)
            tp = (anomalies_pred1 == True) & (anomalies == True)  # True Positives (TP)
            self.basic_plotter(relev_df1, title, anomalies, anomalies_pred1, fn, fp, tp)

        del df_test['Timestamp_helper']

        df_test['ml_anomaly_predicted'] = y_pred
        
        return df_test

    def basic_plotter_no_xaxis(self, df, title, anomalies, anomalies2=np.array([])):
        #Visualization of station data
        plt.figure(figsize=(11, 6))
        plt.plot(df[self.measurement_column], label='Measured Water Level', color='black', marker='o',  markersize=2, linestyle='None')
        plt.scatter(df.index[anomalies], df[self.measurement_column][anomalies], color='blue', label='Manual spike label', facecolors='none', edgecolors='blue', zorder=0.5)
        # Add QC flags
        if not anomalies2.size == 0:
            plt.scatter(df.index[anomalies2], df[self.measurement_column][anomalies2], color='red', label='ML spike prediction', zorder=0.2)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))  # Format: YYYY-MM-DD HH:MM
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(plt.gca().get_xticks()[::2])
        plt.legend(frameon=False)
        plt.xlabel('Timestamp')
        plt.ylabel('Water Level [m]')
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder_path,f"{title}- Date: {df[self.time_column].iloc[0]}.png"), dpi=300, bbox_inches="tight")
        plt.close() 

    def basic_plotter(self, df, title, anomalies, anomalies2=np.array([]), fn=np.array([]), fp=np.array([]), tp=np.array([])):
        #Visualization of station data
        plt.figure(figsize=(11, 6))
        plt.plot(df['Timestamp'], df[self.measurement_column], label='Measured Water Level', color='black', marker='o',  markersize=1, linestyle='None')
        #plt.scatter(df['Timestamp'][anomalies], df[self.measurement_column][anomalies], color='blue', marker='*', label='Manual spike label', edgecolors='blue', zorder=1)
        # Add QC flags
        #if not anomalies2.size == 0:
        #    plt.scatter(df['Timestamp'][anomalies2], df[self.measurement_column][anomalies2], color='red', label='ML spike prediction', zorder=0.1)
        #Add false negatives
        if not fn.size == 0:
            plt.scatter(df['Timestamp'][fn], df[self.measurement_column][fn], color='lawngreen', label='False Negatives (FN)', zorder=0.15)
        #Add false positives
        if not fp.size == 0:
            plt.scatter(df['Timestamp'][fp], df[self.measurement_column][fp], color='blue', label='False Positives (FP)', zorder=0.15)
        #Add true positives
        if not tp.size == 0:
            plt.scatter(df['Timestamp'][tp], df[self.measurement_column][tp], color='red', label='True Positives (TP)', zorder=0.1)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))  # Format: YYYY-MM-DD HH:MM
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(plt.gca().get_xticks()[::2])
        plt.legend(frameon=False)
        plt.xlabel('Timestamp')
        plt.ylabel('Water Level [m]')
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder_path,f"{title}- Date: {df[self.time_column].iloc[0]}.png"), dpi=300, bbox_inches="tight")
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

        