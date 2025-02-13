#######################################################################################################################
## Written by frb for GronSL project (2024-2025)                                                                     ##
## This is the main ML-plug-in script for the QC. It splits the ts in 3 different classes (good, probably bad, bad). ##
#######################################################################################################################

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import builtins
import json

import imblearn
print(imblearn.__version__)

import source.helper_methods as helper

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

class MLOutlierDetection(): 

    def __init__(self):

        self.helper = helper.HelperMethods()

        #Empty element to store different texts about QC tests and save as a summary document
        self.information = []

        #Define percentage of data used for training and testing (here: 80/20%)
        self.train_size = 0.75

        # Initialize encoder
        self.le = LabelEncoder()

    def set_station(self, station):
        self.station = station
        self.information.append(['The following text summarizes the supervised ML QC perfromed on measurements from ', station,':'])
    
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

        #Open .csv file and fix the column names
        for file in os.listdir(folder_path):  # Loops through files in the current directory
            print(file)
            if file.endswith(ending) and self.station in file:
                station_df = pd.read_csv(os.path.join(folder_path,file), sep=",", header=0)
                station_df = station_df[[self.time_column, self.measurement_column, self.qc_column]]
                station_df[self.qc_column] = station_df[self.qc_column].fillna(1).astype(int)
                station_df[self.time_column] = pd.to_datetime(station_df[self.time_column])
            elif file.endswith(ending):
                df = pd.read_csv(os.path.join(folder_path,file), sep=",", header=0)
                df = df[[self.time_column, self.measurement_column, self.qc_column]]
                df[self.qc_column] = df[self.qc_column].fillna(1).astype(int)
                df[self.time_column] = pd.to_datetime(df[self.time_column])
                combined_df.append(df)

        self.labelled_df = pd.concat(combined_df, ignore_index=True)
        self.labelled_df = self.labelled_df.sort_values(by=self.time_column)
        self.station_df = station_df

        return self.labelled_df, station_df

    def run(self, df, station_df):

        # Split into train and test
        df_test = station_df[int((self.train_size*len(station_df))):]
        df_train = pd.concat([df, station_df[:int((self.train_size*len(station_df)))]], ignore_index=True) 
        X_train = df_train[self.measurement_column]
        y_train = df_train[self.qc_column]
        X_test = df_test[self.measurement_column]
        y_test = df_test[self.qc_column]

        #Visualization training data
        anomalies_train = df_train[self.qc_column].isin([3, 4])
        title = 'ML predicted QC classes (Training)'
        plt.figure(figsize=(12, 6))
        plt.plot(df_train[self.time_column], df_train[self.measurement_column], label='Time Series', color='black', marker='o',  markersize=0.5)
        plt.scatter(df_train[self.time_column][anomalies_train], df_train[self.measurement_column][anomalies_train], color='red', label='QC Classes', zorder=1)
        plt.legend()
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Water Level')
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder_path,f"{title}- Date: {df[self.time_column][int((self.train_size*len(df)))]}-train.png"),  bbox_inches="tight")
        plt.close()

        #Analyse existing dataset
        unique_values, counts = np.unique(df_train[self.qc_column], return_counts=True)
        for value, count in zip(unique_values, counts):
            print(f"Class {value} exists {count} times in training dataset. This is {count/len(df_train)}.")

        unique_values, counts = np.unique(df_test[self.qc_column], return_counts=True)
        for value, count in zip(unique_values, counts):
            print(f"Class {value} exists {count} times in testing dataset. This is {count/len(df_test)}.")

        # Fit and transform labels
        y_train_transformed = self.le.fit_transform(y_train)
        y_test_transformed = self.le.transform(y_test)  # Ensure consistency
        # Print mapping
        print(dict(zip(self.le.classes_, self.le.transform(self.le.classes_))))

        #Get best XGBClassifier model based on hyperparameters
        param_grid = {
            'max_depth': [1, 2, 3, 5],
            'min_child_weight': [1, 3, 6, 10],
            'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1],
            'n_estimators': [10, 50, 100, 500],
            'gamma': [0, 0.1, 0.5, 1]
        }

        #grid_search = GridSearchCV(XGBClassifier(), param_grid, cv=3, verbose=1)
        #grid_search.fit(X_train, y_train_transformed)

        #best_params = grid_search.best_params_ 
        #print(f"Best parameters: {best_params}")

        # Define XGBoost model
        # Apply undersampling to reduce the majority class in the training data
        undersampler = RandomUnderSampler(sampling_strategy = {0: 0.6, 1: 1.0, 2: 1.0}, random_state=42)
        X_train_2d = X_train.values.reshape(-1, 1)  # Convert Series to 2D array
        X_test_2d = X_test.values.reshape(-1, 1)  # Convert Series to 2D array
        X_train_res, y_train_res = undersampler.fit_resample(X_train_2d, y_train_transformed)

        #Creating an instance of SMOTE
        smote = SMOTE(random_state=42, sampling_strategy={0: 1.0, 1: 1.5, 2: 2.0})
        #X_train_res, y_train_res = smote.fit_resample(X_train_2d, y_train_transformed)
        #Balancing the data
        pipeline = Pipeline([('under', undersampler), ('over', smote)])
        #X_train_res, y_train_res = pipeline.fit_resample(X_train_2d, y_train_transformed)

        # Calculate class distribution
        # Compute sample weights for imbalanced classes
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_train_transformed)
        sample_weights = sample_weights * 0.75
        print(f"Scale Weights for each class (based on training set): {sample_weights}")

        #Generate the model
        num_classes = len(np.unique(y_train))
        model = XGBClassifier(objective="multi:softmax", num_class=num_classes, eval_metric="mlogloss", n_estimators=200)
        #model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)

        # Train the model
        #model.fit(X_train, y_train_transformed, sample_weight=sample_weights)
        model.fit(X_train_res, y_train_res)
        #model.fit(X_train, y_train_transformed)

        # Make predictions
        y_pred_transformed = model.predict(X_test)
        #y_pred_transformed = model.predict(X_test_2d)
        y_pred = self.le.inverse_transform(y_pred_transformed)
        
        # Visualization
        #Print-statement
        unique_values, counts = np.unique(y_pred, return_counts=True)
        for value, count in zip(unique_values, counts):
            print(f"Class {value} was predicted {count} times.")
        
        # Create confusion mateix
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(cm)

        # Identify anomalies
        anomalies_test = df_test[self.qc_column].isin([3, 4])
        anomalies_pred = np.isin(y_pred, [3, 4])

        #1.Visualization
        title = 'ML predicted QC classes (Test)'
        plt.figure(figsize=(12, 6))
        plt.plot(df_test[self.time_column], df_test[self.measurement_column], label='Time Series', color='black', marker='o',  markersize=0.5)
        plt.scatter(df_test[self.time_column][anomalies_test], df_test[self.measurement_column][anomalies_test], color='red', label='QC Classes', zorder=1)
        plt.legend()
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Water Level')
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder_path,f"{title}- Date: {df[self.time_column][int((self.train_size*len(df)))]}-test.png"),  bbox_inches="tight")
        plt.close()

        #2.Visualization
        title = 'ML predicted QC classes (Pred)'
        plt.figure(figsize=(12, 6))
        plt.plot(df_test[self.time_column], df_test[self.measurement_column], label='Time Series', color='black', marker='o',  markersize=0.5)
        plt.scatter(df_test[self.time_column][anomalies_pred], df_test[self.measurement_column][anomalies_pred], color='red', label='QC Classes', zorder=1)
        plt.legend()
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Water Level')
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder_path,f"{title}- Date: {df[self.time_column][int((self.train_size*len(df)))]}-pred.png"),  bbox_inches="tight")
        plt.close()

        print('hey')