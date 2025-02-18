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
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

class MLOutlierDetection(): 

    def __init__(self):

        self.helper = helper.HelperMethods()

        #Empty element to store different texts about QC tests and save as a summary document
        self.information = []

        #Define percentage of data used for training and testing (here: 80/20%)
        self.train_size = 0.15
        self.val_size = 0.15

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
                station_df[self.measurement_column] = station_df[self.measurement_column] - station_df[self.measurement_column].mean()
                station_df[self.qc_column] = station_df[self.qc_column].fillna(1).astype(int)
                station_df[self.time_column] = pd.to_datetime(station_df[self.time_column])
            elif file.endswith(ending):
                df = pd.read_csv(os.path.join(folder_path,file), sep=",", header=0)
                df = df[[self.time_column, self.measurement_column, self.qc_column]]
                df[self.measurement_column] = df[self.measurement_column] - df[self.measurement_column].mean()
                df[self.qc_column] = df[self.qc_column].fillna(1).astype(int)
                df[self.time_column] = pd.to_datetime(df[self.time_column])
                combined_df.append(df)

        self.labelled_df = pd.concat(combined_df, ignore_index=True)
        self.labelled_df = self.labelled_df.sort_values(by=self.time_column)
        self.station_df = station_df

        return self.labelled_df, station_df

    def run(self, df, station_df):

        df_train, df_test = self.split_dataset(station_df)

        #Alter imbalanced dataset in order to even out the minority class
        self.preprocessing_data_imb_learn(df_train)
        #self.preprocessing_data(df_train)

        # Fit and transform labels
        self.y_train_transformed = self.le.fit_transform(self.y_train)
        self.y_test_transformed = self.le.fit_transform(self.y_test)
        self.y_val_transformed = self.le.fit_transform(self.y_val)
        self.y_train_res = self.le.fit_transform(self.y_train_res)
        # Print mapping
        print(dict(zip(self.le.classes_, self.le.transform(self.le.classes_))))
        
        self.run_binary(df_train, df_test)
        #self.run_multi(station_df, df)

    def split_dataset(self, df):

        #Split into train and test
        df_train = df[int((self.train_size*len(df))):]
        df_test = df[:int((self.train_size*len(df)))]
        #Split training dataset again into validation and training data
        df_val = df_train[:int((self.val_size*len(df_train)))]
        df_train = df_train[int((self.val_size*len(df_train))):]
        #Generate correct data subsets
        self.X_train = df_train[self.measurement_column]
        self.y_train = df_train[self.qc_column]
        self.X_test = df_test[self.measurement_column]
        self.y_test = df_test[self.qc_column]
        self.X_val = df_val[self.measurement_column]
        self.y_val = df_val[self.qc_column]
        #For Randomforest
        self.X_train_2d = self.X_train.values.reshape(-1, 1)  # Convert Series to 2D array
        self.X_test_2d = self.X_test.values.reshape(-1, 1)  # Convert Series to 2D array

        #Visualization training data
        anomalies_train = df_train[self.qc_column].isin([3, 4])
        title = 'QC classes for Training- Binary'
        plt.figure(figsize=(12, 6))
        plt.plot(df_train[self.time_column], df_train[self.measurement_column], label='Time Series', color='black', marker='o',  markersize=0.5, linestyle='None')
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

        return df_train, df_test
    
    def preprocessing_data(self, df):
        #Undersampling: Mark 40% of random good rows to be deleted
        rows_to_drop = df[df[self.qc_column]==1].sample(frac=0.3, random_state=42).index
        # Drop selected rows
        df_res = df.drop(rows_to_drop).reset_index(drop=True)
        
        #Oversampling:Filter rows which bad qc flag and multiply them and assign them randomly 
        bad_rows = df_res[df_res[self.qc_column].isin([3,4])]
        bad_rows_expanded = pd.concat([bad_rows] * 4, ignore_index=True)
        bad_rows_expanded[self.time_column] = bad_rows_expanded[self.time_column]  + pd.to_timedelta(np.random.randint(5, 10, size=len(bad_rows_expanded)), unit='m')
        new_bad_rows = df_res[df_res[self.time_column].isin(bad_rows_expanded[self.time_column])].copy()
        df_res.loc[df_res[self.time_column].isin(new_bad_rows[self.time_column]), self.measurement_column] = new_bad_rows[self.measurement_column]
        df_res.loc[df_res[self.time_column].isin(new_bad_rows[self.time_column]), self.qc_column] = 3
        
        # Oversampling:Filter rows which bad qc flag and multiply them and assign them randomly 
        bad_rows = df_res[df_res[self.qc_column].isin([3,4])]
        bad_rows_expanded = pd.concat([bad_rows] * 2, ignore_index=True)
        random_times = df_res[self.time_column].sample(n=len(bad_rows_expanded), random_state=42)
        more_bad_values = bad_rows_expanded[self.measurement_column].values
        np.random.shuffle(more_bad_values)
        df_res.loc[random_times.index, self.measurement_column] = more_bad_values
        df_res.loc[random_times.index, self.qc_column] = 3

        #Visualization resampled data
        anomalies_res = df_res[self.qc_column].isin([3, 4])
        title = 'QC classes - Resampled'
        plt.figure(figsize=(12, 6))
        plt.plot(df_res[self.time_column], df_res[self.measurement_column], label='Time Series', color='black', marker='o',  markersize=0.5, linestyle='None')
        plt.scatter(df_res[self.time_column][anomalies_res], df_res[self.measurement_column][anomalies_res], color='red', label='QC Classes', zorder=1)
        plt.legend()
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Water Level')
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder_path,f"{title}.png"),  bbox_inches="tight")
        plt.close()

        #Analysis
        unique_values, counts = np.unique(df_res[self.qc_column], return_counts=True)
        for value, count in zip(unique_values, counts):
            print(f"Class {value} exists {count} times in the resampled dataset. This is {count/len(df_res)}.")

        #Need balanced output
        self.X_train_res = df_res[self.measurement_column].values.reshape(-1, 1)  # Convert Series to 2D array
        self.y_train_res = df_res[self.qc_column].values.reshape(-1, 1)  # Convert Series to 2D array

    def preprocessing_data_imb_learn(self,df):
        # Define XGBoost model
        # Apply undersampling to reduce the majority class in the training data
        undersampler = RandomUnderSampler(sampling_strategy = 0.01, random_state=42)
        #self.X_train_res, self.y_train_res = undersampler.fit_resample(self.X_train_2d, self.y_train_transformed)

        #Creating an instance of SMOTE
        #smote = SMOTE(random_state=42, sampling_strategy=0.1)
        smote = RandomOverSampler(random_state=42, sampling_strategy='not majority')
        self.X_train_res, self.y_train_res = smote.fit_resample(df[self.measurement_column].values.reshape(-1, 1), df[self.qc_column])
        #Balancing the data
        pipeline = Pipeline([('under', undersampler), ('over', smote)])
        smotetomek = SMOTETomek(random_state=42, sampling_strategy=0.2)
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
        plt.ylabel('Water Level')
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder_path,f"{title}.png"),  bbox_inches="tight")
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
        plt.ylabel('Water Level')
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

        #best_params = grid_search.best_params_ 
        #print(f"Best parameters: {best_params}")

        #Generate the model
        grid_search = GridSearchCV(XGBClassifier(), param_grid, cv=3, verbose=1)
        #grid_search.fit(self.X_train_res, self.y_train_res)
        #best_params = grid_search.best_params_ 
        #print(f"Best parameters: {best_params}")

        #scale_pos_weight = sum(y_train_transformed == 0) / sum(y_train_transformed == 1)
        #scale_pos_weight = sum(y_train_res == 0) / sum(y_train_res == 1)
        #model = XGBClassifier(objective="binary:logistic", scale_pos_weight=scale_pos_weight, eval_metric="logloss", max_depth=4, learning_rate=0.01, n_estimators=200, random_state=42)
        model = XGBClassifier(objective="binary:logistic", eval_metric="logloss", max_depth=4, learning_rate=0.01, n_estimators=200, random_state=42)
        #model = XGBClassifier(objective="binary:logistic", eval_metric="aucpr", random_state=42, **best_params)
        #model = RandomForestClassifier(n_estimators=200, random_state=42)
        #model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)

        # Train the model
        #model.fit(self.X_train_res, self.y_train_res, eval_set=[(self.X_train_res, self.y_train_res), (self.X_val, self.y_val_transformed)], verbose=True)
        model.fit(self.X_train_res, self.y_train_res)
        #model.fit(self.X_train, self.y_train_transformed, eval_set=[(X_train, y_train_transformed), (X_val, y_val_transformed)], verbose=True)
        #model.fit(self.X_train, self.y_train_transformed)

        # Make predictions (test)
        y_pred_prob = model.predict(self.X_test_2d)
        y_pred_transformed = (y_pred_prob > 0.3).astype(int)
        y_pred = self.le.inverse_transform(y_pred_transformed)
        
        # Make predictions (train)
        y_pred_train_prob = model.predict(self.X_train_2d)
        y_pred_train_transformed = (y_pred_train_prob > 0.3).astype(int)
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
        cm = confusion_matrix(self.y_train, y_pred_train)
        print("Confusion Matrix for training data:")
        print(cm)

        # Identify anomalies
        anomalies_test = self.y_test.isin([3, 4])
        anomalies_pred = np.isin(y_pred, [3, 4])
        anomalies_train = np.isin(y_pred_train, [3, 4])

        #1.Visualization
        title = 'QC classes (Testing data)- Binary'
        plt.figure(figsize=(12, 6))
        plt.plot(df_test[self.time_column], df_test[self.measurement_column], label='Time Series', color='black', marker='o',  markersize=0.5)
        plt.scatter(df_test[self.time_column][anomalies_test], df_test[self.measurement_column][anomalies_test], color='red', label='QC Classes', zorder=1)
        plt.legend()
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Water Level')
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder_path,f"{title}-test.png"),  bbox_inches="tight")
        plt.close()

        #2.Visualization
        title = 'ML predicted QC classes (for testing data)- Binary'
        plt.figure(figsize=(12, 6))
        plt.plot(df_test[self.time_column], df_test[self.measurement_column], label='Time Series', color='black', marker='o',  markersize=0.5)
        plt.scatter(df_test[self.time_column][anomalies_pred], df_test[self.measurement_column][anomalies_pred], color='red', label='QC Classes', zorder=1)
        plt.legend()
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Water Level')
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder_path,f"{title}-pred-test.png"),  bbox_inches="tight")
        plt.close()

        title = 'ML predicted QC classes (for training data)- Binary'
        plt.figure(figsize=(12, 6))
        plt.plot(df_train[self.time_column], df_train[self.measurement_column], label='Time Series', color='black', marker='o',  markersize=0.5)
        plt.scatter(df_train[self.time_column][anomalies_train], df_train[self.measurement_column][anomalies_train], color='red', label='QC Classes', zorder=1)
        plt.legend()
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Water Level')
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder_path,f"{title}-pred-training.png"),  bbox_inches="tight")
        plt.close()

        print('hey')


    def run_multi(self, station_df, df):
        
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

        # Calculate class distribution
        # Compute sample weights for imbalanced classes
        #sample_weights = compute_sample_weight(class_weight='balanced', y=y_train_transformed)
        sample_weights = compute_sample_weight(class_weight='balanced', y=self.y_train_res)
        #sample_weights = sample_weights * 0.75
        print(f"Scale Weights for each class (based on training set): {sample_weights}")

        #Generate the model
        num_classes = len(np.unique(self.y_train))
        model = XGBClassifier(objective="multi:softmax", num_class=num_classes, eval_metric="mlogloss", n_estimators=300)
        #model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)

        # Train the model
        #model.fit(self.X_train, self.y_train_transformed, sample_weight=sample_weights)
        model.fit(self.X_train_res, self.y_train_res)
        #model.fit(self.X_train_res, self.y_train_res, sample_weight=sample_weights)
        #model.fit(self.X_train, self.y_train_transformed)

        # Make predictions
        y_pred_transformed = model.predict(self.X_test)
        #y_pred_transformed = model.predict(self.X_test_2d)
        y_pred = self.le.inverse_transform(y_pred_transformed)
        
        # Visualization
        #Print-statement
        unique_values, counts = np.unique(y_pred, return_counts=True)
        for value, count in zip(unique_values, counts):
            print(f"Class {value} was predicted {count} times.")
        
        # Create confusion matrix
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