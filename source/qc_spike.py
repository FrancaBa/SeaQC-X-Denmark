import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import random

from cotede import datasets, qctests
from scipy.signal import argrelextrema

from xgboost import XGBRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import source.helper_methods as helper

class SpikeDetector():

    def __init__(self):
        self.helper = helper.HelperMethods()

        self.cotede_threshold = 0.01
        self.improved_cotede_threshold = 0.02
        self.max_window_neighbours = 60 #under the assumption that timestamp is in 1 min

        # Parameters for ml
        self.n_lags = 500       # Number of lag features
        self.rolling_window = 30  # Rolling window size for additional features
        self.forecast_horizon = 10  # Predict 10 steps ahead
        self.test_size = 10000    # Size of the test set

        #Selene package needed constants
        self.splinedegree = 2 #3
        self.winsize = 200
        self.nsigma = 4
        self.maxiter = 3

    def set_output_folder(self, folder_path):
        self.helper.set_output_folder(folder_path)

    def remove_spikes_cotede(self, df_meas_long, adapted_meas_col_name, quality_column_name, time_column, measurement_column, qc_classes):
        # used 'altered WL' series to detect spike as it is already cleaned (NOT raw measurement series)
        # The spike check is a quite traditional one and is based on the principle of comparing one measurement with the tendency observed from the neighbor values.
        # This is already implemented in CoTeDe as qctests.spike
        # Revelant weaknesses:
        # 1. Cannot detect spikes when NaN in the neighbourhood
        # 2. Cannot detect smaller, continious spiking behaviour

        # If no threshold is required, you can do like this:
        sea_level_spike = qctests.spike(df_meas_long[adapted_meas_col_name])  
        print("The largest spike observed was: {:.3f}".format(np.nanmax(np.abs(sea_level_spike))))
        print("Value at detected position", sea_level_spike[np.nanargmax(np.abs(sea_level_spike))])

        #Assessment of max spike via plots
        min_value = np.nanargmax(np.abs(sea_level_spike)) - 500
        max_value = np.nanargmax(np.abs(sea_level_spike)) + 1500
        self.helper.plot_df(df_meas_long[time_column][min_value:max_value], df_meas_long[adapted_meas_col_name][min_value:max_value],'Water Level','Timestamp','Measured water level wo outliers in 1 min timestamp (zoomed to max spike)')
        self.helper.plot_df(df_meas_long[time_column], sea_level_spike,'Detected spike [m]','Timestamp','WL spikes in measured ts')

        # If the spike is less than 1 cm (absolut value), it is not a spike -> Do not remove this small noise
        sea_level_spike[abs(sea_level_spike) < self.cotede_threshold] = 0.0
        #Plot together
        self.helper.plot_two_df(df_meas_long[time_column][min_value:max_value], sea_level_spike[min_value:max_value],'Detected spikes', df_meas_long[measurement_column][min_value:max_value], 'Water Level', 'Timestamp','Spikes and measured WL')

        #balance out spike
        df_meas_long['WaterLevel_shiftedpast'] = df_meas_long[adapted_meas_col_name].shift(50)
        df_meas_long['WaterLevel_shiftedfuture'] = df_meas_long[adapted_meas_col_name].shift(-50)
        df_meas_long['bound'] = (df_meas_long['WaterLevel_shiftedfuture']+df_meas_long['WaterLevel_shiftedpast'])/2
        df_meas_long[adapted_meas_col_name] = np.where(
            df_meas_long['bound'] <= df_meas_long[adapted_meas_col_name],  # Condition 1: bound <= WaterLevel
            df_meas_long[adapted_meas_col_name] - abs(sea_level_spike),         # Action 1: WaterLevel - abs(sea_level_spike)
            np.where(
                df_meas_long['bound'] > df_meas_long[adapted_meas_col_name],  # Condition 2: bound > WaterLevel
                df_meas_long[adapted_meas_col_name] + abs(sea_level_spike),         # Action 2: WaterLevel + abs(sea_level_spike)
                df_meas_long[adapted_meas_col_name]                                    # Else: keep original value from 'altered'
            )
        )

        #Mask spikes
        sea_level_spike_bool = (~np.isnan(sea_level_spike)) & (sea_level_spike != 0)
        df_meas_long.loc[(df_meas_long[quality_column_name] == qc_classes['good_data']) & (sea_level_spike_bool), quality_column_name] = qc_classes['bad_data_correctable']
        if qc_classes['bad_data_correctable'] in df_meas_long[quality_column_name].unique():
            ratio = (len(df_meas_long[df_meas_long[quality_column_name] == qc_classes['bad_data_correctable']])/len(df_meas_long))*100
            print(f"There are {len(df_meas_long[df_meas_long[quality_column_name] == qc_classes['bad_data_correctable']])} spikes in this timeseries. This is {ratio}% of the overall dataset.")

        #More plots
        for i in range(1, 41):
            min = random.randint(np.where(~np.isnan(sea_level_spike))[0][0], len(df_meas_long)-4000)
            max = min + 3000
            self.helper.plot_two_df_same_axis(df_meas_long[time_column][min:max], df_meas_long[adapted_meas_col_name][min:max],'Water Level', 'Water Level (corrected)', df_meas_long[measurement_column][min:max], 'Timestamp', 'Water Level (measured)',f'Graph-Cotede{i}- Measured water level wo outliers and spikes in 1 min timestamp vs orig (zoomed to max spike) (improved)')


        #delete helper columns
        df_meas_long = df_meas_long.drop(columns=['WaterLevel_shiftedpast', 'WaterLevel_shiftedfuture', 'bound'])
        
        #Analyse spike detection
        self.helper.plot_df(df_meas_long[time_column], df_meas_long[adapted_meas_col_name],'Water Level','Timestamp','Measured water level wo outliers and spikes in 1 min timestamp (all)')
        self.helper.plot_two_df(df_meas_long[time_column][min_value:max_value], df_meas_long[adapted_meas_col_name][min_value:max_value],'Water Level', df_meas_long[quality_column_name][min_value:max_value], 'Quality flag', 'Timestamp','Measured water level wo outliers and spikes in 1 min timestamp (zoomed to max spike)')
        self.helper.plot_two_df(df_meas_long[time_column], df_meas_long[adapted_meas_col_name],'Water Level', df_meas_long[quality_column_name], 'Quality flag', 'Timestamp','Measured water level wo outliers and spikes in 1 min timestamp incl. flags')
        self.helper.plot_two_df(df_meas_long[time_column], df_meas_long[measurement_column],'Water Level', df_meas_long[quality_column_name], 'Quality flag', 'Timestamp','Measured water level in 1 min timestamp incl. flags')
        self.helper.plot_two_df_same_axis(df_meas_long[time_column][min_value:max_value], df_meas_long[adapted_meas_col_name][min_value:max_value],'Water Level', 'Water Level (corrected)', df_meas_long[measurement_column][min_value:max_value], 'Timestamp', 'Water Level (measured)', 'Measured water level wo outliers and spikes in 1 min timestamp vs orig (zoomed to max spike)')
        #Some more plots for assessment
        self.max_value_plotting = max_value
        self.min_value_plotting = min_value
        min_value = np.nanargmax(np.abs(df_meas_long[adapted_meas_col_name])) - 500
        max_value = np.nanargmax(np.abs(df_meas_long[adapted_meas_col_name])) + 500
        self.helper.plot_two_df(df_meas_long[time_column][min_value:max_value], df_meas_long[adapted_meas_col_name][min_value:max_value],'Water Level', df_meas_long[quality_column_name][min_value:max_value], 'Quality flag', 'Timestamp','Measured water level wo outliers and wo spike in 1 min timestamp (zoomed to max spike after removing spike)')
        #is deshifting of spike working?
        self.helper.plot_df(df_meas_long[time_column][:1000000], df_meas_long[adapted_meas_col_name][:1000000],'Water Level','Timestamp','Measured water level wo outliers and spikes in 1 min timestamp 1')
        self.helper.plot_df(df_meas_long[time_column][1000000:2000000], df_meas_long[adapted_meas_col_name][1000000:2000000],'Water Level','Timestamp','Measured water level wo outliers and spikes in 1 min timestamp2')
        self.helper.plot_df(df_meas_long[time_column][2000000:3000000], df_meas_long[adapted_meas_col_name][2000000:3000000],'Water Level','Timestamp','Measured water level wo outliers and spikes in 1 min timestamp3')
        self.helper.plot_df(df_meas_long[time_column][3000000:4000000], df_meas_long[adapted_meas_col_name][3000000:4000000],'Water Level','Timestamp','Measured water level wo outliers and spikes in 1 min timestamp4')
        #ubsequent line makes the code slow, only enable when need
        #self.helper.zoomable_plot_df(df_meas_long[time_column][:1000000], df_meas_long[measurement_column][:1000000],'Water Level','Timestamp','Measured water level wo outliers and spikes in 1 min timestamp', 'measured water level')
        #self.helper.zoomable_plot_df(df_meas_long[time_column][:1000000], df_meas_long[measurement_column][:1000000],'Water Level','Timestamp','Measured water level wo outliers and spikes in 1 min timestamp', 'measured water level')
        #self.helper.zoomable_plot_df(df_meas_long[time_column][:1000000], df_meas_long[measurement_column][:1000000],'Water Level','Timestamp','Measured water level wo outliers and spikes in 1 min timestamp', 'measured water level')
        #self.helper.zoomable_plot_df(df_meas_long[time_column][:1000000], df_meas_long[measurement_column][:1000000],'Water Level','Timestamp','Measured water level wo outliers and spikes in 1 min timestamp', 'measured water level')
        #self.helper.zoomable_plot_df(df_meas_long[time_column][:1000000], df_meas_long[measurement_column][:1000000],'Water Level','Timestamp','Measured water level wo outliers and spikes in 1 min timestamp', 'measured water level')
        #self.helper.zoomable_plot_df(df_meas_long[time_column][:1000000], df_meas_long[measurement_column][:1000000],'Water Level','Timestamp','Measured water level wo outliers and spikes in 1 min timestamp', 'measured water level')
                
    """  
    Improve the cotede script by looking at the next existing neighbours and not just the next nieghbours
    Now: Neighbours within 2 hours of the center point are allowed
    This allows to work with different time scales and detect spikes even there are NaN values
    """  

    def remove_spikes_cotede_improved(self, df_meas_long, adapted_meas_col_name, quality_column_name, time_column, measurement_column, qc_classes):
        self.get_valid_neighbours(df_meas_long, adapted_meas_col_name, 'next_neighbour', True, max_distance=self.max_window_neighbours)
        self.get_valid_neighbours(df_meas_long, adapted_meas_col_name, 'past_neighbour', False, max_distance=self.max_window_neighbours)
        sea_level_spikes = np.abs(df_meas_long[adapted_meas_col_name] - (df_meas_long['past_neighbour'] + df_meas_long['next_neighbour']) / 2.0) - np.abs((df_meas_long['next_neighbour'] - df_meas_long['past_neighbour']) / 2.0)
        sea_level_spikes[abs(sea_level_spikes) < self.improved_cotede_threshold] = 0.0
        self.helper.plot_df(df_meas_long[time_column], sea_level_spikes,'Detected spike [m]','Timestamp ','WL spikes in measured ts (improved)')

        #Remove spike
        df_meas_long['bound'] = (df_meas_long['next_neighbour']+df_meas_long['past_neighbour'])/2
        df_meas_long[adapted_meas_col_name] = np.where(
            df_meas_long['bound'] <= df_meas_long[adapted_meas_col_name],  # Condition 1: bound <= WaterLevel
            df_meas_long[adapted_meas_col_name] - abs(sea_level_spikes),         # Action 1: WaterLevel - abs(sea_level_spike)
            np.where(
                df_meas_long['bound'] > df_meas_long[adapted_meas_col_name],  # Condition 2: bound > WaterLevel
                df_meas_long[adapted_meas_col_name] + abs(sea_level_spikes),         # Action 2: WaterLevel + abs(sea_level_spike)
                df_meas_long[adapted_meas_col_name]                                    # Else: keep original value from 'altered'
            )
        )

        #Mask spikes to flags
        sea_level_spike_bool = (~np.isnan(sea_level_spikes)) & (sea_level_spikes != 0)
        df_meas_long.loc[(df_meas_long[quality_column_name] == qc_classes['good_data']) & (sea_level_spike_bool), quality_column_name] = qc_classes['bad_data_correctable']
        if qc_classes['bad_data_correctable'] in df_meas_long[quality_column_name].unique():
            ratio = (len(df_meas_long[df_meas_long[quality_column_name] == qc_classes['bad_data_correctable']])/len(df_meas_long))*100
            print(f"There are {len(df_meas_long[df_meas_long[quality_column_name] == qc_classes['bad_data_correctable']])} spikes in this timeseries. This is {ratio}% of the overall dataset.")

        #Analyse spike detection
        self.helper.plot_df(df_meas_long[time_column], df_meas_long[adapted_meas_col_name],'Water Level','Timestamp ','Measured water level wo outliers and spikes in 1 min timestamp (all) (improved)')
        self.helper.plot_two_df(df_meas_long[time_column][self.min_value_plotting:self.max_value_plotting], df_meas_long[adapted_meas_col_name][self.min_value_plotting:self.max_value_plotting],'Water Level', df_meas_long[quality_column_name][self.min_value_plotting:self.max_value_plotting], 'Quality flag', 'Timestamp ','Measured water level wo outliers and spikes in 1 min timestamp (zoomed to max spike) (improved)')
        self.helper.plot_two_df_same_axis(df_meas_long[time_column][self.min_value_plotting:self.max_value_plotting], df_meas_long[adapted_meas_col_name][self.min_value_plotting:self.max_value_plotting],'Water Level', 'Water Level (corrected)', df_meas_long[measurement_column][self.min_value_plotting:self.max_value_plotting], 'Timestamp ', 'Water Level (measured)', 'Measured water level wo outliers and spikes in 1 min timestamp vs orig (zoomed to max spike) (improved)')
        self.helper.plot_two_df(df_meas_long[time_column], df_meas_long[adapted_meas_col_name],'Water Level', df_meas_long[quality_column_name], 'Quality flag', 'Timestamp ','Corrected water level wo outliers and spikes in 1 min timestamp (improved)')
        self.helper.plot_two_df(df_meas_long[time_column], df_meas_long[measurement_column],'Water Level', df_meas_long[quality_column_name], 'Quality flag', 'Timestamp ','Measured water level in 1 min timestamp incl. flags (improved)')
        
        #More plots
        for i in range(1, 41):
            min = random.randint(1, len(df_meas_long)-4000)
            max = min + 4000
            self.helper.plot_two_df_same_axis(df_meas_long[time_column][min:max], df_meas_long[adapted_meas_col_name][min:max],'Water Level', 'Water Level (corrected)', df_meas_long[measurement_column][min:max], 'Timestamp ', 'Water Level (measured)', f'Graph{i}- Measured water level wo outliers and spikes in 1 min timestamp vs orig (zoomed to max spike) (improved)')
            self.helper.plot_two_df(df_meas_long[time_column][min:max], df_meas_long[measurement_column][min:max],'Water Level', df_meas_long[quality_column_name][min:max], 'Quality flag', 'Timestamp ','Graph{i}- Measured water level incl. flags')
        
    def get_valid_neighbours(self, df, column, column_name, shift_future, max_distance=60):

        #Create a list of shifted columns for 60 min shifted values (depends on max_distance)
        # Create shifted columns using numpy (pandas memory issue)
        if shift_future:
            for i in range(1, max_distance + 1):
                df[f'shift_{i}'] = np.roll(df[column], -i)
                df.iloc[-i:, df.columns.get_loc(f'shift_{i}')] = np.nan
        else:
            for i in range(1, max_distance + 1):
                df[f'shift_{i}'] = np.roll(df[column], i)
                df.loc[:i, f'shift_{i}'] = np.nan

        #combine shifted cells to find next closest neighbour value
        df[column_name] = df['shift_1'].fillna(df['shift_2'])
        del df['shift_1']
        del df['shift_2']
        for i in range(3, max_distance+1):
            df[column_name] = df[column_name].fillna(df[f'shift_{i}'])
            del df[f'shift_{i}']
            
    """
    Using Selene package for spike detection
    """   

    def selene_spike_detection(self, df_meas_long, adapted_meas_col_name, quality_column_name, time_column, measurement_column):
        #spiketest    
        spikedetected = True
        iter = 0
        badixs = []
        df_meas_long['quality'] = 0
        df_meas_long['new_data'] = 0
        while spikedetected and iter < self.maxiter:
            print('hereeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee')
            #drop spike values detected in previous iteration
            spikedetected = False
            for ix in range(len(df_meas_long.index.values)):
                print('here')
                if (ix < self.winsize/2):
                    ini= 0
                    end= self.winsize-1
                    winix = ix
                elif (ix > len(df_meas_long.index) - self.winsize/2):
                    ini=len(df_meas_long.index)- self.winsize
                    end=len(df_meas_long.index)-1
                    winix = (self.winsize-1)-(end-ix+1)
                else:
                    ini = int(ix - self.winsize/2)
                    end = int(ix + self.winsize/2)
                    winix = int(self.winsize/2)
                winx = df_meas_long.index.values[ini:end].astype(float)
                windata = df_meas_long[adapted_meas_col_name].values[ini:end]
                splinefit = np.polyfit(winx, windata, self.splinedegree)
                splinedata = np.polyval(splinefit,winx)
                rmse = self.rmse(splinedata, np.array(windata))
                if (abs(splinedata[winix]-df_meas_long[adapted_meas_col_name][ix]) >= self.nsigma*rmse):
                    df_meas_long.loc[df_meas_long.index.values[ix],'quality'] = 4
                    df_meas_long.loc[df_meas_long.index.values[ix],'new_data'] = splinedata[winix]
                    badixs.append(df_meas_long.index.values[ix])
                    spikedetected = True
            iter=iter+1
        
        #make Plots
        #Analyse spike detection
        #self.helper.plot_df(df_meas_long[time_column], df_meas_long[adapted_meas_col_name],'Water Level','Timestamp ','Measured water level wo outliers and spikes in 1 min timestamp (all) (improved)')
        self.helper.plot_two_df(df_meas_long[time_column][self.min_value_plotting:self.max_value_plotting], df_meas_long[adapted_meas_col_name][self.min_value_plotting:self.max_value_plotting],'Water Level', df_meas_long[quality_column_name][self.min_value_plotting:self.max_value_plotting], 'Quality flag', 'Timestamp ','SELENE: Measured water level wo outliers and spikes in 1 min timestamp (zoomed to max spike')
        #self.helper.plot_two_df_same_axis(df_meas_long[time_column][self.min_value_plotting:self.max_value_plotting], df_meas_long[adapted_meas_col_name][self.min_value_plotting:self.max_value_plotting],'Water Level', 'Water Level (corrected)', df_meas_long[measurement_column][self.min_value_plotting:self.max_value_plotting], 'Timestamp ', 'Water Level (measured)', 'Measured water level wo outliers and spikes in 1 min timestamp vs orig (zoomed to max spike) (improved)')
        self.helper.plot_two_df(df_meas_long[time_column], df_meas_long['new_data'],'Spike', df_meas_long[quality_column_name], 'Quality flag', 'Timestamp ','SELENE: Measured water level wo outliers and spikes in 1 min timestamp ')
        self.helper.plot_two_df(df_meas_long[time_column], df_meas_long[measurement_column],'Water Level', df_meas_long['quality'], 'Quality flag', 'Timestamp ','SELENE: Measured water level in 1 min timestamp incl. flags')
        
        #More plots
        for i in range(1, 41):
            min = random.randint(1, len(df_meas_long)-4000)
            max = min + 4000
            self.helper.plot_two_df(df_meas_long[time_column][min:max], df_meas_long['new_data'][min:max],'Spike', df_meas_long[measurement_column][min:max], 'Water Level (measured)', 'Timestamp ', f'SELENE: Graph{i}- Measured water level wo outliers and spikes in 1 min timestamp vs orig (zoomed to max spike) (improved)')

        return df_meas_long
    
    def rmse(self, predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean()) 
        
    """
    Using ML for spike detection
    """    

    # Feature Engineering
    def create_features(self, data):
        lagged_features = pd.concat([data.shift(i) for i in range(1, self.n_lags + 1)], axis=1)
        lagged_features.columns = [f'lag_{i}' for i in range(1, self.n_lags + 1)]
        
        # Generate rolling statistics
        rolling_mean = data.shift(1).rolling(window=self.rolling_window).mean()
        rolling_std = data.shift(1).rolling(window=self.rolling_window).std()
        
        # Concatenate all features into a single DataFrame
        X = pd.concat([lagged_features, rolling_mean, rolling_std], axis=1)
        X.columns = [f'lag_{i}' for i in range(1, self.n_lags + 1)] + ['rolling_mean', 'rolling_std']
        
        # Target variable (forecasting horizon steps ahead)
        y = data.shift(-self.forecast_horizon)

        return np.array(X), np.array(y)

    def spike_detection_ml(self, data):

        # Prepare data
        print('here')
        X, y = self.create_features(data)
        print('here1')

        # Split into train and test
        X_train, X_test = X[:-self.test_size], X[-self.test_size:]
        y_train, y_test = y[:-self.test_size], y[-self.test_size:]

        # Impute missing values using IterativeImputer
        imputer = IterativeImputer()
        X_train= imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        # Impute the target variable y_train (note: we reshape it)
        y_train = imputer.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_test = imputer.transform(y_test.reshape(-1, 1)).ravel()

        # Scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Initialize XGBoost model
        model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.2)
        print('here2')

        # Time Series Cross-Validation
        tscv = TimeSeriesSplit(n_splits=10)
        for train_index, val_index in tscv.split(X_train):
            X_t, X_val = X_train[train_index], X_train[val_index]
            y_t, y_val = y_train[train_index], y_train[val_index]
            model.fit(X_t, y_t)
            y_pred = model.predict(X_val)
            val_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            print(f"Validation RMSE: {val_rmse:.3f}")

        # Final training on full train set
        model.fit(X_train, y_train)

        # Test predictions
        y_pred_test = model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        print(f"Test RMSE: {test_rmse:.3f}")

        # Plot actual vs predicted
        plt.plot(range(len(y_test)), y_test, label='Actual')
        plt.plot(range(len(y_pred_test)), y_pred_test, label='Predicted')
        plt.xlabel('Time Steps')
        plt.ylabel('y')
        plt.legend()
        plt.title("XGBoost Time Series Forecasting")
        plt.show()

        return data