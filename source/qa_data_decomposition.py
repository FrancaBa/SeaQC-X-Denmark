#############################################################
###Python script which will be used later on for detiding ###
## Written by frb for GronSL project (2024-2025) ############
#############################################################

import numpy as np
import pandas as pd

#Packages used for decomposition of ts
import utide
from statsmodels.tsa.seasonal import MSTL
from statsmodels.tsa.seasonal import DecomposeResult

#This is a template for later on and currently not in use

class TSDecomposer():

    def tide_removal(self, df, cleaned_data_column_name):
        
        #Calculate the anomaly as it is needed as input for utide
        df['anomaly'] = df[cleaned_data_column_name] - df[cleaned_data_column_name].mean()
        
        #Interpolation needed!
        #
        coef = utide.solve(
            df.index,
            df['anomaly'].values.flatten(),
            lat=latitude_obs= dic_stations[station][0],
            method="ols",
            conf_int="MC",
            trend=False,
            nodal=True,
            verbose=False,
        )
        
        #reconstruct the tide for the relevant period
        tide = utide.reconstruct(obs.index, coef, verbose=False)

        print('hey')
    
    def multi_seasonal_decomposition(self, df):
        print('hey')
        #get weekly, seasonal and monthly variation. Are there any outliers in the periods?
