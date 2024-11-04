import numpy as np
import pandas as pandas
import ruptures as rpt
import matplotlib.pyplot as plt

from ruptures.detection import Pelt

class ShiftDetector():

    def detect_shifts(self, df, data_column_name, quality_column_name):
        # Detect shifts in mean using Pruned Exact Linear Time approach
        algo = Pelt(model="l2").fit(df[data_column_name].values)
        result = algo.predict(pen=1000)

        # Detect shifts in variance using Pruned Exact Linear Time approach
        #result_2 = algo.predict(pen=50)

        # Plot the results
        rpt.display(df[data_column_name], result)  # Display the signal with detected change points
        plt.show()
        #rpt.display(df[data_column_name], result_2)  # Display the signal with detected change points
        #plt.show()

        return df