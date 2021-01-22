"""
@Author: Bart-Jan Boverhof
@Last Modified by: Bart-Jan Boverhof
@Description: Creates .csv files based on the desired markers.
"""



################### 0. Prerequisites ###################
import pyxdf
import numpy as np
import matplotlib.pyplot as plt

#Load xdf file
data, header = pyxdf.load_xdf('5.data/bc10/bci10 operations.xdf')

print(data)

for stream in data:
    y = stream['time_series']

    if isinstance(y, list):
        # list of strings, draw one vertical line for each marker
        for timestamp, marker in zip(stream['time_stamps'], y):
            plt.axvline(x=timestamp)
            print(f'Marker "{marker[0]}" @ {timestamp:.2f}s')
    elif isinstance(y, np.ndarray):
        # numeric data, draw as lines
        plt.plot(stream['time_stamps'], y)
    else:
        raise RuntimeError('Unknown stream format')

plt.show()





'''
def xdf2csv(x, output_location,
            epoch_length = 10, 
            data_col = "time_series", time_col = "time_stamps"):
    """
    FUNCTION DESCRIPTION:
        The purpose of this function is to read in the .xdf files, and return csv files. 

    FUNCTION ARGUMENTS:
        x             The dictionary obtained from the "load_xdf" function.
        epoc_lenght   The length of the selected data window in seconds.
        data_col      Name of the column from which to select data. Current default is "time series".
        time_col      Name of the column to base the selected data window on. 
    """
'''
