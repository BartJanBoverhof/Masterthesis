#!/usr/bin/env python
"""
@Author: Bart-Jan Boverhof
@Description: Central script with which data is read, windows are created and epochs are constructed.
"""
###########################################################################
############################## PREREQUISITES ##############################
###########################################################################
#Importing packages
import pyxdf
import pandas as pd
import pickle
import os
import csv

#Setting WD 
os.chdir("pipeline")

try: #Importing from subfile
    from transformer import transformer
except ModuleNotFoundError:
    wd = os.getcwd()
    print("Please make sure that working directory is set as '../Masterthesis/pipeline'")
    print("Current working directory is:", wd)


#Opening labels file
labels = pd.read_csv("data/labels.csv", delimiter = ",")    


###########################################################################
############################## PRELIMINARIES ##############################
###########################################################################
#Defining required events and streams
event_list = ['1', '2', '3']
stream_types_list = ['PPG', 'GSR', 'EEG']

#Window creation related parameters
block_start_margin = 0  
block_end_margin = 0    
window_size = 5  
window_overlap = 1     
window_exact = False     



###########################################################################
################################### RUN ###################################
###########################################################################
if __name__ == '__main__': 

    seg1 = transformer(filename = "data/bc10/bci10 operations.xdf", #First segment
                        event_list = event_list, block_start_margin = block_start_margin, 
                        block_end_margin = block_end_margin, window_size = window_size, 
                        window_overlap = window_overlap, window_exact = window_exact, 
                        stream_types_list = stream_types_list, labels = labels, 
                        role = "Operations") 

    with open('prepared_data/10_1.pickle', 'wb') as handle: #Save as pickle
        pickle.dump(seg1, handle, protocol=pickle.HIGHEST_PROTOCOL)

    seg2 = transformer(filename = "data/bc10/bci 10 engineering+.xdf", #Second segment
                        event_list = event_list, block_start_margin = block_start_margin, 
                        block_end_margin = block_end_margin, window_size = window_size, 
                        window_overlap = window_overlap, window_exact = window_exact, 
                        stream_types_list = stream_types_list, labels = labels, 
                        role = "Engineer")

    with open('prepared_data/10_2.pickle', 'wb') as handle: #Save as pickle
        pickle.dump(seg2, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #Third segment
    seg3 = transformer(filename = "data/bc10/bci10 tactical.xdf", #Second segment
                        event_list = event_list, block_start_margin = block_start_margin, 
                        block_end_margin = block_end_margin, window_size = window_size, 
                        window_overlap = window_overlap, window_exact = window_exact, 
                        stream_types_list = stream_types_list, labels = labels, 
                        role = "Tactical")
        
    with open('prepared_data/10_3.pickle', 'wb') as handle: #Save as pickle
        pickle.dump(seg3, handle, protocol=pickle.HIGHEST_PROTOCOL)