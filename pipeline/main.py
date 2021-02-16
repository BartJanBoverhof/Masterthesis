#!/usr/bin/env python
"""
@Author: Bart-Jan Boverhof
@Description: Central script with which data is read, windows are created and windows are constructed.
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
import torch

#Setting working directory 
os.chdir("pipeline")

try: #Importing from subfile
    from transformer import transformer
except ModuleNotFoundError:
    wd = os.getcwd()
    print("Error: please make sure that working directory is set as '~/Masterthesis/pipeline'")
    print("Current working directory is:", wd)


#Opening labels file
labels = pd.read_csv("raw_data/labels.csv", delimiter = ",")    


###########################################################################
############################## PRELIMINARIES ##############################
###########################################################################
#Defining required events and streams
event_list = ['1', '2', '3']
stream_types_list = ['PPG', 'GSR', 'EEG']

#Window creation related parameters
block_start_margin = 0  
block_end_margin = 0    
window_size = 8  
window_overlap = 0     
window_exact = True     

#Defining list of all included participants
"""
participants = ["bci10", "bci12", "bci13", "bci17", "bci20", "bci21", "bci22",
                "bci23", "bci24", "bci26", "bci27", "bci28", "bci29", "bci30", 
                "bci31", "bci32", "bci33", "bci34", "bci35", "bci36", "bci37", "bci38", 
                "bci39", "bci40", "bci41", "bci42", "bci43", "bci44"]
"""
participants = ["bci10", "bci12"]

###########################################################################
############################ RUN FOR ALL PERSONS ##########################
###########################################################################
for participant in participants: 
    if __name__ == '__main__': 
        
        datapath = "raw_data/"+participant #Obtaining person-specific path
        os.mkdir("prepared_data/"+participant) #Creating person-specific subdirectory

        operations = transformer(filename = datapath+"/operations.xdf", #First segment
                            event_list = event_list, block_start_margin = block_start_margin, 
                            block_end_margin = block_end_margin, window_size = window_size, 
                            window_overlap = window_overlap, window_exact = window_exact, 
                            stream_types_list = stream_types_list, labels = labels, 
                            role = "Operations") 

        engineering = transformer(filename = datapath+"/engineering.xdf", #Second segment
                            event_list = event_list, block_start_margin = block_start_margin, 
                            block_end_margin = block_end_margin, window_size = window_size, 
                            window_overlap = window_overlap, window_exact = window_exact, 
                            stream_types_list = stream_types_list, labels = labels, 
                            role = "Engineer")

        tactical = transformer(filename = datapath+"/tactical.xdf", #Third segment
                            event_list = event_list, block_start_margin = block_start_margin, 
                            block_end_margin = block_end_margin, window_size = window_size, 
                            window_overlap = window_overlap, window_exact = window_exact, 
                            stream_types_list = stream_types_list, labels = labels, 
                            role = "Tactical")

        #Combining all segments into one dict
        combined_labels = torch.cat((operations["labels"], engineering["labels"], tactical["labels"]),0)
        combined_dat = {"EEG":[], "PPG":[], "GSR":[], "labels":combined_labels}
        
        #Appending EEG windows
        combined_dat["EEG"].extend(operations["EEG"])
        combined_dat["EEG"].extend(engineering["EEG"])
        combined_dat["EEG"].extend(tactical["EEG"])
        
        #Appending PPG windows
        combined_dat["PPG"].extend(operations["PPG"])
        combined_dat["PPG"].extend(engineering["PPG"])
        combined_dat["PPG"].extend(tactical["PPG"])

        #Appending GSR windows
        combined_dat["GSR"].extend(operations["GSR"])
        combined_dat["GSR"].extend(engineering["GSR"])
        combined_dat["GSR"].extend(tactical["GSR"])

        """
        #Cutting stored windows into same size
        eeglengths = [] 
        ppglengths = [] 
        gsrlengths = [] 
        
        #Determining lowest length tensors for each modality
        for i in combined_dat["EEG"]: #EEG
            x = i.shape[1]
            eeglengths.append(x)

        for i in combined_dat["PPG"]: #PPG
            x = i.shape[1]
            ppglengths.append(x)

        for i in combined_dat["GSR"]: #GSR
            x = i.shape[1]
            gsrlengths.append(x)
        

        #TEMP
        from collections import Counter
        eegcount = Counter(eeglengths)
        ppgcount = Counter(ppglengths)
        gsrcount = Counter(gsrlengths)
        counters["eeg"].append(eegcount)
        counters["ppg"].append(ppgcount)
        counters["gsr"].append(gsrcount)

        counters["ppg"][10]
        gsrlengths == ppglengths
        """

        with open("prepared_data/"+participant+"/data.pickle", 'wb') as handle: #Save as pickle
            pickle.dump(combined_dat, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(participant+" done!")
        