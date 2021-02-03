"""
@Author: Thomas de Groot
@Modified by: Bart-Jan Boverhof
@Description: Creates pandas dataframes based on the desired markers.
"""
#### Prerequisites ####
#Importing packages
import pyxdf
import pandas as pd
import torch
 


def load_data(filename):
    """
    Purpose: 
        Load the data streams from the xdf file
    Arguments:
        filename: Path of the respective .xdf file
    """

    print("""
   transforming...
         __
 _(\    |@@|
(__/\__ \--/ __
   \___|----|  |   __
       \ }{ /\ )_ / _
       /\__/\ \__O (__
      (--/\--)    \__/
      _)(  )(_
     `---''---`     
    """)

    streams, _ = pyxdf.load_xdf(filename)


    return streams



def create_windows(streams, events, 
                   start_margin, end_margin, 
                   window_size, window_overlap, window_exact, 
                   labels, role):
    """
    Purpose:
        Use the marker stream to create windows and summarize these in a pandas df
        Additionally, within this function the labels are appended to this df
    Arguments:
        streams:        streams loaded from the .xdf file       
        events:         experimental events to create windows from 
        start_margin:   number of seconds to cut from the start of each block
        end_margin:     number of seconds to cut from the end of each block
        window_size:    windows size in seconds
        window_overlap: window overlap in seconds
        window_exaxt:   whether or not smaller windows then window_size are used
        labels:         labels df object 
        role:           experimental role of the data type (Engineer /  Operations / Tactical)
    Return:
        dataframe containing:
            ROWS:
                all cutted windows
            COLS:
                index: window index
                durations: window durations
                start: start timecode window
                task: task difficulty (1, 2, 3,)
                tries: number of tries to cut the window
                window: window index per task (thus seperate index for each task)
                workload: workload assessment lable
    """

    #Windows result object 
    result = pd.DataFrame()
    tries = {}

    #Create labels df 
    pfilter = labels["Username"] == "bci10"
    labels = labels[pfilter]
    rolefilter = labels["Role"] == role
    labels = labels[rolefilter]
    counter = 0 #Counter for attributing labels to windows df 
    
    for stream in streams: #Iterate over the 18 streams
        stream_type = stream['info']['type'][0] #Determine the type of the stream
        if stream_type == 'GameEvents' and stream["time_stamps"].size !=0 : #We create the windows only based on the event stream & if it is not empty (some event streams are double)!
            data = stream['time_series'] #Data object: contains the events 
            timestamps = stream['time_stamps'] #Timestamps object: contains all time stamps 

            for index, event in enumerate(data): #For all of the selected events (844)
                #Only for events that we are interested in
                if event[0] in events: #If we have an event that is in the list events
                    block_name = event[0] #We make a block
                    block_start = timestamps[index] + start_margin #Calculate start time block
                    block_end = timestamps[index + 1] - end_margin #Calculate end time block

                    #Check if this task has been done before and count tries
                    if block_name not in tries:
                        tries[block_name] = 0
                    tries[block_name] += 1

                    # Check window_size, if 0 windows will be the entire block. Otherwise the previously defined size
                    if window_size == 0:
                        window_length = block_end - block_start
                    else:
                        window_length = window_size

                    window_id = 0 #Index the windows by a number
                    
                    #Select correct label col
                    counter = counter+1 #Counter +1
                    foofilter = labels["taskN"] == counter #Create filter
                    foo = labels[foofilter] #Select correct col

                    next_window_start_ts = block_start 
                    
                    #While loop for defining the consecutive window
                    while next_window_start_ts < block_end:
                        window_start_ts = next_window_start_ts
                        window_end_ts = window_start_ts + window_length
                        if not window_end_ts <= block_end:
                            window_end_ts = block_end
                        window_duration = window_end_ts - window_start_ts

                        #Check if the actual duration of the window is equal to the pre specified size
                        if window_duration < window_length: 
                            if window_exact:
                                print('window not exact size')
                                break
                            if (window_duration / window_length) < 0.9:
                                print('window smaller then 90% of wanted size')
                                break

                        result = result.append({'task': block_name,
                                                'tries': tries[block_name],
                                                'window': window_id,
                                                'start': window_start_ts,
                                                'stop': window_end_ts,
                                                'duration': window_duration,
                                                'workload': int(foo["Q01_Mental demand->low|high"])},                                
                                                 ignore_index=True)
                        window_id += 1
                        next_window_start_ts = window_end_ts - window_overlap

                        

            break
    return result



def create_dataframes(streams, stream_types):
    """
    Purpose:
        Put all data into pandas dataframe format
    Arguments:
        streams: the xdf formatted data        
        stream type: a list of the required streams
    Return:
        Dictionary containing:
            GSR dataframe containing all GSR data 
            PPG dataframe containing all PPG data               
            GSR dataframe containing all GSR data
    """

    result = {} #Create dict to store results in
    for stream in streams: #Iterate over the 18 streams
        stream_name = stream['info']['name'][0] #Obtain the stream name for this respective stream
        stream_type = stream['info']['type'][0] #Obtain the stream type for this respective stream
        if stream_type in stream_types: #Only select streams of interest (in the stream_types object)
            stream_desc = stream['info']['desc'][0]
            timestamps = stream['time_stamps'] #Obtain all timestamps
            data = stream['time_series'] #Obtain all data points (equal in length to time_stamps)

            #Create stream labels on basis of the stream description
            labels = None
            if stream_desc is not None:
                labels = []
                channels = stream_desc['channels']
                for channel in channels:
                    channel_labels = channel['channel']
                    for channel_info in channel_labels:
                        channel_label = channel_info['label'][0]
                        labels.append(channel_label)

            stream_data = pd.DataFrame(data, columns=labels) #Create a seperate dataframe with the data for each stream
            stream_data.insert(0, "timestamps", timestamps, True) #Add column with timestamps to data
            result[stream_type] = stream_data
    return result



def cut_epochs(stream_dataframes, windows):
    """
    Purpose:
        Cut data the data into epochs of the desired sizes
    Arguments:
        stream_dataframes: dictionary containing all data in dataframe format (resulting from create_dataframe function)
        windows: dataframe containing desired epoch sizes (resulting from the create_windows function)
    Return:
        Dictionary containing:
            GSR:    *number of epochs* PyTorch tensors containing GSR data. 
            EEG:    *number of epochs* PyTorch tensors containing EEG data. 
            PPG:    *number of epochs* PyTorch tensors containing PPG data. 
            labels: dataframe listing all epochs and their label.
    """

    #Extract utilized modalities
    stream_labels = list(stream_dataframes.keys())

    #Create dict to store values AND append labels df to dict
    result = {stream_labels[0]:[], stream_labels[1]:[], stream_labels[2]:[], "labels":windows}

    #Create a dataset for every stream in every window
    for window in windows.itertuples(): #Iterate over all windows defined in the windows dataframe
        print('--- window', window.task, window.tries)
        start_ts = window.start
        stop_ts = window.stop
        for dataframe_id in stream_dataframes: #Iterate over the three streams
            dataframe = stream_dataframes[dataframe_id]
            
            #Select only relevant timestamps
            after_start = dataframe['timestamps'] >= start_ts
            before_end = dataframe['timestamps'] < stop_ts
            epoch = dataframe[after_start & before_end]
            epoch = epoch.drop("timestamps", axis=1) #Drop timestamps col
            
            #Transfer to PyTorch tensor
            epoch.tensor = torch.FloatTensor(epoch.values) #Transfer to tensor and store only data
            result[dataframe_id].append(epoch.tensor) #Append to dict

    #Cutting stored epochs into same size
    lengths = []
    for i in result["PPG"]:
        x = len(i)
        lengths.append(x)

    lowest = min(lengths)

    count = 0
    for  i in result["PPG"]:
        if len(i) != lowest:
            result["PPG"][count] = i.narrow(0,0,1279)
        count +=1

    return result



def transformer(filename, event_list, 
                block_start_margin, block_end_margin, 
                window_size, window_overlap, 
                window_exact, stream_types_list, 
                labels, role): 
    """
    Purpose:
        Function combining and running all previously defined functions 
    Arguments:
        filename:               filename (and path) of the data object to be transformed 
        event_list:             experimental events to create windows out of 
        block_start_margin:     number of seconds to cut from the start of each block
        block_end_margin:       number of seconds to cut from the end of each block
        window_size:            windows size in seconds
        window_overlap:         window overlap in seconds
        window_exaxt:           whether or not smaller windows then window_size are used
        labels:                 labels df object 
        role:                   experimental role of the data type (Engineer /  Operations / Tactical)
    Return:
        Dictionary containing:
            GSR:    *number of epochs* PyTorch tensors containing GSR data. 
            EEG:    *number of epochs* PyTorch tensors containing EEG data. 
            PPG:    *number of epochs* PyTorch tensors containing PPG data. 
            labels: dataframe listing all epochs and their label.
    """
    
    print("filename", filename)
    streams = load_data(filename = filename)
    print('streams', len(streams))
    windows = create_windows(streams = streams, events = event_list, 
                            start_margin = block_start_margin, end_margin = block_end_margin, 
                            window_size = window_size, window_overlap = window_overlap, 
                            window_exact = window_exact, labels = labels, role = role)
    print('windows', windows)
    stream_dataframes = create_dataframes(streams = streams, stream_types = stream_types_list) 
    print('data', stream_dataframes)
    cutted_data = cut_epochs(stream_dataframes, windows)
    
    return cutted_data


