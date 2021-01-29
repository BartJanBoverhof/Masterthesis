"""
@Author: Thomas de Groot
@Modified by: Bart-Jan Boverhof
@Description: Creates pandas dataframes based on the desired markers.
"""
#### Prerequisites ####
#Importing packages
import pyxdf
import pandas as pd

#Defining required events and streams
event_list = ['1', '2', '3', 'break', 'rest']
stream_types_list = ['PPG', 'GSR', 'EEG']

#Window creation related parameters
block_start_margin = 0  
block_end_margin = 0    
window_size = 50       
window_overlap = 1     
window_exact = True     



def load_data(filename):
    """
    Purpose: 
        Load the data streams from the xdf file
    Arguments:
        filename: Path of the respective .xdf file
    """
    print(  """
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
                   start_margin=0, end_margin=0, 
                   window_size=0, window_overlap=0, 
                   window_exact=False):
    """
    Purpose:
        Use the marker stream to create windows and return these in a dataframe
    Arguments:
        streams: streams loaded from the .xdf file       
        event_list: experimental events to create windows from 
        start_margin: number of seconds to cut from the start of each block
        end_margin: number of seconds to cut from the end of each block
        window_size: windows size in seconds
        window_overlap: window overlap in seconds
        window_exaxt: whether or not smaller windows then window_size are used
    Return:
        dataframe:
            ROWS:
                windows
            COLS:
                index: window index
                durations: window durations
                start: start timecode window
                task: task type (1, 2, 3, rest, break)
                tries: number of tries to cut the window
                window: window index per task (thus seperate index for each task)
    """
    result = pd.DataFrame()
    tries = {}
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
                                                'duration': window_duration}, ignore_index=True)
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
            GSR dataframe containing all GSR data + Timestamps
            PPG dataframe containing all PPG data + Timestamps                
            GSR dataframe containing all EEG data + Timestamps
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
        Cut data into specified windows
    Arguments:
        stream_dataframes: 
        windows:
    Return:
    """

    #Extract utilized modalities
    stream_labels = list(stream_dataframes.keys())
    result = {stream_labels[0]:[], stream_labels[1]:[], stream_labels[2]:[]}

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

            #Append to dict
            result[dataframe_id].append(epoch)

    return result



def transformer(filename): 
    print("filename", filename)
    streams = load_data(filename)
    print('streams', len(streams))
    windows = create_windows(streams, event_list, block_start_margin, block_end_margin, window_size, window_overlap, window_exact)
    print('windows', windows)
    stream_dataframes = create_dataframes(streams, stream_types_list) 
    print('data', stream_dataframes)
    cutted_data = cut_epochs(stream_dataframes, windows)
    return cutted_data