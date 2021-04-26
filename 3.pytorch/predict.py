"""
@Author: Bart-Jan Boverhof
@Last Modified by: Bart-Jan Boverhof
@Description: This file contains the code from which all networks networks are tested.
"""
#############################################################################################
######################################### Prerequisites #####################################
#############################################################################################
#Importing packages
import numpy as np
import torch
import os
import pickle
import math
import matplotlib.pyplot as plt

#Import local files
try:
    import loop_single, loop_multi, utility
except ModuleNotFoundError:
    wd = os.getcwd()
    print("Error: please make sure that working directory is set as '~/Masterthesis'.")
    print("Current working directory is:", wd)


#############################################################################################
##################################### 1. Obtain results #####################################
#############################################################################################
def ObtainResults(modality):
    """
    Purpose:
        Uses test windows to calculate predictions of trained networks.
    Arguments:
        modality: network variation to output results for (EEG / PPG / GSR / Multi) 
        returndict: whether to return the results as dictionary and hereby retain person specific information (True)
                    or returns results as single numpy arrays (False)
    Return:
        If returndict == True: Returns a dict of predictions and a dict of labels wherein keys correspond to participants
        If returndict == False: Returns a np array of predictions and a np array of labels
    """

    #Objects for storing values
    predictions_array = []
    labels_array = []
    results = {}
    #Select correct hpo file
    hpo = pickle.load(open("hpo/hyper_parameters/"+modality+".pickle", "rb"))   
                              
    if modality == "multi":
        for i in participants:

            predictions, labels = loop_multi.MultiTrainLoop(participant = i,
                            batch_size = batch_size, hpo = hpo,
                            epochs = None, trainortest = "test")


            predictions_array = np.append(predictions_array, predictions.squeeze().detach().numpy())
            labels_array = np.append(labels_array, labels.squeeze().detach().numpy())  
            results["predictions"] = predictions_array
            results["labels"] = labels_array

    else:
        for i in participants:

            predictions, labels = loop_single.SingleTrainLoop(participant = i, modality = modality,
                            batch_size = batch_size, hpo = hpo,
                            epochs = None, trainortest = "test")

            predictions_array = np.append(predictions_array, predictions.squeeze().detach().numpy())
            labels_array = np.append(labels_array, labels.squeeze().detach().numpy())
            
            results["predictions"] = predictions_array
            results["labels"] = labels_array

    with open("results/model_performance/predictions/"+modality+".pickle", 'wb') as handle: #Save as pickle
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)



np.random.seed(3791)
torch.manual_seed(3791)
torch.cuda.manual_seed(3791)
batch_size = 10

#Included participants
participants = ["bci10", "bci12", "bci13", "bci17", "bci21", "bci22",
                "bci23", "bci24", "bci26", "bci27", "bci28", "bci29", "bci30", 
                "bci31", "bci32", "bci33", "bci34", "bci35", "bci36", "bci37", 
                "bci38", "bci39", "bci40", "bci41", "bci42", "bci43", "bci44"]

#Modalities
modalities = ["PPG", "GSR", "EEG", "multi"]

for modality in modalities:
    ObtainResults(modality)