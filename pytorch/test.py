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
    import loop_single, loop_multi, dataprep
except ModuleNotFoundError:
    wd = os.getcwd()
    print("Error: please make sure that working directory is set as '~/Masterthesis'.")
    print("Current working directory is:", wd)

#Included participants
participants = ["bci10", "bci12", "bci13", "bci17", "bci21", "bci22",
                "bci23", "bci24", "bci26", "bci27", "bci28", "bci29", "bci30", 
                "bci31", "bci32", "bci33", "bci34", "bci35", "bci36", "bci37", 
                "bci38", "bci39", "bci40", "bci41", "bci42", "bci43", "bci44"]

#Modalities
modalities = ["PPG"]


#############################################################################################
##################################### 1. Obtain results #####################################
#############################################################################################
np.random.seed(3791)
torch.manual_seed(3791)
torch.cuda.manual_seed(3791)
batch_size = 10



def ObtainResults(modality, returndict):
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
    predictions_dict = {}
    labels_dict = {}

    predictions_array = []
    labels_array = []

    #Select correct hpo file
    hpo = pickle.load(open("hpo/hyper_parameters/"+modality+".pickle", "rb"))   

    if modality == "multi":
        for i in participants:

            predictions, labels = train_multi.TrainLoop(participant = i,
                            drop = drop, batch_size = batch_size, 
                            epochs = None, trainortest = "test")

            predictions_dict[i] = predictions
            labels_dict[i] = labels

            predictions_array = np.append(predictions_array, predictions.squeeze().detach().numpy())
            labels_array = np.append(labels_array, labels.squeeze().detach().numpy())    
    else:
        for i in participants:

            predictions, labels = loop_single.SingleTrainLoop(participant = i, modality = modality,
                            batch_size = batch_size, hpo = hpo,
                            epochs = None, trainortest = "test")

            predictions_dict[i] = predictions
            labels_dict[i] = labels

            predictions_array = np.append(predictions_array, predictions.squeeze().detach().numpy())
            labels_array = np.append(labels_array, labels.squeeze().detach().numpy())

    if returndict == True:
        return predictions_dict, labels_dict

    elif returndict == False:
        return predictions_array, labels_array



def Performance(modality):
    """
    Purpose:
        Calculates and print several performance statistics
    Arguments:
        modality: network variation to output results for (EEG / PPG / GSR / Multi) 
    """

    #Obtain results
    predictions, labels = ObtainResults(modality = modality, returndict = False)

    #Metric 1: Correlation predictions and labels
    corr = np.corrcoef(predictions, labels)
    print("Correlation predictions and labels:", round(float(corr[1][0]), 5))  

    #Metric 2: Mean absolute error
    np.set_printoptions(suppress=True)
    mae = sum(abs(predictions - labels)) / len(predictions)

    print("Mean asbsolute error "+modality+"_networks:", round(float(mae), 5))
    print("Mean absolute error "+modality+"_networks on the original scale:", round(float(mae*20), 5))
    
    #Metric 3: RMSE
    rmse = math.sqrt(sum((predictions - labels) **2) / len(predictions))
    print("Root mean error "+modality+"_networks:", round(float(rmse), 5))

    #Metric 4: SE
    stdev = np.std(predictions - labels)
    print("Standard deviation "+modality+"_networks:", round(float(stdev), 5))



def HistGrid():
    """
    Purpose:
        Show performance histograms and arrange in grid
    """

    #Create Grid    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
    
    #EEG plot
    #Preperation
    predictions, labels = ObtainResults(modality = "PPG", returndict = False)
    x = predictions - labels
    x = np.sort(x)
    lowerbound = x[round(len(predictions)*0.1)]
    upperbound = x[round(len(predictions)*0.9)]
    median = x[round(len(predictions)*0.5)]

    ax1.hist(x, bins = 70, color = "dodgerblue")
    ax1.set_title("Single-modular EEG-Networks") 
    ax1.vlines((lowerbound, median, upperbound), 0, 50, colors= "darkslategray", linestyles= ("dashed", "solid", "dashed"))
    ax1.set_xticklabels([-1,-0.5, 0, 0.5, 1])

    #PPG plot
    #Preperation
    predictions, labels = ObtainResults(modality = "PPG", returndict = False)
    x = predictions - labels
    x = np.sort(x)
    lowerbound = x[round(len(predictions)*0.1)]
    upperbound = x[round(len(predictions)*0.9)]
    median = x[round(len(predictions)*0.5)]

    ax2.hist(x, bins = 70, color = "dodgerblue")
    ax2.set_title("Single-modular PPG-Networks") 
    ax2.vlines((lowerbound, median, upperbound), 0, 50, colors= "darkslategray", linestyles= ("dashed", "solid", "dashed"))
    ax2.set_xticklabels([-1,-0.5, 0, 0.5, 1])


    #GSR plot
    #Preperation
    predictions, labels = ObtainResults(modality = "PPG", returndict = False)
    x = predictions - labels
    x = np.sort(x)
    lowerbound = x[round(len(predictions)*0.1)]
    upperbound = x[round(len(predictions)*0.9)]
    median = x[round(len(predictions)*0.5)]

    ax3.hist(x, bins = 70, color = "dodgerblue")
    ax3.set_title("Single-modular GSR-Networks") 
    ax3.vlines((lowerbound, median, upperbound), 0, 50, colors= "darkslategray", linestyles= ("dashed", "solid", "dashed"))
    ax3.set_xticklabels([-1,-0.5, 0, 0.5, 1])


    #Multi plot
    #Preperation
    predictions, labels = ObtainResults(modality = "PPG", returndict = False)
    x = predictions - labels
    x = np.sort(x)
    lowerbound = x[round(len(predictions)*0.1)]
    upperbound = x[round(len(predictions)*0.9)]
    median = x[round(len(predictions)*0.5)]

    ax4.hist(x, bins = 70, color = "dodgerblue")
    ax4.set_title("Multi-modular Networks") 
    ax4.vlines((lowerbound, median, upperbound), 0, 50, colors= "darkslategray", linestyles= ("dashed", "solid", "dashed"))
    ax4.set_xticklabels([-1,-0.5, 0, 0.5, 1])

    fig.suptitle('Difference between prediction and labels')
    plt.show()



def DataPlot(modality):
    """
    (Not used in current version of manuscript)
    Purpose:
        Visualize data sample
    """   

    #Read in data and select windows to plot
    dat = pickle.load(open('pipeline/prepared_data/bci13.pickle', "rb")) #Open pickle
    windows = dat[modality][50].squeeze().detach().numpy()

    if modality == "EEG":
        plt.plot(windows[:,0], label = "line1")
        plt.plot(windows[:,1], label = "line2")
        plt.plot(windows[:,2], label = "line3")
        plt.plot(windows[:,3], label = "line4")

    else:
        plt.plot(windows)
        
    plt.show()


