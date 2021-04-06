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
import pandas as pd


def Performance(modality):
    """
    Purpose:
        Calculates and print several performance statistics
    Arguments:
        modality: network variation to output results for (EEG / PPG / GSR / Multi) 
    """

    print("-----------------------------------------------------------------")
    print("-------------------- Results "+modality+"------------------------")
    print("-----------------------------------------------------------------")


    #Open results
    results = pickle.load(open("results/model_performance/predictions/"+modality+".pickle", "rb"))  
    
    predictions = results["predictions"]
    labels = results["labels"]

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
    #Open results
    results = pickle.load(open("results/model_performance/predictions/EEG.pickle", "rb"))  
    
    predictions = results["predictions"]
    labels = results["labels"]
    
    #Plot prerequisites 
    x = predictions - labels
    x = np.sort(x)
    lowerbound = x[round(len(predictions)*0.1)]
    upperbound = x[round(len(predictions)*0.9)]
    median = x[round(len(predictions)*0.5)]

    #Plot
    ax1.hist(x, bins = 70, color = "dodgerblue")
    ax1.set_title("EEG-Networks", fontsize = 10, fontstyle = "italic") 
    ax1.vlines((lowerbound, median, upperbound), 0, 71, colors= "darkslategray", linestyles= ("dashed", "solid", "dashed"))
    ax1.set(ylabel='Count')

    #PPG plot
    #Open results
    results = pickle.load(open("results/model_performance/predictions/PPG.pickle", "rb"))  
    
    predictions = results["predictions"]
    labels = results["labels"]

    #Plot prerequisites 
    x = predictions - labels
    x = np.sort(x)
    lowerbound = x[round(len(predictions)*0.1)]
    upperbound = x[round(len(predictions)*0.9)]
    median = x[round(len(predictions)*0.5)]

    #Plot
    ax2.hist(x, bins = 70, color = "dodgerblue")
    ax2.set_title("PPG-Networks", fontsize = 10, fontstyle = "italic") 
    ax2.vlines((lowerbound, median, upperbound), 0, 68, colors= "darkslategray", linestyles= ("dashed", "solid", "dashed"))


    #GSR plot
    #Open results
    results = pickle.load(open("results/model_performance/predictions/GSR.pickle", "rb"))  
    
    predictions = results["predictions"]
    labels = results["labels"]

    #Plot prerequisites 
    x = predictions - labels
    x = np.sort(x)
    lowerbound = x[round(len(predictions)*0.1)]
    upperbound = x[round(len(predictions)*0.9)]
    median = x[round(len(predictions)*0.5)]

    #Plot
    ax3.hist(x, bins = 70, color = "dodgerblue")
    ax3.set_title("GSR-Networks", fontsize = 10, fontstyle = "italic") 
    ax3.vlines((lowerbound, median, upperbound), 0, 69, colors= "darkslategray", linestyles= ("dashed", "solid", "dashed"))
    ax3.set(ylabel='Count', xlabel = 'MAE')


    #Multi plot
    #Open results
    results = pickle.load(open("results/model_performance/predictions/multi.pickle", "rb"))  
    
    predictions = results["predictions"]
    labels = results["labels"]

    #Plot prerequisites 
    x = predictions - labels
    x = np.sort(x)
    lowerbound = x[round(len(predictions)*0.1)]
    upperbound = x[round(len(predictions)*0.9)]
    median = x[round(len(predictions)*0.5)]

    #Plot
    ax4.hist(x, bins = 70, color = "dodgerblue")
    ax4.set_title("Multi-Modular Networks", fontsize = 10, fontstyle = "italic") 
    ax4.vlines((lowerbound, median, upperbound), 0, 51, colors= "darkslategray", linestyles= ("dashed", "solid", "dashed"))
    ax4.set(xlabel='MAE')
    
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



def Scatter():

    #Create Grid    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)

    #EEG
    #Open results
    results = pickle.load(open("results/model_performance/predictions/EEG.pickle", "rb"))  

    df = pd.DataFrame(data = [(results["predictions"]), (results["labels"])])
    df = df.transpose()
    df = df.rename(columns={0:"predictions", 1:"labels"})
    df.insert(2, "difference", abs(df["predictions"]- df["labels"]))

    corr = np.corrcoef(df["labels"], df["difference"])
    text = "r = "+str(round(corr[0][1],2))
    m, b = np.polyfit(df["predictions"], df["difference"], 1)


    ax1.scatter(df["labels"], df["difference"], s = 8, c = df["labels"], cmap = "winter")
    ax1.plot(df["labels"], m*df["labels"] + b, color = "red", linewidth = 3)
    ax1.text(0.5, 0.6, text, horizontalalignment='center', verticalalignment='center', 
            color = "black", fontstyle = "italic", fontsize = 22,
            bbox=dict(boxstyle="round", alpha = 0.68,
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8),
                   ))
    ax1.set(ylabel = "MAE")
    ax1.set_title("EEG-Networks", fontsize = 10, fontstyle = "italic")


    #PPG
    #Open results
    results = pickle.load(open("results/model_performance/predictions/PPG.pickle", "rb"))  

    df = pd.DataFrame(data = [(results["predictions"]), (results["labels"])])
    df = df.transpose()
    df = df.rename(columns={0:"predictions", 1:"labels"})
    df.insert(2, "difference", abs(df["predictions"]- df["labels"]))

    corr = np.corrcoef(df["labels"], df["difference"])
    text = "r = "+str(round(corr[0][1],2))
    m, b = np.polyfit(df["predictions"], df["difference"], 1)


    ax2.scatter(df["labels"], df["difference"], s = 8, c = df["labels"], cmap = "winter")
    ax2.plot(df["labels"], m*df["labels"] + b, color = "red", linewidth = 3)
    ax2.text(0.5, 0.6, text, horizontalalignment='center', verticalalignment='center', 
            color = "black", fontstyle = "italic", fontsize = 22,
            bbox=dict(boxstyle="round", alpha = 0.68,
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8),
                   ))
    ax2.set_title("PPG-Networks", fontsize = 10, fontstyle = "italic") 


    #GSR
    #Open results
    results = pickle.load(open("results/model_performance/predictions/GSR.pickle", "rb"))  

    df = pd.DataFrame(data = [(results["predictions"]), (results["labels"])])
    df = df.transpose()
    df = df.rename(columns={0:"predictions", 1:"labels"})
    df.insert(2, "difference", abs(df["predictions"]- df["labels"]))

    corr = np.corrcoef(df["labels"], df["difference"])
    text = "r = "+str(round(corr[0][1],2))
    m, b = np.polyfit(df["predictions"], df["difference"], 1)


    ax3.scatter(df["labels"], df["difference"], s = 8, c = df["labels"], cmap = "winter")
    ax3.plot(df["labels"], m*df["labels"] + b, color = "red", linewidth = 3)
    ax3.text(0.5, 0.6, text, horizontalalignment='center', verticalalignment='center', 
            color = "black", fontstyle = "italic", fontsize = 22,
            bbox=dict(boxstyle="round", alpha = 0.68,
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8),
                   ))
    ax3.set(ylabel = "MAE", xlabel = "Label")
    ax3.set_title("GSR-Networks", fontsize = 10, fontstyle = "italic") 

    #EEG
    #Open results
    results = pickle.load(open("results/model_performance/predictions/multi.pickle", "rb"))  

    df = pd.DataFrame(data = [(results["predictions"]), (results["labels"])])
    df = df.transpose()
    df = df.rename(columns={0:"predictions", 1:"labels"})
    df.insert(2, "difference", abs(df["predictions"]- df["labels"]))

    corr = np.corrcoef(df["labels"], df["difference"])
    text = "r = "+str(round(corr[0][1],2))
    m, b = np.polyfit(df["predictions"], df["difference"], 1)


    ax4.scatter(df["labels"], df["difference"], s = 8, c = df["labels"], cmap = "winter")
    ax4.plot(df["labels"], m*df["labels"] + b, color = "red", linewidth = 3)
    ax4.text(0.5, 0.6, text, horizontalalignment='center', verticalalignment='center', 
            color = "black", fontstyle = "italic", fontsize = 22,
            bbox=dict(boxstyle="round", alpha = 0.68,
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8),
                   ))
    ax4.set(xlabel = "Label")
    ax4.set_title("Multi-Modular Networks", fontsize = 10, fontstyle = "italic") 


    plt.show()






#Included participants
participants = ["bci10", "bci12", "bci13", "bci17", "bci21", "bci22",
                "bci23", "bci24", "bci26", "bci27", "bci28", "bci29", "bci30", 
                "bci31", "bci32", "bci33", "bci34", "bci35", "bci36", "bci37", 
                "bci38", "bci39", "bci40", "bci41", "bci42", "bci43", "bci44"]

#Modalities
modalities = ["PPG", "GSR", "EEG", "multi"]


for modality in modalities:
    Performance(modality)

#HistGrid()
Scatter()
