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
import scipy
from scipy.stats import t
import csv

def Descriptives():
    """
    Purpose:
        Calculates and print several demographic descriptive information
    """    
    
    #Open data
    with open("4.results/descriptives/Demographics.csv") as csvfile:
        data = pd.read_csv(csvfile)

        #Remove participants that were not included in the final analysis
        participants = ["bci 10", "bci 12", "bci 13", "bci 17", "bci 21", "bci 22",
                "bci 23", "bci 24", "bci 26", "bci 27", "bci 28", "bci 29", "bci 30", 
                "bci 31", "bci 32", "bci 33", "bci 34", "bci 35", "bci 36", "bci 37", 
                "bci 38", "bci 39", "bci 40", "bci 41", "bci 42", "bci 43", "bci 44"]
        participants = pd.DataFrame(participants)
        participants = participants.rename(columns = {0:"User full name"})
        data = data[data["User full name"].isin(participants["User full name"])]

        #Print desriptives
        print("---------------------------------------------------")
        print("The total number of cases is N = "+str(len(data)))
        print("---------------------------------------------------")
        print("Number of Females = "+str(data["What is your gender?"].value_counts()["Female"]))
        print("Number of Males = "+str(data["What is your gender?"].value_counts()["Male"]))
        print("---------------------------------------------------")
        print("Mean Age = "+str(round(sum(data['What is your age?<span class="boundaries"></span>']) / len(data))))
        print("Standard Deviation Age = "+str(round(np.std(data['What is your age?<span class="boundaries"></span>']),1)))

def Performance(modality, data):
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
    if data == "test":
        results = pickle.load(open("4.results/predictions/"+modality+".pickle", "rb"))  
    elif data == "train":
        results = pickle.load(open("4.results/predictions_traindat/"+modality+".pickle", "rb"))  

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

    #Metric 4: sd
    stdev = np.std(predictions - labels)
    print("Standard deviation 0-1 "+modality+"_networks:", round(float(stdev), 5))
    print("Standard deviation 0-20 "+modality+"_networks:", round(float(stdev*20), 5))


def HistGrid():
    """
    Purpose:
        Show performance histograms and arrange in grid
    """

    #Create Grid    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
    
    #EEG plot
    #Open results
    results = pickle.load(open("4.results/predictions/EEG.pickle", "rb"))  
    
    predictions = results["predictions"]
    labels = results["labels"]
    
    #Plot prerequisites 
    x = predictions - labels
    x = np.sort(x)
    lowerbound = x[round(len(predictions)*0.1)]
    upperbound = x[round(len(predictions)*0.9)]
    median = x[round(len(predictions)*0.5)]

    #Plot
    ax1.hist(x, bins = 70, color = "#0092cb")
    ax1.set_title("EEG Architecture", fontsize = 10, fontstyle = "italic") 
    ax1.vlines((lowerbound, median, upperbound), 0, 71, colors= "darkslategray", linestyles= ("dashed", "solid", "dashed"))
    ax1.text(lowerbound-0.05, 60, round(lowerbound,2), horizontalalignment='right', verticalalignment='center', 
            color = "black", fontstyle = "italic", fontsize = 10)
    ax1.text(upperbound+0.05, 60, round(upperbound,2), horizontalalignment='left', verticalalignment='center', 
            color = "black", fontstyle = "italic", fontsize = 10)
    ax1.set(ylabel='Count')

    #PPG plot
    #Open results
    results = pickle.load(open("4.results/predictions/PPG.pickle", "rb"))  
    
    predictions = results["predictions"]
    labels = results["labels"]

    #Plot prerequisites 
    x = predictions - labels
    x = np.sort(x)
    lowerbound = x[round(len(predictions)*0.1)]
    upperbound = x[round(len(predictions)*0.9)]
    median = x[round(len(predictions)*0.5)]

    #Plot
    ax2.hist(x, bins = 70, color = "#0092cb")
    ax2.set_title("PPG Architecture", fontsize = 10, fontstyle = "italic") 
    ax2.vlines((lowerbound, median, upperbound), 0, 68, colors= "darkslategray", linestyles= ("dashed", "solid", "dashed"))
    ax2.text(lowerbound-0.05, 60, round(lowerbound,2), horizontalalignment='right', verticalalignment='center', 
            color = "black", fontstyle = "italic", fontsize = 10)
    ax2.text(upperbound+0.05, 60, round(upperbound,2), horizontalalignment='left', verticalalignment='center', 
            color = "black", fontstyle = "italic", fontsize = 10)

    #GSR plot
    #Open results
    results = pickle.load(open("4.results/predictions/GSR.pickle", "rb"))  
    
    predictions = results["predictions"]
    labels = results["labels"]

    #Plot prerequisites 
    x = predictions - labels
    x = np.sort(x)
    lowerbound = x[round(len(predictions)*0.1)]
    upperbound = x[round(len(predictions)*0.9)]
    median = x[round(len(predictions)*0.5)]

    #Plot
    ax3.hist(x, bins = 70, color = "#0092cb")
    ax3.set_title("GSR Architecture", fontsize = 10, fontstyle = "italic") 
    ax3.vlines((lowerbound, median, upperbound), 0, 69, colors= "darkslategray", linestyles= ("dashed", "solid", "dashed"))
    ax3.set(ylabel='Count', xlabel = 'Prediction Error')
    ax3.text(lowerbound-0.05, 60, round(lowerbound,2), horizontalalignment='right', verticalalignment='center', 
            color = "black", fontstyle = "italic", fontsize = 10)
    ax3.text(upperbound+0.05, 60, round(upperbound,2), horizontalalignment='left', verticalalignment='center', 
            color = "black", fontstyle = "italic", fontsize = 10)

    #Multi plot
    #Open results
    results = pickle.load(open("4.results/predictions/multi.pickle", "rb"))  
    
    predictions = results["predictions"]
    labels = results["labels"]

    #Plot prerequisites 
    x = predictions - labels
    x = np.sort(x)
    lowerbound = x[round(len(predictions)*0.1)]
    upperbound = x[round(len(predictions)*0.9)]
    median = x[round(len(predictions)*0.5)]

    #Plot
    ax4.hist(x, bins = 70, color = "#0092cb")
    ax4.set_title("Multimodal Architecture", fontsize = 10, fontstyle = "italic") 
    ax4.vlines((lowerbound, median, upperbound), 0, 51, colors= "darkslategray", linestyles= ("dashed", "solid", "dashed"))
    ax4.set(xlabel='Prediction Error')
    ax4.text(lowerbound-0.05, 36, round(lowerbound,2), horizontalalignment='right', verticalalignment='center', 
            color = "black", fontstyle = "italic", fontsize = 10)
    ax4.text(upperbound+0.05, 36, round(upperbound,2), horizontalalignment='left', verticalalignment='center', 
            color = "black", fontstyle = "italic", fontsize = 10)
   
    plt.show()



def DataPlot(modality):
    """
    (Not used in current version of manuscript)
    Purpose:
        Visualize data sample
    """   

    #Read in data and select windows to plot
    dat = pickle.load(open('1.pipeline/prepared_data/bci13.pickle', "rb")) #Open pickle
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
    results = pickle.load(open("4.results/predictions/EEG.pickle", "rb"))  

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
            color = "black", fontstyle = "italic", fontsize = 16,
            bbox=dict(boxstyle="round", alpha = 0.68,
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8),
                   ))
    ax1.set(ylabel = "Absolute Prediction Error")
    ax1.set_title("EEG Architecture", fontsize = 10, fontstyle = "italic")


    #PPG
    #Open results
    results = pickle.load(open("4.results/predictions/PPG.pickle", "rb"))  

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
            color = "black", fontstyle = "italic", fontsize = 16,
            bbox=dict(boxstyle="round", alpha = 0.68,
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8),
                   ))
    ax2.set_title("PPG Architecture", fontsize = 10, fontstyle = "italic") 


    #GSR
    #Open results
    results = pickle.load(open("4.results/predictions/GSR.pickle", "rb"))  

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
            color = "black", fontstyle = "italic", fontsize = 16,
            bbox=dict(boxstyle="round", alpha = 0.68,
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8),
                   ))
    ax3.set(ylabel = "Absolute Prediction Error", xlabel = "Label Value")
    ax3.set_title("GSR Architecture", fontsize = 10, fontstyle = "italic") 

    #EEG
    #Open results
    results = pickle.load(open("4.results/predictions/multi.pickle", "rb"))  

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
            color = "black", fontstyle = "italic", fontsize = 16,
            bbox=dict(boxstyle="round", alpha = 0.68,
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8),
                   ))
    ax4.set(xlabel = "Label Value")
    ax4.set_title("Multimodal Architecture", fontsize = 10, fontstyle = "italic") 


    plt.show()

def LabelPlot():
    
    labels = np.array([])

    for participant in participants:
        path = "1.pipeline/prepared_data/"+participant+".pickle"
        dat = pickle.load(open(path, "rb")) #Open pickle
        labels = np.append(labels,dat["labels_EEG"].numpy())

    labels = labels-1
    labels = labels/20
    labels = np.sort(labels)
    lowerbound = labels[round(len(labels)*0.1)]
    upperbound = labels[round(len(labels)*0.9)]
    median = labels[round(len(labels)*0.5)]

    plt.hist(labels, bins = 22, color = "#0092cb")
    plt.vlines((lowerbound, median, upperbound), 0, 1026, colors= "darkslategray", linestyles= ("dashed", "solid", "dashed"))
    plt.ylabel(ylabel='Count')
    plt.xlabel(xlabel = 'Label Value')
    plt.text(lowerbound-0.02, 850, round(lowerbound,2), horizontalalignment='right', verticalalignment='center', 
            color = "black", fontstyle = "italic", fontsize = 11)
    plt.text(upperbound+0.02, 850, round(upperbound,2), horizontalalignment='left', verticalalignment='center', 
            color = "black", fontstyle = "italic", fontsize = 11)
    plt.text(median+0.04, 850, round(median,2), horizontalalignment='left', verticalalignment='center', 
            color = "black", fontstyle = "italic", fontsize = 11)
    plt.show()

def TTests():
    eeg = pickle.load(open("4.results/predictions/EEG.pickle", "rb"))  
    gsr = pickle.load(open("4.results/predictions/GSR.pickle", "rb"))  
    ppg = pickle.load(open("4.results/predictions/PPG.pickle", "rb"))  
    multi = pickle.load(open("4.results/predictions/multi.pickle", "rb"))  

    eegmean = sum(abs(eeg["predictions"] - eeg["labels"])) / len(eeg["predictions"])
    gsrmean = sum(abs(gsr["predictions"] - gsr["labels"])) / len(gsr["predictions"])
    multimean = sum(abs(multi["predictions"] - multi["labels"])) / len(multi["predictions"])
    ppgmean = sum(abs(ppg["predictions"] - ppg["labels"])) / len(ppg["predictions"])

    #Calculate sd
    eegsd, gsrsd, ppgsd, multisd = scipy.std(eeg["predictions"], ddof=1), scipy.std(gsr["predictions"], ddof=1), scipy.std(ppg["predictions"], ddof=1), scipy.std(multi["predictions"], ddof=1)

    #Calculate se
    eegse, gsrse, ppgse, multise = eegsd/ np.sqrt(len(eeg["predictions"])), gsrsd/ np.sqrt(len(gsr["predictions"])), ppgsd/ np.sqrt(len(ppg["predictions"])), multisd/ np.sqrt(len(multi["predictions"]))

    #Calculate sed
    sed_eegmulti = np.sqrt(eegse**2 + multise**2)
    sed_eeggsr = np.sqrt(eegse**2 + gsrse**2)
    sed_gsrmulti = np.sqrt(multise**2 + gsrse**2)
    sed_ppgmulti = np.sqrt(multise**2 + ppgse**2)
    sed_gsrppg = np.sqrt(gsrse**2 + ppgse**2)

    #T stat & df
    t_eegmulti = (eegmean - multimean) / sed_eegmulti
    t_eeggsr = (eegmean - gsrmean) / sed_eeggsr
    t_gsrmulti= (gsrmean - multimean) / sed_gsrmulti
    t_ppgmulti= (multimean - ppgmean) / sed_ppgmulti
    t_gsrppg= (gsrmean - ppgmean) / sed_gsrppg

    df = len(eeg["predictions"]) + len(eeg["predictions"]) - 2

    #p-value
    p_eegmulti = (1 - t.cdf(abs(t_eegmulti), df)) *2
    p_eeggsr = (1 - t.cdf(abs(t_eeggsr), df)) *2
    p_gsrmulti = (1 - t.cdf(abs(t_gsrmulti), df)) *2
    p_ppgmulti = (1 - t.cdf(abs(t_ppgmulti), df)) *2
    p_gsrppg = (1 - t.cdf(abs(t_gsrppg), df)) *2

    #Print results
    print("T-Test EEG and Multimodal:\t t-value = "+ str(round(t_eegmulti,3))+ "\tp = "+ str(round(p_eegmulti,3))+"\tdf = "+ str(df))
    print("T-Test EEG and GSR: \t\t t-value = "+ str(round(t_eeggsr,3))+ "\tp = "+ str(round(p_eeggsr,3))+"\tdf = "+ str(df))
    print("T-Test GSR and Multimodal:\t t-value = "+ str(round(t_gsrmulti,3))+ "\tp = "+ str(round(p_gsrmulti,3))+"\tdf = "+ str(df))
    print("T-Test PPG and Multimodal:\t t-value = "+ str(round(t_ppgmulti,3))+ "\tp = "+ str(round(p_ppgmulti,3))+"\tdf = "+ str(df))
    print("T-Test GSR and PPG:\t\t t-value = "+ str(round(t_gsrppg,3))+ "\tp = "+ str(round(p_gsrppg,3))+"\tdf = "+ str(df))

#Included participants
participants = ["bci10", "bci12", "bci13", "bci17", "bci21", "bci22",
                "bci23", "bci24", "bci26", "bci27", "bci28", "bci29", "bci30", 
                "bci31", "bci32", "bci33", "bci34", "bci35", "bci36", "bci37", 
                "bci38", "bci39", "bci40", "bci41", "bci42", "bci43", "bci44"]

#Modalities
modalities = ["EEG", "GSR", "PPG", "multi"]


for modality in modalities:
    Performance(modality, "test") #Performance statistics per modality

for modality in modalities:
    Performance(modality, "train") #


Descriptives() #Print descriptives
TTests() #Print T-Test results
LabelPlot() #Plot label hist
HistGrid() #Plot grid of error hists
Scatter() #Plot grid of scatterplots
