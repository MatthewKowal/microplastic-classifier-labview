# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 14:12:15 2021

@author: Matt
"""

import numpy as np
import pandas as pd
import os
import pickle
import pywt
from matplotlib import pyplot as plt


def add(a, b):
    return a+b

def rp(path):
    newpath = path + "hello"
    return newpath



class Spectra:
    def __init__(self, filename, rawdata, category, color, samplename, source, status):
        self.filename  = filename
        self.rawdata    = rawdata
        self.specdata   = pd.DataFrame()
        self.dwtdata    = pd.DataFrame()
        self.traindata  = pd.DataFrame()
        self.category   = category
        self.color      = color
        self.samplename = samplename
        self.source     = source
        self.status     = status
        self.match      = "no match yet"
        self.dotscore   = 0
        self.matchspec  = pd.DataFrame()
        self.min        = 0
        self.max        = 0

def get_lib_filelist(path): #the library files are in subfolders with material names
    filelist = []
    print(os.listdir(path))
    print("teststst")
    for folder in os.listdir(path):
        working_path = os.path.join(path, folder)
        print(working_path)
        print(os.listdir(working_path))
        for sample in os.listdir(working_path):
            working_file = os.path.join(working_path, sample)
            print(working_file)
            filelist.append(working_file)
    return filelist


def normalize(spectra):
    '''spectra is a single pandas series representing one spectrum'''

    #spectra -= np.min(spectra)
    sumofsquares = np.sum(np.square(spectra)) #compute sum of squares
    newspec = spectra/np.sqrt(sumofsquares)  #normalize to sum of squares
    
    return newspec

def lowpass(spectra, level): 
    '''spectra is a single pandas series representing one spectrum'''
    
    #unpack spectra from dataframe
    #rawdata = spectra#["int"]
    
    #first decompose to spectra to the specified level.
    coeffs = pywt.wavedec(spectra, 'sym5', level=level) #tranform to wavelets
    
    #remove all high frequency levels
    for i in range(1,(level+1)):
        coeffs[i] = np.zeros_like(coeffs[i])
        #print(i)
    
    #reconstruct the wavelets to spectra space
    newspec = pywt.waverec(coeffs, 'sym5')
    
    
    #save data in a new column in dataframe
    #spectra["int2"] = newspec
    return newspec#spectra
    
def removebaseline(spectra, level, iterations):
    '''spectra is a single pandas series representing one spectrum'''
    #rawdata = spectra#["int2"]  #unpack spectra from the dataframe
    bg = spectra#rawdata  #the background will be calculated from the spectra
    for i in range(iterations):
        
        #first decompose to spectra to the specified level.
        coeffs = pywt.wavedec(bg, 'sym5', level=level) #tranform to wavelets
        
        #remove all high frequency levels
        for i in range(2,(level+1)):
            coeffs[i] = np.zeros_like(coeffs[i])
            #print(i)
        
        #reconstruct the wavelets to spectra space
        low_freq_spec = pywt.waverec(coeffs, 'sym5')
        
        #take the mimimum of current spec background vs low freq reconstruction
        bg = np.minimum(bg, low_freq_spec)
        
    #subtract the background from the original spectra
    newspec = spectra - bg#rawdata - bg
    
    #spectra["int3"] = newspec
    return newspec#spectra


def processSpectra(spectra_list, rangeMin, rangeMax):

    
    
    print("processing data...\n\t-Resampling...\n\tCalculating Gradient...\n\t",
          "Calculating Zero-Baseline...\n\tCalculating DWT Coefficients")
    c=0        
    for s in spectra_list:
        print("WORKING ON            ", s.filename)
        #print(a_spectra.status)
        if s.status == "processed":
            print("write code to handle data thats already been processed")
        elif s.status == "raw":
            #print("processing raw data...")
            
            #################################################
            #                                               #
            #   determine the length of each Spectra        #
            #     -do you want to pick the endpoints?       #
            #     -or do you want to leave them untrimmed   #
            #                                               #
            #################################################
            
            #print(s.rawdata)
            trim_all_spectra = True #should we trim all spectra so theyre all
                                    #the same length, start and stop point?
            if trim_all_spectra == False:
                start = (s.rawdata['cm-1'].min()+1).astype(int)
                stop = s.rawdata['cm-1'].max().astype(int)
                if (stop-start)%2 == 1:  stop-=1 #make sure spectrum is even
                num=stop-start
            elif trim_all_spectra == True:
                start = rangeMin
                stop = rangeMax
                if (stop-start)%2 == 1:  stop-=1 #make sure spectrum is even
                num=stop-start
            #print(start, stop, num)
            
            s.min = rangeMin
            s.max = rangeMax
            
            #############################
            #                           #
            #   resample the spectra    #
            #                           #
            #############################
            #print(s.rawdata)
            #assign the cm-1 columns to be the index of the raw data
            s.rawdata.index = s.rawdata['cm-1']
            #create a new regular index
            reg_idx = np.linspace(start, (stop-1), num)
            #create a new dataframe for the newly indexed, blank data
            reg_idx_data = pd.DataFrame(data=[],
                                        index=reg_idx,
                                        columns=["int"])
            #add the new blank dataframe to the original dataframe, then sort it by index
            upsampled_spectrum = pd.concat([s.rawdata, reg_idx_data]).sort_index()
            #print(upsampled_spectrum)
            #interpolate the data and fill it in relative to the index and drop duplicates
            #print(upsampled_spectrum.loc[402:])
            #print(type(upsampled_spectrum))
            interpolated_spectrum = upsampled_spectrum.interpolate(method='index').drop_duplicates()
            #print(interpolated_spectrum.loc[402:])
            #resample the data by saving only the data at the disired index
            resampled_spectrum = interpolated_spectrum.reindex(reg_idx)
            #print(resampled_spectrum)
            #update the spectra object
            s.specdata = resampled_spectrum
            #print(resampled_spectrum)
            
            #############################
            #                           #
            #   prepare gradient        #
            #   processed spectra       #
            #                           #
            #############################
            trimlen = 50
            print(type(s.specdata))
            print(s.specdata['int'].iloc[trimlen])
            s.specdata['int'].iloc[:trimlen] = s.specdata['int'].iloc[trimlen].copy() #flatten first n datapoints
            
            #print("print the type: ", type(spec[i].data['int']))
            #gradient & mean centered spectrum
            s.specdata['grad'] =        lowpass(s.specdata['int'].copy(), 3)
            #s.specdata['grad'] = removebaseline(s.specdata['grad'].copy(), 7, 10)
            s.specdata['grad'] =    np.gradient(s.specdata['grad'].copy(), edge_order=2)
            s.specdata['grad'] =                s.specdata['grad'] - np.mean(s.specdata['grad'].copy())
            s.specdata['grad'] =      normalize(s.specdata['grad'].copy())
            #print(np.sum(np.square(s.specdata['grad'])))        
            
            ######## con todo #########
            
            #gradient & mean centered spectrum
            s.specdata['contodo'] =        lowpass(s.specdata['int'].copy(), 3)
            s.specdata['contodo'] = removebaseline(s.specdata['contodo'].copy(), 7, 10)
            s.specdata['contodo'] =    np.gradient(s.specdata['contodo'].copy(), edge_order=2)
            s.specdata['contodo'] =                s.specdata['contodo'] - np.mean(s.specdata['grad'].copy())
            s.specdata['contodo'] =      normalize(s.specdata['contodo'].copy())
            #print(np.sum(np.square(s.specdata['grad'])))        
            
            
            
            #############################
            #                           #
            #   prepare zero-baseline   #
            #   processed spectra       #
            #                           #
            #############################
                    
            #zero baseline spectrum
            s.specdata['zeroed'] =        lowpass(s.specdata['int'].copy(), 3)
            s.specdata['zeroed'] = removebaseline(s.specdata['zeroed'].copy(), 7, 10)
            s.specdata['zeroed'] =      normalize(s.specdata['zeroed'].copy())
            #print(spec[i].data['zeroed'].index)
            
            
            #############################
            #                           #
            #   prepare dwt coeff       #
            #   of the  spectra         #
            #                           #
            #############################
            #dwt data generation
            level=7
            coeffs = pywt.wavedec(s.specdata['grad'], 'sym5', level=level)
            
            #make sure the index of the dataframe is long enough to store the longest
            #set of dwt coefficients
            dwtidx = np.linspace(0,len(coeffs[level]), (len(coeffs[level])+1))
            #print(dwtidx)
            s.dwtdata = pd.DataFrame(index=dwtidx) #initialize spectra objects dwt data as a dataframe
            
            for i in range(len(coeffs)):
                s.dwtdata[("dwt-"+str(i))]  = pd.Series(coeffs[i])
                
            #s.dwtdata['dwt-0']  = pd.Series(coeffs[0])
            #s.dwtdata['dwt-1']  = pd.Series(coeffs[1])
            #s.dwtdata['dwt-2']  = pd.Series(coeffs[2])
            #s.dwtdata['dwt-3']  = pd.Series(coeffs[3])
            #s.dwtdata['dwt-4']  = pd.Series(coeffs[4])
            
            # print("dwt0 coeffs: ", len(coeffs[0]))
            # print(s.dwtdata['dwt-0'].index)
            # print("dwt1 coeffs: ", len(coeffs[1]))
            # print(s.dwtdata['dwt-1'].index)
            # print("dwt2 coeffs: ", len(coeffs[2]))
            # print(s.dwtdata['dwt-2'].index)
            # print("dwt3 coeffs: ", len(coeffs[3]))
            # print(s.dwtdata['dwt-3'].index)
            # print("dwt4 coeffs: ", len(coeffs[4]))
            # print(s.dwtdata['dwt-4'])
            
            s.traindata = s.specdata['grad']#.to_numpy()
            
            #############################
            #                           #
            #   change the status       #
            #                           #
            #############################
            print(s.filename, "...Done")
            s.status = "processed"
            c+=1
    print("\t\t", c, "of", len(spectra_list), "files processed.")
           
    return spectra_list


###
###
###
###                  LABVIEW ROUTINES
###
###
###


def rawlibrary_to_processedpickle(path_):
    '''reads a folder tree of raw plastic Raman standard spectra.
        Then, processed them and saves the data as a pickle file'''
    
    filelist = get_lib_filelist(path_)

    
    
    print("Importing Spectra...")
    spectra_list = [] 
    for a_file in filelist:
        
        print("HELLO", a_file)
        
        file_name = os.path.basename(a_file)
        
    
        metadata = file_name.split('__')
        if len(metadata)!=5:
            print("ERROR: file (", file_name, ") is missing 1 or more properties")
            print("\t Length of ", len(metadata), " should be 5")
            print("\t", metadata)
        category   = metadata[0] #e.g. pa, ce, pp, pe, etc.
        color      = metadata[1] #e.g. red, green, blue
        samplename = metadata[2] #e.g. Polyamide 1., zoo-23-177
        source     = metadata[3] #e.g. Grant Lab, SLoPP
        status     = metadata[4][:-4] #e.g. raw, preprocessed
                      #metadata[4][:-4] #e.g. raw, preprocessed
        
        #read the csv file from the file list and store it in a pandas dataframe
        print(a_file)
        rawdata = pd.read_csv(a_file, names=['cm-1', 'int'])
        rawdata['int'] = rawdata['int'].astype(float)
        
        #make sure the spectra is of even length
        if len(rawdata)%2 == 1: rawdata = rawdata[:-1] #drop last row if odd
    
        #build a new instance of the spectra class with the spectrums name and data and add it to a list
        new_spectra = Spectra(file_name,
                          rawdata,
                          category, color, samplename, source, status)
        spectra_list.append(new_spectra)   
    print("\t", len(spectra_list), " files loaded.\n")
    
    
    #process spectra
    processed_spectra_list = processSpectra(spectra_list, 402, 1990)
    
    #save processed spectra as a pickle file
    base_path = os.path.split(path_)[0]
    save_pickle_path = os.path.join(base_path, "mp library processed.pickle")
    print("saving pickle fils to: ", save_pickle_path)
    pickle.dump(processed_spectra_list, open(save_pickle_path, "wb"))
    return True
    

    
def get_library_stats(path):
    #print(path)
    spectra_list = pickle.load(open(path, 'rb'))
    #base_path = os.path.split(path)[0]
    #save_pickle_path = os.path.join(base_path, "mp library processed infotest.pickle")
    #print(save_pickle_path)
    #print(spectra_list)
    #pickle.dump(spectra_list, open(save_pickle_path, "wb"))
    return len(spectra_list), spectra_list[0].min, spectra_list[0].max
   

def get_exp_filelist(path): #experimental spetra are all in one folder
    filelist = []
    print(os.listdir(path))
    for filename in os.listdir(path):
        full_filepath = os.path.join(path, filename)
        filelist.append(full_filepath)
    return filelist

def dotScore(v1, v2):
    len1 = np.sqrt(np.dot(v1, v1))
    len2 = np.sqrt(np.dot(v2, v2))
    return np.arccos(np.dot(v1, v2) / (len1 * len2))

def matchSpectra(exp_list, lib_list, col):
    matched_exp_list = []
    for s in exp_list:  
        scores = []
        for l in lib_list:
            scores.append(dotScore(s.specdata[col], l.specdata[col]))           
        best_match_idx = np.argmin(scores)       
        s.match = lib_list[best_match_idx].category
        s.dotscore = scores[best_match_idx]
        s.matchspec = lib_list[best_match_idx].specdata        
        matched_exp_list.append(s)    
    return matched_exp_list
    


def lpmp_exp_data(exp_path, lib_pkl_path):
    #import experimental data
    exp_data = []
    filelist = get_exp_filelist(exp_path)
    for a_file in filelist:
        #determine metadate
        filename = os.path.basename(a_file)
        category   = "unknown"                       #metadata[0] #e.g. pa, ce, pp, pe, etc.
        color      = "not specified"                 #metadata[1] #e.g. red, green, blue
        samplename = filename[:-4]                   #metadata[2] #e.g. Polyamide 1., zoo-23-177
        source     = "Grant Lab Experimental Sample" #metadata[3] #e.g. Grant Lab, SLoPP
        status     = "raw" 
        #read spectral data
        rawdata = pd.read_csv(a_file, names=['cm-1', 'int'])
        rawdata['int'] = rawdata['int'].astype(float)
        #make sure the spectra is of even length
        if len(rawdata)%2 == 1: rawdata = rawdata[:-1] #drop last row if odd
        #build a new instance of the spectra class with the spectrums name and data and add it to a list
        new_spectra = Spectra(filename, rawdata, category, color, samplename, source, status)
        exp_data.append(new_spectra)
        #exp_data is a list of Spectra that conotains all our experimental data
    #process spectra
    p_exp_data = processSpectra(exp_data, 402, 1990)

    #import library data
    lib_data = pickle.load(open(lib_pkl_path, 'rb'))
    
    #find matches between exp_data and library
    matched_spectra = matchSpectra(p_exp_data, lib_data, 'contodo')
    
    #export the matched experimental data as a pickle
    base_path = os.path.split(exp_path)[0]
    save_pickle_path = os.path.join(base_path, "exp data processed and matched.pickle")
    pickle.dump(matched_spectra, open(save_pickle_path, "wb"))
    
    a_string  = "Loaded new Experimental dataset from: " + "\n"
    a_string += "\t\t" + exp_path + "\n"
    a_string += "\t\t" + str(len(filelist)) + " files loaded" +"\n"
    a_string += "\nProcessed & Matched  all spectra\n"
    a_string += "\nSaved pickle file as: \n"
    a_string += "\t\t" + save_pickle_path
    #a_string += exp_path + "hhh" + "bbb" +"\n"+ lib_pkl_path +"\n"+ filelist[0] +"\n"+ filename +"\n"+ category +"\n"
    #a_string += exp_data[0].source +"\n"+ p_exp_data[0].color  +"\n"+ lib_data[0].source +"\n"+ exp_data[0].match
    #a_string += "\n"+ str(matched_spectra[0].dotscore)
    #a_string += "\n"+base_path +"\n"+ save_pickle_path
    return a_string



def export_excel(m_exp_data_pkl_path):
    #load matched data
    m_exp_data =  pickle.load(open(m_exp_data_pkl_path, 'rb'))
    
    #create blank excel file
    output = pd.DataFrame(columns=[])
    
    #fill in excel file
    for s in m_exp_data:
        output = output.append({"sample name":s.samplename,
                                "prediction":s.match,
                                "match score":s.dotscore,
                                }, ignore_index=True)  

    #save excel file
    base_path = os.path.split(m_exp_data_pkl_path)[0]                  
    save_filename = "exp data_matches_excel.csv"
    save_excel_path = os.path.join(base_path, save_filename)
    output.to_csv(save_excel_path)
    
    
    a_string  = "Saved Match Data to:\n"
    a_string += "\t\t" + save_excel_path +"\n\n"
    a_string += "Results:\n"
    a_string += "\n"+ output.to_string() +"\n"
    #a_string = m_exp_data_pkl_path+"\n"+base_path+"\n"+save_filename+"\n"+save_excel_path
    #a_string += "\n"+ output.to_string()
    
    return a_string

def ei(mpath): #export images
    col="contodo"
    base_path = os.path.split(mpath)[0]
    expdata_matches = pickle.load(open(mpath, 'rb'))
    a_string = "Saving composite spectra images:\n"
    
    for s in expdata_matches:
        img_filepath = os.path.join(base_path, (s.samplename + "-spec.png"))
    
        # plt.figure(figsize=(8,8))
        # plt.suptitle(s.samplename)
    
        # plt.subplot(2,1,1)
        # #plot unprocessed experimental data
        # plt.plot(s.specdata["cm-1"], s.specdata["int"],
        #           color="#002145",
        #           label=("Experimental Spectra, Unprocessed"))
        # #plot unprocessed library match
        # plt.plot(s.matchspec["cm-1"], s.matchspec["int"],
        #            color="#00A7E1",
        #            label=("Library Spectra, Unprocessed"))
        # plt.legend()
    
        # plt.subplot(2,1,2)
        # # #plot baseline corrected match
        # plt.plot(s.specdata["cm-1"], s.specdata[col],
        #           color="#002145",
        #           label=("Experimental Spectra, Processed"))
        # # #plot baseline corrected sample
        # plt.plot(s.matchspec["cm-1"], s.matchspec[col],
        #           color="#00A7E1",
        #           label=("Library Spectra, Processed: " + s.match))
                  
        plt.figure(figsize=(12,8))
        plt.suptitle(s.samplename)
    
        plt.subplot(3,1,1)
        #plot unprocessed experimental data
        plt.plot(s.specdata["cm-1"], s.specdata["int"],
                  color="#002145",
                  label=("Experimental Spectra, Unprocessed"))
        #plot unprocessed library match
        plt.plot(s.matchspec["cm-1"], s.matchspec["int"],
                   color="#00A7E1",
                   label=("Library Spectra, Unprocessed"))
        plt.legend()
    
        plt.subplot(3,1,2)
        #plot unprocessed experimental data
        plt.plot(s.specdata["cm-1"], s.specdata["zeroed"],
                  color="#002145",
                  label=("Experimental Spectra, Baseline Corrected"))
        #plot unprocessed library match
        plt.plot(s.matchspec["cm-1"], s.matchspec["zeroed"],
                   color="#00A7E1",
                   label=("Library Spectra, Baseline Corrected"))
        plt.legend()
    
    
        plt.subplot(3,1,3)
        # #plot baseline corrected match
        plt.plot(s.specdata["cm-1"], s.specdata[col],
                  color="#002145",
                  label=("Experimental Spectra, Processed"))
        # #plot baseline corrected sample
        plt.plot(s.matchspec["cm-1"], s.matchspec[col],
                  color="#00A7E1",
                  label=("Library Spectra, Processed: " + s.match))
        
    
    
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(img_filepath)#, bbox_inches="tight")
        #plt.show()
        plt.close()
        a_string += "\t\t" + img_filepath + "\n"
        
    a_string += "\n"    
    #a_string = "test" + mpath + "\n" + base_path + "\n" + img_filepath
    
    return a_string
    
    
# #rawlibrary_to_processedpickle(r'C:\Users\Matt\Desktop\Microplastic Classifier V3\mp library')
    
# #spl = get_library_size(r'C:\Users\Matt\Desktop\Microplastic Classifier V3\mp library processed.pickle')

# #exp_path = r'C:\Users\Matt\Desktop\Microplastic Classifier V3\experimental data'
# #lib_path = r'C:\Users\Matt\Desktop\Microplastic Classifier V3\mp library processed.pickle'

# #load_process_match_pickle_exp_data(exp_path, lib_path)