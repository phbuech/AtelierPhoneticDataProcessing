#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 11:53:30 2023

@author: philipp
"""

import sys
import os
import tqdm
import scipy as sp
import argparse
import shutil
import numpy as np
import pandas as pd

# function deifinition
def create_folder(path):
    try:
        os.mkdir(path)
        print("output folder sucessfully created.")
    except:
        print("output folder was not created.")

def get_common_files(wav_files, txt_files):

    #remove file suffix
    tmp_wav_files = np.array([wav_files[i].split(".")[0] for i in range(len(wav_files))])
    tmp_txt_files = np.array([txt_files[i].split(".")[0] for i in range(len(txt_files))])

    return np.intersect1d(tmp_wav_files,tmp_txt_files)

def extract_utterances(full_input_path,full_output_path,speaker_idx,exception=None):
    files = os.listdir(full_input_path)
    files = [files[i] for i in range(len(files)) if files[i] != ".DS_Store"]
    txt_files = [files[i] for i in range(len(files)) if files[i].endswith(".txt")]
    wav_files = [files[i] for i in range(len(files)) if files[i].endswith(".wav") or files[i].endswith(".WAV")]
    suffix = "." + wav_files[0].split(".")[1]
    common_files = get_common_files(txt_files, wav_files)
    df = pd.DataFrame(columns=["Filename","Transcription"])
    for i in common_files:

        fs, s = sp.io.wavfile.read(full_input_path + i + suffix)

        t = np.linspace(0,len(s)/fs,num=len(s))
        txt_file = list(open(full_input_path + "/"+i+".txt",mode="r",encoding="utf8"))
        txt_file = [txt_file[line_idx].replace("\n","") for line_idx in range(len(txt_file))]
        txt_file = np.array([line.split("\t") for line in txt_file])
        counter = 0
        for j in tqdm.tqdm(txt_file, desc="Extracting utterances, File: "+i):
            counter += 1
            if exception != None:
                if len([ex for ex in exception if ex in j[2]]) > 0 or j[2] == "" or j[2] == "-":
                    continue

            tmin_idx = np.abs(np.float32(j[0]) - t).argmin()
            tmax_idx = np.abs(np.float32(j[1]) - t).argmin()
            snippet = s[tmin_idx:tmax_idx]
            outputfname = speaker_idx+"_"+"fname-"+i+"_U-"+str(counter)
            sp.io.wavfile.write(full_output_path+outputfname+".wav",fs,snippet)
            df.loc[len(df)] = [i,j[2]]
            with open(full_output_path+outputfname+".lab",encoding="utf8",mode="w") as lab_file:
                lab_file.write(j[2])

    df.to_csv(full_output_path+"transcription.csv",sep=",",encoding="utf8",index=False)
        


def prepare_dataset(input_path,output_path,mf,exception=None):
    if os.path.isdir(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)
    if exception != None:
        exception_list = open(exception,mode="r",encoding="utf8")
        exception_list = list(exception_list)
        exception_list = [exception_list[i].replace("\n","") for i in range(len(exception_list))]
    if mf == "yes":
        folder_list = os.listdir(input_path)
        folder_list = [folder_list[i] for i in range(len(folder_list)) if folder_list[i] != ".DS_Store"]

        for folder in folder_list:
            os.mkdir(output_path+"/"+folder)
            if exception != None:
                extract_utterances(input_path + "/" + folder+"/",output_path + "/" + folder+"/",folder,exception_list)
            else:
                extract_utterances(input_path + "/" + folder+"/",output_path + "/" + folder+"/",folder)
            
    print("dataset prepared.")
        
    #folder

parser = argparse.ArgumentParser()
parser.add_argument("-i","--input_path",help="path to the input folder")
parser.add_argument("-o","--output_path",help="path to the output folder")
parser.add_argument("-mf","--multiple_folders",help="multiple folders in the input folder")
parser.add_argument("-el","--exception_list",help="cleans transcription")
args = parser.parse_args()


input_path = args.input_path
output_path = args.output_path
mf = str(args.multiple_folders)
el = args.exception_list

abs_input_path = os.path.abspath(input_path)
abs_output_path = os.path.abspath(output_path)
if os.path.exists(abs_input_path) == False:
    print("input_folder not found.")
elif os.path.exists(output_path) == False:
    answer = input("output folder not found. Create? (yes/no): ")
    if answer == "yes":
        create_folder(abs_output_path)
        prepare_dataset(input_path,output_path,mf)
    else:
        print("corpus preparation aborted.")
elif os.path.exists(output_path) == True:
    answer = input("Warning: existing output folder will be overwritten. Proceed? (yes/no) ")
    if answer == "yes":
        prepare_dataset(input_path,output_path,mf,el)
    else:
        print("corpus preparation aborted.")
else:
    print("corpus preparation aborted.")
