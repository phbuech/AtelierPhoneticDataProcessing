#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 04:50:14 2023

@author: philipp
"""

import os
import tqdm
import argparse
import shutil
import numpy as np
import scipy as sp
from scipy import signal
from scipy.io import wavfile
import json
import pandas as pd
import chardet
import matplotlib.pyplot as plt
import functools
from collections import OrderedDict
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
allowed_params = ["x","y","z","x-vel","y-vel","z-vel","tvel","tvel-deriv","eucl","eucl-vel","eucl-acc"]
#plt.rcParams["figure.dpi"] = 100
### define functions

def read_header(path_to_pos_file):
    pos_file = open(path_to_pos_file,mode="rb")
    file_content = pos_file.read()
    pos_file.seek(0)
    pos_file.readline()
    header_size = int(pos_file.readline().decode("utf8"))
    header_section = file_content[0:header_size]
    header = header_section.decode("utf8").split("\n")
    device = header[0].split("x")[0] +"x"
    num_of_channels = int(header[2].split("=")[1])
    ema_samplerate = int(header[3].split("=")[1])
    return ema_samplerate, num_of_channels, device

def read_pos_file(path_to_pos_file):
    #read file
    pos_file = open(path_to_pos_file,mode="rb")
    file_content = pos_file.read()

    #read header
    pos_file.seek(0)
    pos_file.readline()
    header_size = int(pos_file.readline().decode("utf8"))
    header_section = file_content[0:header_size]
    header = header_section.decode("utf8").split("\n")
    device = header[0]
    num_of_channels = int(header[2].split("=")[1])
    ema_samplerate = int(header[3].split("=")[1])
    
    #read data
    data = file_content[header_size:]
    if "AG50x" in device:
        data = np.frombuffer(data,np.float32)
        data = np.reshape(data,newshape=(-1,112))
    return ema_samplerate, num_of_channels, data

def extract_ema_data(data,ema_channels,sample_order):
    """
    extracts the ema data for the x and y dimensions for all recorded channels (see channel_allocation)
    """
    # initialize dictionary
    channel_names = list(ema_channels.keys())
    ext_data = {}
    for i in range(len(channel_names)): ext_data[channel_names[i]+"_x"] = []
    for i in range(len(channel_names)): ext_data[channel_names[i]+"_y"] = []
    for i in range(len(channel_names)): ext_data[channel_names[i]+"_z"] = []

    for sample_idx in range(len(data)):
        sample = data[sample_idx]
        sample = np.reshape(sample,newshape=(-1,len(sample_order)))

        for channel_name_idx in range(len(channel_names)):

            # get sample values for x and y
            channel_number = ema_channels[channel_names[channel_name_idx]]
            this_channel_x_sample = sample[channel_number-1,sample_order["x"]]
            this_channel_y_sample = sample[channel_number-1,sample_order["y"]]
            this_channel_z_sample = sample[channel_number-1,sample_order["z"]]

            # add values
            ext_data[channel_names[channel_name_idx]+"_x"].append(this_channel_x_sample)
            ext_data[channel_names[channel_name_idx]+"_y"].append(this_channel_y_sample)
            ext_data[channel_names[channel_name_idx]+"_z"].append(this_channel_z_sample)
    #convert to numpy array
    dimensions = ["_x","_y","_y"]
    for dim_idx in range(len(dimensions)):
        for channel_name_idx in range(len(channel_names)):
            ext_data[channel_names[channel_name_idx]+dimensions[dim_idx]] = np.array(ext_data[channel_names[channel_name_idx]+dimensions[dim_idx]])
    return ext_data

def mean_filter(data,N):
    return np.convolve(data,np.ones(N)/N,mode="same")

def butter_lowpass(cutoff, nyq_freq, order=4):
    normal_cutoff = float(cutoff) / nyq_freq
    b, a = signal.butter(order, normal_cutoff, btype='lowpass')
    return b, a

def butter_lowpass_filter(data, cutoff_freq, nyq_freq, order):
    # Source: https://github.com/guillaume-chevalier/filtering-stft-and-laplace-transform
    b, a = butter_lowpass(cutoff_freq, nyq_freq, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def smoothing(data,signal_filter,ema_fs):
    filter_type = list(signal_filter.keys())[0]
    
    if filter_type == "butter":
        for i in range(len(data)): data[list(data.keys())[i]] = butter_lowpass_filter(data=data[list(data.keys())[i]], cutoff_freq=signal_filter["butter"][0], nyq_freq=ema_fs/2, order=signal_filter["butter"][1])
    elif filter_type == "moving_average":
        for i in range(len(data)): data[list(data.keys())[i]] = mean_filter(data=data[list(data.keys())[i]], N=signal_filter["moving_average"])
    return data

def derivation(data,ema_fs,order):
    x = np.linspace(0.0,len(data)/ema_fs,num=len(data))
    tmp = data
    dx = x[1]-x[0]
    for i in range(order):
        tmp = np.gradient(tmp,dx)
    return tmp

def get_common_files(wav_files, ema_files, tg_files):

    #remove file suffix
    tmp_wav_files = np.sort(np.array([wav_files[i].replace(".wav","") for i in range(len(wav_files))]))
    tmp_ema_files = np.sort(np.array([ema_files[i].replace(".pos","") for i in range(len(ema_files))]))
    tmp_tg_files = np.sort(np.array([tg_files[i].replace(".TextGrid","") for i in range(len(tg_files))]))
    return functools.reduce(np.intersect1d,(tmp_wav_files,tmp_ema_files,tmp_tg_files))

def extract_parameters_of_interest(data,poi,ema_fs):

    # initialize results dictionary
    ext_param_data = {}
    
    poi_sensors = list(poi.keys())

    for i in range(len(poi_sensors)):

        # A "parameter of interest" (poi) consists of two parts:
        # 1) the sensor with a number in front of it, e.g., "2_ttip"
        # 2) the dimension of movement, e.g., "y" or "y-vel"
        current_poi_sensor = poi_sensors[i]
        current_poi_dimension = poi[current_poi_sensor]

        if len(current_poi_sensor.split("+")) == 1:

            # position (x, y, or z)
            if len(current_poi_dimension) == 1:
                channel_name = current_poi_sensor.split("_")[1]
                ext_param_data[current_poi_sensor+"_"+current_poi_dimension] = data[channel_name+"_"+current_poi_dimension]

            # velocity and tangential velocity
            elif current_poi_dimension.endswith("vel"):

                # tangential velocity of x and y dimension
                if current_poi_dimension == "tvel":
                    channel_name = current_poi_sensor.split("_")[1]
                    tmp_data_x_vel = derivation(data=data[channel_name+"_x"],ema_fs=ema_fs,order=1)
                    tmp_data_y_vel = derivation(data=data[channel_name+"_y"],ema_fs=ema_fs,order=1)
                    ext_param_data[current_poi_sensor+"_"+current_poi_dimension] = np.sqrt(tmp_data_x_vel**2 + tmp_data_y_vel**2)

                # regular velocity one dimension
                else:
                    channel_name = current_poi_sensor.split("_")[1]
                    dimension = current_poi_dimension.split("-")[0]
                    ext_param_data[current_poi_sensor+"_"+current_poi_dimension] = derivation(data[channel_name+"_"+dimension],ema_fs=ema_fs,order=1)

            # acceleration (2nd derivative of position)
            elif current_poi_dimension.endswith("acc"):
                channel_name = current_poi_sensor.split("_")[1]
                dimension = current_poi_dimension.split("-")[0]
                ext_param_data[current_poi_sensor+"_"+current_poi_dimension] = derivation(data[channel_name+"_"+dimension],ema_fs=ema_fs,order=2)

            # derivative of tangential velocity of x and y
            elif current_poi_dimension == "tvel-deriv":
                channel_name = current_poi_sensor.split("_")[1]
                tmp_data_x_vel = derivation(data=data[channel_name+"_x"],ema_fs=ema_fs,order=1)
                tmp_data_y_vel = derivation(data=data[channel_name+"_y"],ema_fs=ema_fs,order=1)
                tvel = np.sqrt(tmp_data_x_vel**2 + tmp_data_y_vel**2)
                ext_param_data[current_poi_sensor+"_"+current_poi_dimension] = derivation(data=tvel,ema_fs=ema_fs,order=1)

        # If the definition contains a plus, the euclidean distance of two sensor is demanded
        # e.g., "0_ulip+llip"
        elif len(current_poi_sensor.split("+")) == 2:

            # euclidean distance between two sensors
            if current_poi_dimension == "eucl":
                ext_param_data[current_poi_sensor + "_" + current_poi_dimension] = calculate_euclidean_distance(current_poi_sensor, data)

            # velocity of euclidean distance between two sensors
            elif current_poi_dimension == "eucl-vel":
                ext_param_data[current_poi_sensor + "_" + current_poi_dimension] = get_eucl_derivative(current_poi_sensor, data, ema_fs, 1)
            
            # acceleration of euclidean distance between two sensors
            elif current_poi_dimension == "eucl-acc":
                ext_param_data[current_poi_sensor+ "_" + current_poi_dimension] = get_eucl_derivative(current_poi_sensor, data, ema_fs, 2)

    return ext_param_data

def calculate_euclidean_distance(parameter_of_interest, data):
    articulators = parameter_of_interest.split("_")[1]
    articulator1, articulator2 = articulators.split("+")
    articulator1_x = data[articulator1+"_x"]
    articulator1_y = data[articulator1+"_y"]
    articulator2_x = data[articulator2+"_x"]
    articulator2_y = data[articulator2+"_y"]
    return np.array([np.sqrt((articulator1_x[j]-articulator2_x[j])**2+(articulator1_y[j]-articulator2_y[j])**2) for j in range(len(articulator1_x))] )

def get_eucl_derivative(parameter_of_interest, data, ema_fs, order):
    eucl = calculate_euclidean_distance(parameter_of_interest, data)
    eucl_deriv = derivation(data = eucl, ema_fs = ema_fs, order = order)
    return(eucl_deriv)


def get_quotation_marks_idxs(line):
    return np.where(np.array(list(line)) == "\"")[0]


def deconstruct_tier(tier):
    structured_data = []
    for tier_idx in range(len(tier)):
        tmp_tier = tier[tier_idx+1]
        tier_class_denomination_idxs = get_quotation_marks_idxs(tmp_tier[1])
        tier_class = "".join(list(tmp_tier[1])[tier_class_denomination_idxs[0]+1:tier_class_denomination_idxs[1]])
        
        tier_name_denomination_idxs = get_quotation_marks_idxs(tmp_tier[2])
        tier_name = "".join(list(tmp_tier[2])[tier_name_denomination_idxs[0]+1:tier_name_denomination_idxs[1]])
        if tier_class == "IntervalTier":
            number_of_intervals = int(tmp_tier[5].replace("        intervals: size = ","").replace(" ","").replace("\n",""))
            
            intervals_lines = np.array(tmp_tier[6:len(tmp_tier)])
            
            intervals = np.reshape(intervals_lines,newshape=(-1,4))
            for interval_idx in range(len(intervals)):
                intervals[interval_idx][0] = int(intervals[interval_idx][0].replace("        intervals [","").replace("]:","").replace("\n",""))
                intervals[interval_idx][1] = float(intervals[interval_idx][1].replace("            xmin = ","").replace("\n","").replace(" ",""))
                intervals[interval_idx][2] = float(intervals[interval_idx][2].replace("            xmax = ","").replace("\n","").replace(" ",""))
                label_idxs = get_quotation_marks_idxs(intervals[interval_idx][3])
                intervals[interval_idx][3] = intervals[interval_idx][3][label_idxs[0]+1:label_idxs[1]]
            structured_data.append([tier_idx,tier_class,tier_name,intervals])
        elif tier_class == "TextTier":
            number_of_intervals = int(tmp_tier[5].replace("        points: size = ","").replace(" ","").replace("\n",""))
            intervals_lines = np.array(tmp_tier[6:len(tmp_tier)])
            intervals = np.reshape(intervals_lines,newshape=(-1,3))
            nintervals = []
            for interval_idx in range(len(intervals)):
                intervals[interval_idx][0] = int(intervals[interval_idx][0].replace("        points [","").replace("]:","").replace("\n",""))
                intervals[interval_idx][1] = float(intervals[interval_idx][1].replace("            number = ","").replace("\n","").replace(" ",""))
                label_idxs = get_quotation_marks_idxs(intervals[interval_idx][2])
                intervals[interval_idx][2] = intervals[interval_idx][2][label_idxs[0]+1:label_idxs[1]]
                nintervals.append([intervals[interval_idx][0],intervals[interval_idx][1],intervals[interval_idx][1],intervals[interval_idx][2]])
            nintervals = np.array(nintervals)
            structured_data.append([tier_idx,tier_class,tier_name,nintervals])
    return structured_data
    
def get_df(data):
    df = pd.DataFrame(columns=["TierID","TierClass","TierName","IntervallNo","tmin","tmax","label"])
    for tier_idx in range(len(data)):
        for interval_idx in range(len(data[tier_idx][3])):
            df.loc[len(df)+1] = [tier_idx+1,data[tier_idx][1],data[tier_idx][2]] + data[tier_idx][3][interval_idx].tolist()
    df["TierClass"] = df["TierClass"].astype(str)
    df["TierName"] = df["TierName"].astype(str)
    df["label"] = df["label"].astype(str)
    df["tmin"] = df["tmin"].astype(float)
    df["tmax"] = df["tmax"].astype(float)
    #df.to_csv(output_path+filename+".csv",sep=",",encoding="utf8",index=False)
    return df

def read_textgrid(path):
    rawdata = open(path,"rb").read()
    result = chardet.detect(rawdata)
    enc = result["encoding"]
        
    file = open(path,mode="r",encoding=enc)
    lines = list(file)
        
    number_of_tiers = int(lines[6].replace("size","").replace("=","").replace(" ",""))
    tiers_idxs = [j for j in range(len(lines)) if "   item [" in lines[j]]
        
    part_tier_lines = {}
    if len(tiers_idxs) == 1:
            part_tier_lines[1] = lines[tiers_idxs[0]:len(lines)]
    elif len(tiers_idxs) > 1:
            for idx in range(len(tiers_idxs)-1):
                part_tier_lines[idx+1] = lines[tiers_idxs[idx]:tiers_idxs[idx+1]]
            part_tier_lines[len(part_tier_lines)+1] = lines[tiers_idxs[len(tiers_idxs)-1]:len(lines)]
    struct_data = deconstruct_tier(part_tier_lines)
    df = get_df(data=struct_data)
    return df


def tangential_velocity(data,params,fs):
    tmp_sum = derivation(data[params+"_x"],fs,1)**2 + derivation(data[params+"_y"],fs,1)**2
    
    return np.sqrt(tmp_sum)


def detrend_data_2params(data,time, params,fs,order,lower,upper):
    detrended_data = {}
    trendps = {}
    X = np.reshape(time,(len(time),1))
    pf = PolynomialFeatures(degree=order)
    Xp = pf.fit_transform(X)
    md1 = LinearRegression()
    md2 = LinearRegression()
    md1.fit(Xp,data[params+"_x"][lower:upper])
    md2.fit(Xp,data[params+"_y"][lower:upper])
    trendps[params+"_x"] = md1.predict(Xp)
    trendps[params+"_y"] = md2.predict(Xp)
    detrended_data[params+"_x"] = (data[params+"_x"][lower:upper] - trendps[params+"_x"])
    detrended_data[params+"_y"] = (data[params+"_y"][lower:upper] - trendps[params+"_y"])
    
    return trendps, detrended_data
    
def detrend_data_1param(data,time,param,fs,order,lower,upper):
    detrended_data = {}
    trendps = {}
    X = np.reshape(time,(len(time),1))
    pf = PolynomialFeatures(degree=order)
    Xp = pf.fit_transform(X)
    md1 = LinearRegression()
    md1.fit(Xp,data[param][lower:upper])
    trendps[param] = md1.predict(Xp)
    detrended_data[param] = (data[param][lower:upper] - trendps[param])    
    return trendps, detrended_data

   
def gauss_function(x,mu,sigma):
    return np.exp(-(x-mu )**2/(2*sigma**2))


def build_window(x,g1,g2,tmin,tmax):
    tmp = []
    for xi in range(len(x)):
        if x[xi] < tmin:
            tmp.append(g1[xi])
        elif x[xi] >= tmin and x[xi] <= tmax:
            tmp.append(1.0)
        elif x[xi] > tmax:
            tmp.append(g2[xi])
    return np.array(tmp)

def window_function(x,tmin,target_time,release_time,tmax):
    mu1 = target_time
    sigma1 = np.abs(tmin-target_time)/4
    gauss_function1 = gauss_function(x,mu1,sigma1)
    
    mu2 = release_time
    sigma2 = np.abs(release_time-tmax)/4
    gauss_function2 = gauss_function(x,mu2,sigma2)
    
    
    
    window_fun = build_window(x,gauss_function1,gauss_function2,target_time,release_time)
    
    return window_fun


def landmark_detection_tvel(x,data1, data2, tmid_idx,wfun):
    landmarks = {}
    
    windowed_signal = data2*wfun
    pvel_to_idx = np.where(windowed_signal[:tmid_idx] == np.max(windowed_signal[:tmid_idx]))[0][0]
    pvel_fro_idx = np.where(windowed_signal[tmid_idx:] == np.max(windowed_signal[tmid_idx:]))[0][0] + tmid_idx
    
    minima = sp.signal.argrelextrema(data1,np.less)[0]
    minima_between_pvels = minima[np.where(np.logical_and(minima > pvel_to_idx,minima < pvel_fro_idx))[0]]
    
    minima_before_pvel_to = minima[np.where(minima < pvel_to_idx)[0]]
    minima_after_pvel_fro = minima[np.where(minima > pvel_fro_idx)[0]]
    
    velmax_pvel_to = data1[pvel_to_idx]*0.2
    velmax_pvel_fro = data1[pvel_fro_idx]*0.2
    
    
    tgt_idx = np.abs(velmax_pvel_to - data1[pvel_to_idx:minima_between_pvels[0]]).argmin() + pvel_to_idx
    release_idx = np.abs(velmax_pvel_fro - data1[minima_between_pvels[len(minima_between_pvels)-1]:pvel_fro_idx]).argmin() + minima_between_pvels[len(minima_between_pvels)-1]
    
    onset_idx = np.abs(velmax_pvel_to - data1[minima_before_pvel_to[len(minima_before_pvel_to)-1]:pvel_to_idx]).argmin() + minima_before_pvel_to[len(minima_before_pvel_to)-1]
    offset_idx = np.abs(velmax_pvel_fro - data1[pvel_fro_idx:minima_after_pvel_fro[0]]).argmin() + pvel_fro_idx
    
    landmarks["pvel_to"] = pvel_to_idx
    landmarks["pvel_fro"] = pvel_fro_idx
    landmarks["target"] = tgt_idx
    landmarks["release"] = release_idx
    landmarks["onset"] = onset_idx
    landmarks["offset"] = offset_idx
    return landmarks

def detect_landmarks_vel(x,velocity,wfun):
    landmarks = {}
    
    windowed_signal = velocity*wfun
    maximum = np.where(windowed_signal == np.max(windowed_signal))[0][0]
    minimum = np.where(windowed_signal == np.min(windowed_signal))[0][0]
    if x[maximum] > x[minimum]:
        pvel_to_idx = minimum
        pvel_fro_idx = maximum
    else:
        pvel_to_idx = maximum
        pvel_fro_idx = minimum
        
    velmax_pvel_to = velocity[pvel_to_idx]*0.2
    velmax_pvel_fro = velocity[pvel_fro_idx]*0.2
    
    tgt_idx = np.abs(velmax_pvel_to - velocity[pvel_to_idx:pvel_fro_idx]).argmin() + pvel_to_idx
    release_idx = np.abs(velmax_pvel_fro - velocity[pvel_to_idx:pvel_fro_idx]).argmin() + pvel_to_idx
    
    onset_idx = np.abs(velmax_pvel_to -velocity[:pvel_to_idx]).argmin()
    offset_idx = np.abs(velmax_pvel_fro - velocity[pvel_fro_idx:]).argmin() + pvel_fro_idx
    
    landmarks["pvel_to"] = pvel_to_idx
    landmarks["pvel_fro"] = pvel_fro_idx
    landmarks["target"] = tgt_idx
    landmarks["release"] = release_idx
    landmarks["onset"] = onset_idx
    landmarks["offset"] = offset_idx
    
    return landmarks
    
    
def detect_landmarks_vel20(x,velocity):
    
    landmarks = {}
    maximum = np.where(velocity == np.max(velocity))[0][0]
    minimum = np.where(velocity == np.min(velocity))[0][0]
    if x[maximum] > x[minimum]:
        pvel_to_idx = minimum
        pvel_fro_idx = maximum
    else:
        pvel_to_idx = maximum
        pvel_fro_idx = minimum
        
    velmax_pvel_to = velocity[pvel_to_idx]*0.2
    velmax_pvel_fro = velocity[pvel_fro_idx]*0.2
    
    tgt_idx = np.abs(velmax_pvel_to - velocity[pvel_to_idx:pvel_fro_idx]).argmin() + pvel_to_idx
    release_idx = np.abs(velmax_pvel_fro - velocity[pvel_to_idx:pvel_fro_idx]).argmin() + pvel_to_idx
    
    onset_idx = np.abs(velmax_pvel_to -velocity[:pvel_to_idx]).argmin()
    offset_idx = np.abs(velmax_pvel_fro - velocity[pvel_fro_idx:]).argmin() + pvel_fro_idx
    
    landmarks["pvel_to"] = pvel_to_idx
    landmarks["pvel_fro"] = pvel_fro_idx
    landmarks["target"] = tgt_idx
    landmarks["release"] = release_idx
    landmarks["onset"] = onset_idx
    landmarks["offset"] = offset_idx
    
    return landmarks

def detect_landmarks_tvel20(x,tvel,tmid_idx):
    landmarks = {}
    
    pvel_to_idx = np.where(tvel[:tmid_idx] == np.max(tvel[:tmid_idx]))[0][0]
    pvel_fro_idx = np.where(tvel[tmid_idx:] == np.max(tvel[tmid_idx:]))[0][0] + tmid_idx
    
    minima = sp.signal.argrelextrema(tvel,np.less)[0]
    minima_between_pvels = minima[np.where(np.logical_and(minima > pvel_to_idx,minima < pvel_fro_idx))[0]]
    
    minima_before_pvel_to = minima[np.where(minima < pvel_to_idx)[0]]
    minima_after_pvel_fro = minima[np.where(minima > pvel_fro_idx)[0]]
    
    velmax_pvel_to = tvel[pvel_to_idx]*0.2
    velmax_pvel_fro = tvel[pvel_fro_idx]*0.2
    

    tgt_idx = np.abs(velmax_pvel_to - tvel[pvel_to_idx:minima_between_pvels[0]]).argmin() + pvel_to_idx
    release_idx = np.abs(velmax_pvel_fro - tvel[minima_between_pvels[len(minima_between_pvels)-1]:pvel_fro_idx]).argmin() + minima_between_pvels[len(minima_between_pvels)-1]
    
    onset_idx = np.abs(velmax_pvel_to - tvel[minima_before_pvel_to[len(minima_before_pvel_to)-1]:pvel_to_idx]).argmin() + minima_before_pvel_to[len(minima_before_pvel_to)-1]
    offset_idx = np.abs(velmax_pvel_fro - tvel[pvel_fro_idx:minima_after_pvel_fro[0]]).argmin() + pvel_fro_idx
    
    landmarks["pvel_to"] = pvel_to_idx
    landmarks["pvel_fro"] = pvel_fro_idx
    landmarks["target"] = tgt_idx
    landmarks["release"] = release_idx
    landmarks["onset"] = onset_idx
    landmarks["offset"] = offset_idx
    return landmarks


def detect_landmarks_vel20_windowed(x,velocity,wfun):
    landmarks = {}
    velocity_windowed = velocity*wfun
    maximum = np.where(velocity_windowed == np.max(velocity_windowed))[0][0]
    minimum = np.where(velocity_windowed == np.min(velocity_windowed))[0][0]
    if x[maximum] > x[minimum]:
        pvel_to_idx = minimum
        pvel_fro_idx = maximum
    else:
        pvel_to_idx = maximum
        pvel_fro_idx = minimum
        
    velmax_pvel_to = velocity[pvel_to_idx]*0.2
    velmax_pvel_fro = velocity[pvel_fro_idx]*0.2
    
    tgt_idx = np.abs(velmax_pvel_to - velocity[pvel_to_idx:pvel_fro_idx]).argmin() + pvel_to_idx
    release_idx = np.abs(velmax_pvel_fro - velocity[pvel_to_idx:pvel_fro_idx]).argmin() + pvel_to_idx
    
    onset_idx = np.abs(velmax_pvel_to -velocity[:pvel_to_idx]).argmin()
    offset_idx = np.abs(velmax_pvel_fro - velocity[pvel_fro_idx:]).argmin() + pvel_fro_idx
    
    landmarks["pvel_to"] = pvel_to_idx
    landmarks["pvel_fro"] = pvel_fro_idx
    landmarks["target"] = tgt_idx
    landmarks["release"] = release_idx
    landmarks["onset"] = onset_idx
    landmarks["offset"] = offset_idx
    
    return landmarks

def detect_landmarks_tvel20_windowed(x,tvel,wfun,tmid_idx):
    landmarks = {}
    tvel_windowed = tvel*wfun
    pvel_to_idx = np.where(tvel_windowed[:tmid_idx] == np.max(tvel_windowed[:tmid_idx]))[0][0]
    pvel_fro_idx = np.where(tvel_windowed[tmid_idx:] == np.max(tvel_windowed[tmid_idx:]))[0][0] + tmid_idx
    
    minima = sp.signal.argrelextrema(tvel,np.less)[0]
    minima_between_pvels = minima[np.where(np.logical_and(minima > pvel_to_idx,minima < pvel_fro_idx))[0]]
    
    minima_before_pvel_to = minima[np.where(minima < pvel_to_idx)[0]]
    minima_after_pvel_fro = minima[np.where(minima > pvel_fro_idx)[0]]
    
    velmax_pvel_to = tvel[pvel_to_idx]*0.2
    velmax_pvel_fro = tvel[pvel_fro_idx]*0.2
    
    
    tgt_idx = np.abs(velmax_pvel_to - tvel[pvel_to_idx:minima_between_pvels[0]]).argmin() + pvel_to_idx
    release_idx = np.abs(velmax_pvel_fro - tvel[minima_between_pvels[len(minima_between_pvels)-1]:pvel_fro_idx]).argmin() + minima_between_pvels[len(minima_between_pvels)-1]
    
    onset_idx = np.abs(velmax_pvel_to - tvel[minima_before_pvel_to[len(minima_before_pvel_to)-1]:pvel_to_idx]).argmin() + minima_before_pvel_to[len(minima_before_pvel_to)-1]
    offset_idx = np.abs(velmax_pvel_fro - tvel[pvel_fro_idx:minima_after_pvel_fro[0]]).argmin() + pvel_fro_idx
    
    landmarks["pvel_to"] = pvel_to_idx
    landmarks["pvel_fro"] = pvel_fro_idx
    landmarks["target"] = tgt_idx
    landmarks["release"] = release_idx
    landmarks["onset"] = onset_idx
    landmarks["offset"] = offset_idx
    return landmarks

def get_levels(dictionary):
    levels = list(dictionary.values())
    return levels

def create_annotation(time_ema,annotation,landmark_df,extracted_ema_data,segment_idx,padding,param,ema_fs,counter):
    label = annotation["label"][segment_idx]
    tmin = annotation["tmin"][segment_idx]
    tmax = annotation["tmax"][segment_idx]
    tmid = (tmin+tmax)/2
    tmin_pad = tmin-padding
    tmax_pad = tmax+padding
    tmin_ema_idx = np.abs(tmin_pad-time_ema).argmin()
    tmax_ema_idx = np.abs(tmax_pad-time_ema).argmin()
    time_subset = time_ema[tmin_ema_idx:tmax_ema_idx]
    tmid_ema_idx = np.abs(tmid-time_subset).argmin()
    
    if param["method"] == "vel20":
        velocity = derivation(extracted_ema_data[param["channel"]],ema_fs,1)[tmin_ema_idx:tmax_ema_idx]
        landmarks = detect_landmarks_vel20(time_subset,velocity)
        """
        fig, ax = plt.subplots(2,1,figsize=(10,10))
        ax[0].plot(time_subset,extracted_ema_data[param["channel"]][tmin_ema_idx:tmax_ema_idx])
        ax[1].plot(time_subset,velocity)
        ax[0].scatter(time_subset[landmarks["onset"]],extracted_ema_data[param["channel"]][tmin_ema_idx:tmax_ema_idx][landmarks["onset"]])
        ax[0].scatter(time_subset[landmarks["pvel_to"]],extracted_ema_data[param["channel"]][tmin_ema_idx:tmax_ema_idx][landmarks["pvel_to"]])
        ax[0].scatter(time_subset[landmarks["target"]],extracted_ema_data[param["channel"]][tmin_ema_idx:tmax_ema_idx][landmarks["target"]])
        ax[0].scatter(time_subset[landmarks["release"]],extracted_ema_data[param["channel"]][tmin_ema_idx:tmax_ema_idx][landmarks["release"]])
        ax[0].scatter(time_subset[landmarks["pvel_fro"]],extracted_ema_data[param["channel"]][tmin_ema_idx:tmax_ema_idx][landmarks["pvel_fro"]])
        ax[0].scatter(time_subset[landmarks["offset"]],extracted_ema_data[param["channel"]][tmin_ema_idx:tmax_ema_idx][landmarks["offset"]])
        ax[0].set_title(label)
        """
        landmark_df.loc[len(landmark_df)] = [0,"TextTier",label,0,time_subset[landmarks["onset"]],time_subset[landmarks["onset"]],label+"_ONS"]
        landmark_df.loc[len(landmark_df)] = [0,"TextTier",label,0,time_subset[landmarks["pvel_to"]],time_subset[landmarks["pvel_to"]],label+"_pVelTo"]
        landmark_df.loc[len(landmark_df)] = [0,"TextTier",label,0,time_subset[landmarks["target"]],time_subset[landmarks["target"]],label+"_TAR"]
        landmark_df.loc[len(landmark_df)] = [0,"TextTier",label,0,time_subset[landmarks["release"]],time_subset[landmarks["release"]],label+"_REL"]
        landmark_df.loc[len(landmark_df)] = [0,"TextTier",label,0,time_subset[landmarks["pvel_fro"]],time_subset[landmarks["pvel_fro"]],label+"_pVelFro"]
        landmark_df.loc[len(landmark_df)] = [0,"TextTier",label,0,time_subset[landmarks["offset"]],time_subset[landmarks["offset"]],label+"_OFF"]
        
    elif param["method"] == "tvel20":
        tvel = tangential_velocity(extracted_ema_data,param["channel"],ema_fs)[tmin_ema_idx:tmax_ema_idx]
        landmarks = detect_landmarks_tvel20(time_subset,tvel,tmid_ema_idx)
        """
        fig, ax = plt.subplots(2,1,figsize=(10,10))
        ax[0].plot(time_subset,extracted_ema_data[param["channel"]+"_y"][tmin_ema_idx:tmax_ema_idx])
        ax[1].plot(time_subset,tvel)
        ax[0].scatter(time_subset[landmarks["onset"]],extracted_ema_data[param["channel"]+"_y"][tmin_ema_idx:tmax_ema_idx][landmarks["onset"]])
        ax[0].scatter(time_subset[landmarks["pvel_to"]],extracted_ema_data[param["channel"]+"_y"][tmin_ema_idx:tmax_ema_idx][landmarks["pvel_to"]])
        ax[0].scatter(time_subset[landmarks["target"]],extracted_ema_data[param["channel"]+"_y"][tmin_ema_idx:tmax_ema_idx][landmarks["target"]])
        ax[0].scatter(time_subset[landmarks["release"]],extracted_ema_data[param["channel"]+"_y"][tmin_ema_idx:tmax_ema_idx][landmarks["release"]])
        ax[0].scatter(time_subset[landmarks["pvel_fro"]],extracted_ema_data[param["channel"]+"_y"][tmin_ema_idx:tmax_ema_idx][landmarks["pvel_fro"]])
        ax[0].scatter(time_subset[landmarks["offset"]],extracted_ema_data[param["channel"]+"_y"][tmin_ema_idx:tmax_ema_idx][landmarks["offset"]])
        ax[0].set_title(label)
        """
        landmark_df.loc[len(landmark_df)] = [0,"TextTier",label,0,time_subset[landmarks["onset"]],time_subset[landmarks["onset"]],label+"_ONS"]
        landmark_df.loc[len(landmark_df)] = [0,"TextTier",label,0,time_subset[landmarks["pvel_to"]],time_subset[landmarks["pvel_to"]],label+"_pVelTo"]
        landmark_df.loc[len(landmark_df)] = [0,"TextTier",label,0,time_subset[landmarks["target"]],time_subset[landmarks["target"]],label+"_TAR"]
        landmark_df.loc[len(landmark_df)] = [0,"TextTier",label,0,time_subset[landmarks["release"]],time_subset[landmarks["release"]],label+"_REL"]
        landmark_df.loc[len(landmark_df)] = [0,"TextTier",label,0,time_subset[landmarks["pvel_fro"]],time_subset[landmarks["pvel_fro"]],label+"_pVelFro"]
        landmark_df.loc[len(landmark_df)] = [0,"TextTier",label,0,time_subset[landmarks["offset"]],time_subset[landmarks["offset"]],label+"_OFF"]
    elif param["method"] == "vel20_windowed":
        velocity = derivation(extracted_ema_data[param["channel"]],ema_fs,1)[tmin_ema_idx:tmax_ema_idx]
        wfun = window_function(time_subset, tmin_pad, tmin-0.025, tmax+0.025, tmax_pad)
        landmarks = detect_landmarks_vel20_windowed(time_subset,velocity,wfun)
        """
        fig, ax = plt.subplots(4,1,figsize=(10,10))
        ax[0].plot(time_subset,extracted_ema_data[param["channel"]][tmin_ema_idx:tmax_ema_idx])
        ax[1].plot(time_subset,velocity)
        ax[3].plot(time_subset,velocity*wfun)
        ax[2].plot(time_subset,wfun)
        ax[0].scatter(time_subset[landmarks["onset"]],extracted_ema_data[param["channel"]][tmin_ema_idx:tmax_ema_idx][landmarks["onset"]])
        ax[0].scatter(time_subset[landmarks["pvel_to"]],extracted_ema_data[param["channel"]][tmin_ema_idx:tmax_ema_idx][landmarks["pvel_to"]])
        ax[0].scatter(time_subset[landmarks["target"]],extracted_ema_data[param["channel"]][tmin_ema_idx:tmax_ema_idx][landmarks["target"]])
        ax[0].scatter(time_subset[landmarks["release"]],extracted_ema_data[param["channel"]][tmin_ema_idx:tmax_ema_idx][landmarks["release"]])
        ax[0].scatter(time_subset[landmarks["pvel_fro"]],extracted_ema_data[param["channel"]][tmin_ema_idx:tmax_ema_idx][landmarks["pvel_fro"]])
        ax[0].scatter(time_subset[landmarks["offset"]],extracted_ema_data[param["channel"]][tmin_ema_idx:tmax_ema_idx][landmarks["offset"]])
        ax[0].set_title(label)
        """
        landmark_df.loc[len(landmark_df)] = [0,"TextTier",label,0,time_subset[landmarks["onset"]],time_subset[landmarks["onset"]],label+"_ONS"]
        landmark_df.loc[len(landmark_df)] = [0,"TextTier",label,0,time_subset[landmarks["pvel_to"]],time_subset[landmarks["pvel_to"]],label+"_pVelTo"]
        landmark_df.loc[len(landmark_df)] = [0,"TextTier",label,0,time_subset[landmarks["target"]],time_subset[landmarks["target"]],label+"_TAR"]
        landmark_df.loc[len(landmark_df)] = [0,"TextTier",label,0,time_subset[landmarks["release"]],time_subset[landmarks["release"]],label+"_REL"]
        landmark_df.loc[len(landmark_df)] = [0,"TextTier",label,0,time_subset[landmarks["pvel_fro"]],time_subset[landmarks["pvel_fro"]],label+"_pVelFro"]
        landmark_df.loc[len(landmark_df)] = [0,"TextTier",label,0,time_subset[landmarks["offset"]],time_subset[landmarks["offset"]],label+"_OFF"]
    elif param["method"] == "tvel20_windowed":
        tvel = tangential_velocity(extracted_ema_data,param["channel"],ema_fs)[tmin_ema_idx:tmax_ema_idx]
        wfun = window_function(time_subset, tmin_pad, tmin-0.025, tmax+0.025, tmax_pad)
        landmarks = detect_landmarks_tvel20_windowed(time_subset,tvel,wfun,tmid_ema_idx)
        """
        fig, ax = plt.subplots(4,1,figsize=(10,10))
        ax[0].plot(time_subset,extracted_ema_data[param["channel"]+"_y"][tmin_ema_idx:tmax_ema_idx])
        ax[1].plot(time_subset,tvel)
        ax[2].plot(time_subset,wfun)
        ax[3].plot(time_subset,tvel*wfun)
        ax[0].scatter(time_subset[landmarks["onset"]],extracted_ema_data[param["channel"]+"_y"][tmin_ema_idx:tmax_ema_idx][landmarks["onset"]])
        ax[0].scatter(time_subset[landmarks["pvel_to"]],extracted_ema_data[param["channel"]+"_y"][tmin_ema_idx:tmax_ema_idx][landmarks["pvel_to"]])
        ax[0].scatter(time_subset[landmarks["target"]],extracted_ema_data[param["channel"]+"_y"][tmin_ema_idx:tmax_ema_idx][landmarks["target"]])
        ax[0].scatter(time_subset[landmarks["release"]],extracted_ema_data[param["channel"]+"_y"][tmin_ema_idx:tmax_ema_idx][landmarks["release"]])
        ax[0].scatter(time_subset[landmarks["pvel_fro"]],extracted_ema_data[param["channel"]+"_y"][tmin_ema_idx:tmax_ema_idx][landmarks["pvel_fro"]])
        ax[0].scatter(time_subset[landmarks["offset"]],extracted_ema_data[param["channel"]+"_y"][tmin_ema_idx:tmax_ema_idx][landmarks["offset"]])
        ax[0].set_title(label)
        """
        landmark_df.loc[len(landmark_df)] = [0,"TextTier",label,0,time_subset[landmarks["onset"]],time_subset[landmarks["onset"]],label+"_ONS"]
        landmark_df.loc[len(landmark_df)] = [0,"TextTier",label,0,time_subset[landmarks["pvel_to"]],time_subset[landmarks["pvel_to"]],label+"_pVelTo"]
        landmark_df.loc[len(landmark_df)] = [0,"TextTier",label,0,time_subset[landmarks["target"]],time_subset[landmarks["target"]],label+"_TAR"]
        landmark_df.loc[len(landmark_df)] = [0,"TextTier",label,0,time_subset[landmarks["release"]],time_subset[landmarks["release"]],label+"_REL"]
        landmark_df.loc[len(landmark_df)] = [0,"TextTier",label,0,time_subset[landmarks["pvel_fro"]],time_subset[landmarks["pvel_fro"]],label+"_pVelFro"]
        landmark_df.loc[len(landmark_df)] = [0,"TextTier",label,0,time_subset[landmarks["offset"]],time_subset[landmarks["offset"]],label+"_OFF"]
        
    elif param["method"] == "vel20_robust":
        trendps, detrended_data = detrend_data_1param(extracted_ema_data, time_subset, param["channel"], ema_fs, 3,tmin_ema_idx,tmax_ema_idx)
        
        vel_original = derivation(extracted_ema_data[param["channel"]],ema_fs,1)[tmin_ema_idx:tmax_ema_idx]
        vel_detrended = derivation(detrended_data[param["channel"]],ema_fs,1)
        print(vel_detrended)
        """
        fig, ax = plt.subplots(3,1,figsize=(10,10))
        ax[0].plot(time_subset,extracted_ema_data[param["channel"]][tmin_ema_idx:tmax_ema_idx])
        ax[1].plot(time_subset,vel_original)
        ax[2].plot(time_subset,vel_detrended)
        """
        
        
    return landmark_df

def prepare_landmarks(df,maxtier):
    tiers = np.unique(df["TierName"].to_numpy())
    final_df = pd.DataFrame(columns=df.columns)
    tierid = maxtier
    for tier_idx in range(len(tiers)):
        tmpdf = df[df["TierName"] == tiers[tier_idx]].reset_index(drop=True)
        tmpdf = tmpdf.sort_values(by="tmin",ascending=True)
        tierid += 1
        tmpdf["TierID"] = [tierid for i in range(len(tmpdf))]
        tmpdf["IntervallNo"] = [i+1 for i in range(len(tmpdf))]
        final_df = pd.concat([final_df,tmpdf])
    return final_df
        

def write_TG_preamble(tmax,maxTier):
    line = "File type = \"ooTextFile\"\nObject class = \"TextGrid\"\n\nxmin = 0 \nxmax = "+str(tmax)+"\ntiers? <exists> \nsize = "+str(maxTier)+"\nitem []: \n"
    return line
def write_TG_segments(df,tmax):
    line = "    item [1]:\n        class = \"IntervalTier\"\n        name = \""+df["TierName"][0]+"\"\n        xmin = 0\n        xmax = "+str(tmax)+"\n        intervals: size = "+str(len(df))+"\n"
    for i in range(len(df)):
        line += "        intervals ["+str(df["IntervallNo"][i])+"]:\n"
        line += "            xmin = "+str(df["tmin"][i])+"\n"
        line += "            xmax = "+str(df["tmax"][i])+"\n"
        if df["label"][i] == "nan":
            line += "            text = \"\"\n"
        else:
            line += "            text = \""+df["label"][i]+"\"\n"
    return line

def write_TG_points(df,tmax):
    line = "    item [2]:\n        class = \"TextTier\"\n        name = \""+df["TierName"][0]+"\"\n        xmin = 0 \n        xmax = "+str(tmax)+"\n        points: size = "+str(len(df))+"\n"
    for i in range(len(df)):
        line += "        points ["+str(df["IntervallNo"][i])+"]:\n"
        line += "            number = "+str(df["tmin"][i])+"\n"
        line += "            mark = \""+df["label"][i]+"\"\n"
    return line
        

def create_TextGrid(audio_time,df,filename,output_path):
    max_time = audio_time.max()
    df["label"] = df["label"].astype(str)
    tgname = filename+".TextGrid"
    if os.path.isfile(output_path+tgname):
        os.remove(output_path+tgname)
    tgfile = open(output_path+tgname,mode="a",encoding="utf8")
    tgfile.write(write_TG_preamble(max_time,df["TierID"].max()))
    #write segments
    tiers = np.sort(np.unique(df["TierID"].to_numpy()))
    for i in range(len(tiers)):
        tmp = df[df["TierID"] == tiers[i]].reset_index(drop=True)
        if tmp["TierClass"][0] == "IntervalTier":
            tgfile.write(write_TG_segments(tmp, max_time))
        elif tmp["TierClass"][0] == "TextTier":
            tgfile.write(write_TG_points(tmp,max_time))
    tgfile.close()

parser = argparse.ArgumentParser()
parser.add_argument("-i","--input_path",help="path to the json file")
args = parser.parse_args()

# read json config
jsonpath = args.input_path
#jsonpath = "/home/philipp/Workspace/config2.json"
with open(jsonpath,"r",encoding="utf8") as f:
    config = json.load(f,object_pairs_hook=OrderedDict)

#extract information
input_path = config["input_path"]
channels = config["channels"]
tier = config["tier"]
targets = config["targets"]
target_keys = list(targets.keys())
padding = config["padding"]
signal_filter = config["signal_filter"]
LA = config["LA"]
output_path = config["output_path"]


errors = []
#get filenames
files = os.listdir(input_path)

#filter files
wav = [files[i] for i in range(len(files)) if files[i].endswith(".wav")]
tg = [files[i] for i in range(len(files)) if files[i].endswith(".TextGrid")]
pos = [files[i] for i in range(len(files)) if files[i].endswith(".pos")]

#get only files for which  wav, pos and TextGrid files exist
common_files = get_common_files(wav_files=wav, ema_files=pos, tg_files=tg)

#create results csv if measurements should be retrieved directly

counter = 0
successful_parsing = 0
missing_gestures = 0

progressbar = tqdm.tqdm(common_files)
for filename in progressbar:
    progressbar.set_description("Parse gestures for file: "+filename)
    #load audio
    fs, s = wavfile.read(input_path+filename+".wav")
    #normalize audio
    s = s/np.max(np.abs(s))
    t = np.linspace(0,len(s)/fs,num=len(s))
    #load TextGrid
    annotation = read_textgrid(input_path+filename+".TextGrid")
    annotation_full = annotation.copy()
    maxtier = annotation["TierID"].max()
    annotation = annotation[annotation["TierID"] == tier].reset_index(drop=True)
    
    #create dataframe for saving landmarks
    landmark_df = pd.DataFrame(columns=annotation.columns)
    
    
    #load ema data
    ema_fs, ema_num_of_channels, ema_device = read_header(input_path+"/"+filename+".pos")
    if ema_device == "AG50x":
        sample_order = {"x" : 0, "z" : 1, "y" : 2, "phi" : 3, "t" : 4, "rms" : 5, "extra" : 6}
    ema_fs, ema_num_of_channels, ema_data = read_pos_file(input_path+"/"+filename+".pos")
    extracted_ema_data = extract_ema_data(data=ema_data, ema_channels=channels, sample_order=sample_order)
    if LA != "None":
        extracted_ema_data["LA"] = np.array(
                                            [ np.sqrt(
                                                        (extracted_ema_data[LA[0]+"_x"][j]-extracted_ema_data[LA[1]+"_x"][j])**2
                                                        +(extracted_ema_data[LA[0]+"_y"][j]-extracted_ema_data[LA[1]+"_y"][j])**2
                                                     )
                                                        for j in range(len(extracted_ema_data[LA[1]+"_y"]))    
                                            ] 
                                            
                                            
                                            )
        
    extracted_ema_data = smoothing(data=extracted_ema_data,signal_filter=signal_filter,ema_fs=ema_fs)
    
    time_ema = np.linspace(0,len(extracted_ema_data[list(extracted_ema_data.keys())[0]])/ema_fs,len(extracted_ema_data[list(extracted_ema_data.keys())[0]]))
    time_wav = np.linspace(0,len(s)/fs,len(s))
    
    
    for segment_idx in range(len(annotation)):
        for target_idx in range(len(target_keys)):
            if annotation["label"][segment_idx] == target_keys[target_idx]:
                try:
                    landmark_df = create_annotation(time_ema=time_ema,annotation=annotation,landmark_df=landmark_df,extracted_ema_data=extracted_ema_data,segment_idx=segment_idx,padding=padding,param=targets[target_keys[target_idx]],ema_fs=ema_fs,counter=counter)
                    successful_parsing += 1
                except:
                    errors.append([annotation["label"][segment_idx],filename,annotation["tmin"][segment_idx]])
                    missing_gestures += 1
                break
                    
    cleared_landmark_df = landmark_df.copy()
    Landmarks_textgrids_df = prepare_landmarks(cleared_landmark_df,maxtier)
    textgrid_df = pd.concat([annotation_full,Landmarks_textgrids_df])
    create_TextGrid(audio_time=t,df=textgrid_df,filename=filename,output_path=output_path)
            
#(time_ema,annotation,landmark_df,extracted_ema_data,segment_idx,padding,param,ema_fs,counter):
               
if len(errors) != 0:
    error_log = open(output_path+"error_log.txt",encoding="utf8",mode="w")
    for i in range(len(errors)):
        line = "Gesture for " + errors[i][0] + " in " + errors[i][1] + " at time: " + str(errors[i][2]) + " was not parsed.\n"
        error_log.write(line)

    
print("\n"+str(successful_parsing) + " gestures parsed, "+str(missing_gestures) + " gestures missing")
