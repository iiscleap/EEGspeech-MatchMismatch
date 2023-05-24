# -*- coding: utf-8 -*-
# @Author: aksharasoman
# @Date:   2022-03-29 10:14:25
# @Last Modified by:   aksharasoman
# @Last Modified time: 2022-11-02 21:57:21
import numpy as np
import scipy.io
import torch
import random
from config import *
from torch import round

def my_standardize(X,ax=0):
    """
    STANDARDIZING D DIMENSIONAL DATA
    """
    X_mean = np.mean(X, axis=ax)
    X = X - X_mean
    X_std = np.std(X, axis=ax)
    X = X / X_std
    return X, X_mean, X_std


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
 
#----------------------------- text network related ----------------------------------#
# convert word time boundaries to indices (time sample points)
def time_to_index(onset_time,offset_time,conv2d=False):
    # onset_time: word onset time 
    # offset_time: word offset time
    # conv2d: bool to denote if conv2d layer exist in the speech nw
    
    ref_time = onset_time[0]
    #Step 1: Subtract onset_time[0] from both onset and offset time arrays
    onset = [onset_time[k]-ref_time for k in range(len(onset_time))]
    offset = [offset_time[k]-ref_time for k in range(len(offset_time))]
    upper_index = offset_time[-1]-ref_time

    #Step 2: Multiply all elements by fs & round it
    onset_index = [round(t*fs) for t in onset]
    offset_index = [round(t*fs) for t in offset]
    upper_index = round(upper_index*fs)
    
    #Step 3: Adjust for stride and temporal kernel in the neural network
    if conv2d:
        onset_index = [int(round(((t-c2_temporal_kernelSize)/c2_temporal_stride)+1)) for t in onset_index]
        offset_index = [int(round(((t-c2_temporal_kernelSize)/c2_temporal_stride)+1)) for t in offset_index]
        upper_index = int(round(((upper_index-c2_temporal_kernelSize)/c2_temporal_stride)+1)) 
        
    # remove out of range index
    onset_index = [x if x>0 else 0 for x in onset_index]
    offset_index = [x if x<upper_index else upper_index for x in offset_index]
    
    return onset_index, offset_index


def avg_pooling(mtx,onset_index,offset_index):
    # average pool along x-axis (time): last dimension
    # mtx: input tensor
    mtx = torch.squeeze(mtx) #batch size is usually 1 
    avg_mtx = torch.zeros((mtx.shape[0],len(onset_index)))
    for k in range(len(onset_index)):
        avg_mtx[:,k]  = torch.mean(mtx[:,onset_index[k]:offset_index[k]],1) 
    return avg_mtx

# Refine text time info to filter out short words and sentences without any words
def refine_textInfo(fileId,dur_threshold=0.25):
    # Inputs:
    # dur_threshold(default:0.25s): words below this duration are removed
    # returns list of lists
    
    # Load sentence time boundaries
    text_info = scipy.io.loadmat(f'{text_path}/Run{fileId}.mat')
    sent_boundaries = np.squeeze(text_info['sentence_boundaries'])        
    onset_time = np.squeeze(text_info['onset_time']) 
    offset_time = np.squeeze(text_info['offset_time'])
    words = np.squeeze(text_info['wordVec']) # to access a p-th word: word[p][0]
    
    refined_onset_time = []
    refined_offset_time = []
    word_list = []
    # Remove sent time info without any words & arrange word time info 
    p = 0
    for k in range(len(sent_boundaries)): 
        sent_end = sent_boundaries[k]
        cur_onset_time = []
        cur_offset_time = []
        cur_word_list = []
        while(onset_time[p]<sent_end):
            dur = offset_time[p]-onset_time[p]
            if dur>dur_threshold: #to ignore short words
                cur_onset_time.append(onset_time[p])
                cur_offset_time.append(offset_time[p])
                cur_word_list.append(words[p][0])
            if p < len(onset_time)-1:
                p = p+1
            else:
                break
        if len(cur_onset_time) != 0:  #eliminate empty ones(ie., no words)
            refined_onset_time.append(cur_onset_time)
            refined_offset_time.append(cur_offset_time)
            word_list.append(cur_word_list)
    # return refined_onset_time, refined_offset_time, word_list
    np.savez(f'{text_path}/refined_Run{fileId}.npz',onset_time=refined_onset_time,offset_time=refined_offset_time,word_list=word_list)

def get_testset(n):
    # n: line number to read : corresponds to test set Id
    # n ranges from 0 to 5 
    filename = 'test_subjectSets.txt'   # replace with the name of your file

    with open(filename, 'r') as f:
        lines = f.readlines()
        if n <= len(lines):
            a = lines[n].strip()
            b = [int(x) for x in a.split()]
            return b

                
