# -*- coding: utf-8 -*-
# @Author: aksharasoman
# @Date:   2022-03-28 12:42:07
# @Last Modified by:   aksharasoman
# @Last Modified time: 2022-10-31 20:05:54

''' Stores hyperparameters, system paths '''
import torch
''' For EdLalor_NaturalSpeechDataset '''

##----->DATASET
num_subjects = 19 # 1 to 19
num_files = 20 #number of stimuli files 20
num_eeg_channels = 128
num_classes = 2 # speech (=1) & silence (=0)
test_fileList = list(range(0,num_files)) # all files: SI config
# test_fileList = [4,8,12]

random_seed = 42 # manual seed for ensuring reproducability
##-----> FEATURE GENERATION
n_melFilt = 28 # number of mel filters 
fs = 64 # sampling frequency of mel features [MEL STFT window width & stride are chosen accordingly]
    # Decision Window of features
#frame_length = 3 
# frame_step = frame_length/10 # 10% of frame width => 90% overlap
##-----> Training HYPERPARAMETERS
num_epochs  = 20# 30
learning_rate = 1e-3
batch_size = 32
reg_par    = 1e-4

#----------------------------------------------------------------#
##-----> NN Model Parameters
input_feature_size = 28
c1_spatial_kernelSize=8
c2_numFilters =16
c2_temporal_kernelSize=9
c2_temporal_stride = 3
lstm_size=32
device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
drop = 0.2 #dropout prob.

# #-----> EEG network
td1_size=128
td2_size=32
#------> Common layers
#fc_input_size = (frame_length * fs - c2_temporal_kernelSize)//c2_temporal_stride + 1
#fc_input_size = 104 # input nodes in final fully connected layer
    # NB: this will change if #input time samples (=320) change. 
# ----------------------------------------------------------------#
## PATHS : Comment non-relevant locations

##-----> CLUSTER
eeg_path = '/home/aksharas/datasets/speech_eeg_giovanni/dataCND'
speech_path = '/home/aksharas/datasets/speech_eeg_giovanni/raw_audio_files_16khz'
text_path = '/home/aksharas/datasets/speech_eeg_giovanni/Text'
w2vModel_file = '/home/aksharas/datasets/pretrained_models/word2vec_models/GoogleNews-vectors-negative300.bin'
##-----> COMPUTE CANADA
# eeg_path = '/home/ojus001/data_aks/LalorNatSpeech/dataCND/'
# speech_path = '/home/ojus001/data_aks/raw_audio_files_16khz'
# text_path = '/home/ojus001/data_aks/LalorNatSpeech/Stimuli/Text'
# w2vModel_file = '/home/ojus001/data_aks/pretrained_models/word2vec_models/GoogleNews-vectors-negative300.bin'
##-----> macbook 
# eeg_path = '/Users/aksharas/Resources/datasets/naturalspeech_eeg_lalor/dataCND'
# speech_path = '/Users/aksharas/Resources/datasets/naturalspeech_eeg_lalor/Stimuli/raw_audio_files_16khz'
# text_path = '/Users/aksharas/Resources/datasets/naturalspeech_eeg_lalor/Stimuli/Text'
