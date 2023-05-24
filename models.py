# -*- coding: utf-8 -*-
# @Author: aksharasoman
# @Date:   2022-03-29 10:12:53
# @Last Modified by:   aksharasoman
# @Last Modified time: 2022-08-30 11:32:05
import torch.nn as nn
import torch
from config import *
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence 
import utils

# LSTM based model for Speech data (KU Leuven paper)
# Input: 28 x 320(time=5s decision window) 
# without subnetwork correction
class lstmModel(nn.Module):
    def __init__(self,input_feature_size=28,c1_spatial_kernelSize=8,c2_numFilters=16, c2_temporal_kernelSize=9,c2_temporal_stride=3,lstm_size=32,drop=0.1):
        super(lstmModel, self).__init__()
        self.conv1d = nn.Conv1d(input_feature_size, out_channels=c1_spatial_kernelSize, kernel_size=1)
        self.conv2d = nn.Conv2d(1,c2_numFilters,(1,c2_temporal_kernelSize),stride=(1,c2_temporal_stride))
        self.relu = nn.ReLU()
        self.BN = nn.BatchNorm2d(c2_numFilters)
        self.lstm_input_size = c1_spatial_kernelSize*c2_numFilters 
        self.lstm = nn.LSTM(input_size=self.lstm_input_size,hidden_size=lstm_size,batch_first=True)
        self.drp = nn.Dropout(p=drop)

    def forward(self, x, s,word_onset_time, word_offset_time):
        y = self.conv1d(x)
        y = self.relu(y)
        y = self.drp(y)

        y = torch.unsqueeze(y,1)
        y = self.conv2d(y)
        y = self.BN(y)
        y = self.relu(y)
        y = self.drp(y)
        y = y.reshape(y.shape[0],self.lstm_input_size,-1) # shape: (N,128,L)

        # take word boundaries and avg time pool
        # Obtain feat_dim x num_words representation
            # convert time to index
        pool_y = []
        for k in range(len(y)):
            embedding = y[k] #feat is numpy array
            nw = s[k].item()
            ont = word_onset_time[k][:nw]
            oft = word_offset_time[k][:nw]
            onset_index, offset_index = utils.time_to_index(ont,oft,True) #True: conv2d layer in speech nw
            # average pool along x-axis (time)
            pool_y.append(torch.t(utils.avg_pooling(embedding,onset_index,offset_index)))
        # pad sequence
        pool_y = pad_sequence(pool_y, batch_first=True)
        s = s.cpu()
        y_pack = pack_padded_sequence(pool_y, s, batch_first=True, enforce_sorted=False)
        y_pack = y_pack.to(device)
        
        y,(ht,ct) = self.lstm(y_pack)
        out = ht[-1] # returning last hidden state instead of output
        return out

#### upper part of network dealing with EEG.
class tdModel(nn.Module):
    def __init__(self,num_eeg_channels=64,c1_spatial_kernelSize=32,c2_numFilters=16, c2_temporal_kernelSize=9,c2_temporal_stride=3,td1_size=128,td2_size=32,drop=0.1):
        super(tdModel,self).__init__()
        self.conv1d = nn.Conv1d(num_eeg_channels,out_channels=c1_spatial_kernelSize,  kernel_size=1)
        self.conv2d = nn.Conv2d(1,c2_numFilters,(1,c2_temporal_kernelSize),stride=(1,c2_temporal_stride))
        self.relu = nn.ReLU()
        self.BN = nn.BatchNorm2d(c2_numFilters)
        self.input_size = c1_spatial_kernelSize*c2_numFilters 
        self.tdd1 = nn.Conv1d(self.input_size,td1_size,1) #kernel size 1 ensures dense layer
        self.tdd2 = nn.Conv1d(td1_size,td2_size,1)
        self.tanh = nn.Tanh()
        self.drp = nn.Dropout(p=drop)

    def forward(self, x):
        y = self.conv1d(x)
        # y = self.relu(y) #no activation applied after conv1D
        y = torch.unsqueeze(y,1)
        y = self.conv2d(y)
        y = self.relu(y)
        y = self.drp(y)
        y = self.BN(y)

        y = y.reshape(y.shape[0],self.input_size,-1)
        y = self.tdd1(y)
        y = self.tanh(y)
        y = self.drp(y)
        # y = self.BN(y) # expects 4D input ; so use batchNorm1D if reqd

        y = self.tdd2(y)
        y = self.tanh(y)
        y = self.drp(y)
        # y = self.BN(y)
        return y

class tdModel_conv1d(nn.Module):
    def __init__(self,num_eeg_channels=64,c1_spatial_kernelSize=32, td1_size=128,td2_size=32,drop=0.1):
        super(tdModel,self).__init__()
        self.conv1d = nn.Conv1d(num_eeg_channels,out_channels=c1_spatial_kernelSize,  kernel_size=1)
        # self.conv2d = nn.Conv2d(1,c2_numFilters,(1,c2_temporal_kernelSize),stride=(1,c2_temporal_stride))
        self.relu = nn.ReLU()
        # self.BN = nn.BatchNorm2d(c2_numFilters)
        self.input_size = c1_spatial_kernelSize
        self.tdd1 = nn.Conv1d(self.input_size,td1_size,1) #kernel size 1 ensures dense layer
        self.tdd2 = nn.Conv1d(td1_size,td2_size,1)
        self.tanh = nn.Tanh()
        self.drp = nn.Dropout(p=drop)

    def forward(self, x):
        y = self.conv1d(x)
        y = self.relu(y) 
        # y = torch.unsqueeze(y,1)
        # y = self.conv2d(y)
        # y = self.relu(y)
        y = self.drp(y)
        # y = self.BN(y)

        # y = y.reshape(y.shape[0],self.input_size,-1) #input size: [32,time_samples]
        y = self.tdd1(y)
        y = self.tanh(y)
        y = self.drp(y)
        # y = self.BN(y) # expects 4D input ; so use batchNorm1D if reqd

        y = self.tdd2(y)
        y = self.tanh(y)
        y = self.drp(y)
        # y = self.BN(y)
        return y

### text sub-network
class textnw(nn.Module):
    def __init__(self,inp1=300,fc_out1=120,fc_out2=60,fc_out3=32,c2_temporal_kernelSize=9,c2_temporal_stride=3):
        super(textnw,self).__init__()
        self.fc1  = nn.Linear(inp1,fc_out1)
        self.fc2  = nn.Linear(fc_out1,fc_out2)
        self.fc3  = nn.Linear(fc_out2,fc_out3)
        self.relu = nn.ReLU()

    def forward(self,x):
        # x's dimension: (inp1,nW) where nW is variable (nW: number of words)
        x = x.permute(0,2,1)
        y = self.relu(self.fc1(x))
        y = self.relu(self.fc2(y))
        y = self.relu(self.fc3(y))
        y = y.permute(0,2,1)

        return y
# lstm based text sub-network 
class lstm_textnw(nn.Module):
    def __init__(self,inp_size=300,drp_text=0.2):
        super(lstm_textnw,self).__init__()
        # self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.BN = nn.BatchNorm1d(inp_size)
        self.lstm = nn.LSTM(inp_size, lstm_size, batch_first=True,num_layers = 2, dropout=drp_text) #multi-layer lstm
        
    def forward(self, x, s):
        # x = self.embeddings(x)
        x = self.BN(x) # x.shape: [N,C,L]
        x = x.permute(0,2,1) # lstm needs seq. as 2nd dim=> shape: (N,L,C)
        s = s.cpu()
        x_pack = pack_padded_sequence(x, s, batch_first=True, enforce_sorted=False)
        out_pack, (ht, ct) = self.lstm(x_pack)
        # out = self.linear(ht[-1])
        out = ht[-1]
        return out

#Joint network
# speech nw without conv2d layer: existent layers initialised with pretrained model's weights
class model_text_conv1dSpeechLSTM(nn.Module):
    def __init__(self):
        super(model_text_conv1dSpeechLSTM, self).__init__()
        self.text_nw = textnw()
        self.speech_nw = conv1d_lstmModel(input_feature_size,c1_spatial_kernelSize,lstm_size,drop)
        
        # common layers
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, text_feat, speech_feat,sent_numWords):
        y_text = self.text_nw(text_feat)
        y_speech = self.speech_nw(speech_feat)

        ### Common layers
        ## 1. dot product along the feature dimension
        # normalize vectors
        y_text_normlized = nn.functional.normalize(y_text,p=2,dim=1) 
        y_speech_normlized = nn.functional.normalize(y_speech,p=2,dim=1)
        # cosine similarity score
        cos_score = (y_text_normlized*y_speech_normlized).sum(dim=1) # output shape: (N,max_numWords)
        
        ## 2. Compute mean along words: to find sentence level decision
                # (didn't use fc layer due to varying num_words in each sent)
        sum_score = torch.sum(cos_score,dim=1) 
        mean_score = torch.div(sum_score,sent_numWords)
        
        # 3. min-max scaler (map to 0-1 raange)
        mn = min(mean_score)
        mx = max(mean_score)
        out_prob = (mean_score-mn)/(mx-mn)

        return y_text, y_speech, out_prob

#Joint network
class model_text_eeg(nn.Module):
    def __init__(self):
        super(model_text_eeg, self).__init__()
        self.textnw = lstm_textnw()
        
        # eeg network (Not pre-trained; same as speech nw)
        self.eeg_nw = lstmModel(num_eeg_channels,c1_spatial_kernelSize,c2_numFilters, c2_temporal_kernelSize,c2_temporal_stride,lstm_size,drop)

    def forward(self, text_feat, eeg_feat,sent_numWords, word_onset_time, word_offset_time):
        y_text = self.textnw(text_feat,sent_numWords)
        y_eeg = self.eeg_nw(eeg_feat,sent_numWords,word_onset_time, word_offset_time) # using pre-trained model

        ### Common layers
        ## 1. Manhattan Distance
        diff = y_text - y_eeg
        dist = torch.linalg.norm(diff,ord=1,dim=1)
        out_prob = torch.exp(-dist)
        #2. fc layer & sigmoid (apply non-normalized vectors)
        # out_prob = self.sigmoid(self.fc(score))
        # out_modf = self.w * out_prob + self.b

        return y_text, y_eeg, out_prob

class model_audio_eeg(nn.Module):
    def __init__(self):
        super(model_audio_eeg, self).__init__()
        # self.textnw = lstm_textnw()
        self.audio_nw = lstmModel(input_feature_size,c1_spatial_kernelSize,c2_numFilters, c2_temporal_kernelSize,c2_temporal_stride,lstm_size,drop)

        # eeg network (Not pre-trained; same as speech nw)
        self.eeg_nw = lstmModel(num_eeg_channels,c1_spatial_kernelSize,c2_numFilters, c2_temporal_kernelSize,c2_temporal_stride,lstm_size,drop)

    def forward(self, eeg_feat, eeg_numWords, eeg_time, speech_feat, speech_numWords, speech_time):
        # y_text = self.textnw(text_feat,sent_numWords)
        word_onset_time, word_offset_time = speech_time[:,:,0], speech_time[:,:,1] # 1st column as onset_time
        y_speech = self.audio_nw(speech_feat,speech_numWords,word_onset_time, word_offset_time) 

        word_onset_time, word_offset_time =  eeg_time[:,:,0], eeg_time[:,:,1]
        y_eeg = self.eeg_nw(eeg_feat,eeg_numWords,word_onset_time, word_offset_time) 
        
        ### Common layers
        ## 1. Manhattan Distance
        diff = y_speech - y_eeg
        dist = torch.linalg.norm(diff,ord=1,dim=1)
        out_prob = torch.exp(-dist)

        return y_speech, y_eeg, out_prob


