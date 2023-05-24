# -*- coding: utf-8 -*-
# @Author: aksharasoman
# @Date:   2022-04-15 10:38:49
# @Last Modified by:   aksharasoman
# @Last Modified time: 2022-04-23 11:14:36
import torch
from torchmetrics.classification import BinaryAccuracy, BinaryStatScores
from torchmetrics import PearsonCorrCoef

# MODIFICATION: to receive edited dataset 
def train_epoch(model,device,dataloader,loss_fn,optimizer):
    running_train_loss,running_train_acc=0.0,0
    metric = BinaryAccuracy()
    confusion_mtx = BinaryStatScores()
    model.train()
    print()
    for i, data in enumerate(dataloader,0):
        # get the inputs; data is a list of [inputs, labels]
        eeg_feat, eeg_numWords, eeg_time, speech_feat, speech_numWords, speech_time, labels = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device), data[5].to(device), data[6].to(device)
        
        # del data
        # zero the parameter gradients
        optimizer.zero_grad()       
            
        # forward + backward + optimize
        y_speech,y_eeg,output = model(eeg_feat, eeg_numWords, eeg_time, speech_feat, speech_numWords, speech_time)
        del speech_feat, eeg_feat
        
        output = torch.squeeze(output)
        labels = labels.to(torch.float32)
        loss = loss_fn(output,labels)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()
        output = output.detach().cpu()
        labels = labels.cpu()
        train_acc = metric(output,labels)  #uses 0.5(default) Threshold for transforming probability to binary {0,1} predictions
        running_train_acc += train_acc
        confMtx = confusion_mtx(output,labels) #Computes the number of true positives, false positives, true negatives, false negatives and the support for binary tasks.

        if i%10 == 0:
            print(f'Iteration: {i}; TP: {confMtx[0]}; FP: {confMtx[1]}; TN: {confMtx[2]}; FN: {confMtx[3]}; Training Acc: {train_acc*100:.2f}; Loss: {loss.item():.4f} ')
    num_batches = len(dataloader)
    avg_loss = running_train_loss/num_batches
    avg_train_acc = running_train_acc/num_batches
    return avg_loss,avg_train_acc
  
def valid_epoch(model,device,dataloader,loss_fn,result_file,save_out=False):
    running_val_loss, running_val_acc = 0.0, 0
    metric = BinaryAccuracy()
    confusion_mtx = BinaryStatScores()
    pearson = PearsonCorrCoef(num_outputs=32)
    model.eval()
    
    # to save output values
    target = []
    predicted_prob = []
    speech_embedding = []
    eeg_embedding = []
    for i,data in enumerate(dataloader,0):

        # get the inputs; data is a list of [eeg_feat, speech_feat,label]
        eeg_feat, eeg_numWords, eeg_time, speech_feat, speech_numWords, speech_time, labels = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device), data[5].to(device), data[6].to(device)
        del data

        # forward 
        y_speech, y_eeg, output = model(eeg_feat, eeg_numWords, eeg_time, speech_feat, speech_numWords, speech_time)
        del speech_feat, eeg_feat
        output = torch.squeeze(output)
        labels = labels.to(torch.float32)
        loss = loss_fn(output,labels)
        
        running_val_loss +=loss.item()
        output = output.detach().cpu()
        labels = labels.cpu()
        val_acc = metric(output,labels)  #uses 0.5(default) Threshold for transforming probability to binary {0,1} predictions
        running_val_acc += val_acc
        confMtx = confusion_mtx(output,labels) #Computes the number of true positives, false positives, true negatives, false negatives and the support for binary tasks.
        print(f'TP: {confMtx[0]}; FP: {confMtx[1]}; TN: {confMtx[2]}; FN: {confMtx[3]}; Val Acc: {val_acc}; Loss: {loss.item()} ')
        
        if save_out:
            #save output values
            target.append(labels) #original labels
            predicted_prob.extend(output)
            speech_embedding.append(y_speech.detach().cpu())
            eeg_embedding.append(y_eeg.detach().cpu())
    
    # save output values
    if save_out:
        torch.save({'target':target,'predicted_prob':predicted_prob,'speech_embedding':speech_embedding,'eeg_embedding':eeg_embedding},f'removables/outputVals_{result_file}.pt')
    num_batches = len(dataloader)
    avg_loss = running_val_loss/num_batches
    avg_val_acc = running_val_acc/num_batches
    return avg_loss,avg_val_acc


   