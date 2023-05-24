# -*- coding: utf-8 -*-
# @Author: aksharasoman
# @Date:   2022-04-15 20:49:45
# @Last Modified by:   aksharasoman
# @Last Modified time: 2022-10-09 14:14:23
import torch
import torch.nn as nn 
import torch.optim as optim
import os,sys, csv, shutil

from utils import *
from data_prep import ourDataset_eegAudio_sent, my_collate 
from config import *
from models import *
from train  import *
from pdb import set_trace as bp
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(random_seed)
if not os.path.exists('removables'):
    os.makedirs('removables')
shutil.copy2('config.py','removables')

sub_set = int(sys.argv[1])

if len(test_fileList) == num_files:
    train_fileList = test_fileList
else:
    train_fileList = list(set(np.arange(0,num_files)) - set(test_fileList))# rest of the files as training set

test_subList = get_testset(sub_set)
train_subList = list(set(np.arange(1,num_subjects+1)) - set(test_subList))# rest of the files as training set

# if sub_set == 0:
#     test_subList = [1,6,14]  #all files of 3 subjects as test  #NB: subList start from 1
#     train_subList = list(set(np.arange(1,num_subjects+1)) - set(test_subList))# rest of the files as training set
# elif sub_set == 1:
#     test_subList = [2,12,18]
#     train_subList = list(set(np.arange(1,num_subjects+1)) - set(test_subList))# rest of the files as training set
# elif sub_set == 2:
#     test_subList = [5,9,10]
#     train_subList = list(set(np.arange(1,num_subjects+1)) - set(test_subList))# rest of the files as training set
# elif sub_set == 100: #all subjects in both test and train
#     test_subList = list(np.arange(1,num_subjects+1)) # all subjects
#     train_subList = test_subList

print('Train subjects set: ',train_subList)
print('Test subjects set: ',test_subList)
print('Train files are: ',train_fileList)
print('Test files are: ',test_fileList)

result_file = f'set{sub_set}'

print('----------------')
field_names =['Epoch','train_loss', 'test_loss','train_acc','test_acc']
csvfile = open(f'removables/result_{result_file}.csv', 'w')
writer = csv.DictWriter(csvfile, fieldnames = field_names)
writer.writeheader()
# ---------------- Data ---------------- #
train_dataset = ourDataset_eegAudio_sent(train_fileList,train_subList, eeg_path, speech_path,n_melFilt)
test_dataset = ourDataset_eegAudio_sent(test_fileList,test_subList, eeg_path, speech_path,n_melFilt)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True,collate_fn=my_collate,drop_last=True)
test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=len(test_dataset), collate_fn=my_collate,drop_last=True)

# ---------------- Model Definition ---------------- #
# Model definition
model = model_audio_eeg() #for sentence level
model = model.to(device) 

# pretrained_model = torch.load('../model_set0_frame5.mdl')
# model.speech_nw.conv1d = pretrained_model.speech_nw.conv1d
# del pretrained_model

# Loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=reg_par)
# scheduler = optim.lr_scheduler.StepLR(optimizer, 
#                    step_size = 4, # Period of learning rate decay
#                    gamma = 0.5) # Multiplicative factor of learning rate decay

history = {'train_loss': [], 'test_loss': [],'train_acc':[],'test_acc':[]}
# ---------------- Training ---------------- #
save_out = False # true only for last epoch
for epoch in range(num_epochs):
    print(f'## Epoch:{epoch}')
    train_loss, train_acc=train_epoch(model,device,train_loader,criterion,optimizer)
    print('Validation:')
    if epoch == num_epochs-1:
        save_out =  True
    test_loss, test_acc =valid_epoch(model,device,test_loader,criterion,result_file,save_out)
    
    
    print("Epoch:{}/{} ; AVG Training Loss:{:.3f} ; AVG Test Loss:{:.3f} ; AVG Training Acc: {:.2f} % ; AVG Test Acc: {:.2f} %".format(epoch + 1,
                                                                                                            num_epochs,
                                                                                                            train_loss,
                                                                                                            test_loss,
                                                                                                            train_acc,
                                                                                                            test_acc))
    history['Epoch'] = epoch
    history['train_loss'] = (train_loss)
    history['test_loss'] = (test_loss)
    history['train_acc'] = train_acc.item()*100
    history['test_acc'] = test_acc.item()*100
    writer.writerow(history)
torch.save(model,f'removables/model_{result_file}.mdl')    #saving model
csvfile.close()


