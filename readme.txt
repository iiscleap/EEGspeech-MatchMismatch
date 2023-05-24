change in how we select mismatched features:
- choose a random sentence from the same audio file.
Duration of this sentence (speech) does not matter as we output embedding is last_hidden_state of LSTM- independent of input duration.
-> to do this, we need to feed onset and offset times (& num words) of both stim and resp to the model. 

Edited: data_prep.py/ourDataset_eegAudio_sent(), collate_fn and 2 more related files (removed mismatch mtx returns)
        & train.py/train_epoch()- received dataset arguments changed
        & models.py/model_audio_eeg() -forward fn arguments changed.
#------------------- 
Common layers modification: computing manhattan distance between EEG and text representations (instead of xln& fc layer)
exp(-|hT(egg)-hT(text)|) ; |.| is l1 norm.
Based on paper: AAAI 2016 Siamese nw sentence similarity pr

NB: modification (as in 3_) - scaling sigmoid input with learned coeff is not performed.
Analysis done on sentence level.

Text network: lstm
Both text and eeg networks are trained with a common loss function (BCE).
EEG network modified: using speech network architecture with c1_spatial_filters = 8

#----------- old ------------#
basic model that can vary frame length
Basic model: exp311 with dropout = 0.2

HOW TO RUN
python main.py <sub_set> #sub_set=0,1 or 2

variables used are specified in "settings.py"
