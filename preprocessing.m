% /**
%  * @Author: aksharasoman
%  * @Date:   2022-01-08 01:59:18
%  * @Last Modified by:   aksharasoman
%  * @Last Modified time: 2022-02-01 14:13:06
%  */

% Preprocessing code for EEG from lalor_naturalSpeech dataset
% Author: Akshara Soman
% Created on: Jan 8 2022
% Adapted from CNSP-workshop2021_code/CNSP_tutorial1/CNSP_example1.m

clear all
close all
addpath /Users/aksharas/Resources/CNSPworkshop2021/CNSP-workshop2021_code/tutorials/CNSP_tutorial1/libs/cnsp_utils
addpath /Users/aksharas/Resources/CNSPworkshop2021/CNSP-workshop2021_code/tutorials/CNSP_tutorial1/libs/cnsp_utils/cnd
addpath /Users/aksharas/Resources/CNSPworkshop2021/CNSP-workshop2021_code/tutorials/CNSP_tutorial1/libs/eeglab
% eeglab;
% close gcf;
%% Parameters - Natural speech listening experiment
% dataMainFolder = '/Users/aksharas/Resources/datasets/naturalspeech_eeg_lalor/';
dataMainFolder = '/Users/aksharas/Resources/datasets/speech_eeg_lalor/CocktailParty/';
dataCNDSubfolder = 'dataCND/';

reRefType = 'Mastoids'; % 'Avg' or 'Mastoids'
bandpassFilterRange = [0.5, 32]; % Hz (indicate 0 to avoid running the low-pass
                          % or high-pass filters or both)
                          % e.g., [0,8] will apply only a low-pass filter
                          % at 8 Hz
downFs = 64; % Hz. *** fs/downFs must be an integer value ***

eegFilenames = dir([dataMainFolder,dataCNDSubfolder,'dataSub*.mat']);
nSubs = length(eegFilenames);

%% Preprocess EEG - Natural speech listening experiment
for sub = 1:nSubs
    % Loading EEG data
    eegFilename = [dataMainFolder,dataCNDSubfolder,eegFilenames(sub).name];
    disp(['Loading EEG data: ',eegFilenames(sub).name])
    load(eegFilename,'eeg')
    eeg = cndNewOp(eeg,'Load'); % Saving the processing pipeline in the eeg struct

    % Filtering - LPF (low-pass filter)
    if bandpassFilterRange(2) > 0
        hd = getLPFilt(eeg.fs,bandpassFilterRange(2));
%         Filtering each trial/run with a cellfun statement
        eeg.data = cellfun(@(x) filtfilthd(hd,x),eeg.data,'UniformOutput',false);
        
        % Filtering each trial/run with a for loop
%         for ii = 1:length(eeg.data)
%                 eeg.data{ii} = filtfilthd(hd,eeg.data{ii});
%         end
        
        % Filtering external channels
        if isfield(eeg,'extChan')
            for extIdx = 1:length(eeg.extChan)
                eeg.extChan{extIdx}.data = cellfun(@(x) filtfilthd(hd,x),eeg.extChan{extIdx}.data,'UniformOutput',false);
            end
        end
        
        eeg = cndNewOp(eeg,'LPF_0.5Hz');
    end
    
    % Filtering - HPF (high-pass filter)
    if bandpassFilterRange(1) > 0 
        hd = getHPFilt(eeg.fs,bandpassFilterRange(1));
        
        % Filtering EEG data
        eeg.data = cellfun(@(x) filtfilthd(hd,x),eeg.data,'UniformOutput',false);
        
        % Filtering external channels
        if isfield(eeg,'extChan')
            for extIdx = 1:length(eeg.extChan)
                eeg.extChan{extIdx}.data = cellfun(@(x) filtfilthd(hd,x),eeg.extChan{extIdx}.data,'UniformOutput',false);
            end  
        end
        
        eeg = cndNewOp(eeg,'HPF_32Hz');
    end
    
   % Downsampling EEG and external channels
    eeg = cndDownsample(eeg,downFs);
%     stim{1} = cndDownsample(stim{1},downFs);
    eeg = cndNewOp(eeg,'Downsample_from128to64Hz');
    
    % Replacing bad channels
    if isfield(eeg,'chanlocs')
        for tr = 1:length(eeg.data)
            eeg.data{tr} = removeBadChannels(eeg.data{tr}, eeg.chanlocs);
        end
    end
    
    % Re-referencing EEG data
    eeg = cndReref(eeg,reRefType);
    
    % Removing initial padding (specific to this dataset)
    if isfield(eeg,'paddingStartSample')
        for tr = 1:length(eeg.data)
            eeg.data{tr} = eeg.data{tr}(eeg.paddingStartSample,:);
            for extIdx = 1:length(eeg.extChan)
                eeg.extChan{extIdx}.data = eeg.extChan{extIdx}.data{tr}(1+eeg.paddingStartSample,:);
            end
        end
    end
    
    % Compute z-score
    eeg.data = cellfun(@(x) zscore(x) , eeg.data, 'UniformOutput', false); %This will normalize each channel and each epoch independently.
    eeg = cndNewOp(eeg,'z-score');

    
    % Truncate EEG response to match the sample length of corresponding
    % speech stimulus
     stimFilename = [dataMainFolder,dataCNDSubfolder,'dataStim.mat'];
     load(stimFilename,'stim');
     stim = cndDownsample(stim,downFs);
     
     for K=1:length(stim.data)
         env = stim.data{1,K};
         T = size(env,1);
         if  size(eeg.data{K},1) >= T
             eeg.data{K} = eeg.data{K}(1:T,:);
         else
             disp(K);
             Te = size(eeg.data{K},1);
              eeg.data{K}(Te+1:T,:) = zeros(T-Te,size(eeg.data{K},2)); % padding with zeros
         end
     end
     eeg = cndNewOp(eeg,'TruncateEEG_toMatchSamplesLength');

    % Saving preprocessed data
    eegPreFilename = [dataMainFolder,dataCNDSubfolder,'pre_',eegFilenames(sub).name];
    disp(['Saving preprocessed EEG data: pre_',eegFilenames(sub).name])
    save(eegPreFilename,'eeg')
end
