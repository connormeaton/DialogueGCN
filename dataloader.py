import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle
import pandas as pd
import numpy as np
import glob

class IEMOCAPDataset(Dataset):

    def __init__(self, train=True):
        # this is for IEMOCAP
        # self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
        # self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        # self.testVid = pickle.load(open('./IEMOCAP_features/IEMOCAP_features.pkl', 'rb'), encoding='latin1')
        # # '''
        # # label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        # # '''
        # self.keys = [x for x in (self.trainVid if train else self.testVid)]
        # self.len = len(self.keys)

        # this is for spaff
        self.videoText, self.videoLabels, self.videoSpeakers, self.keys = pickle.load(open("SPAFF_features/spaff_features.pkl", "rb" ))
        self.len = len(self.keys)
        

    def __getitem__(self, index):
        '''
        Inserting randomized tensor in place of OG textf of size (len_convo, 100)
        and it works! 

        Now go back to CNN and get output in shape (len_conv, 100)

        CNN output successfully fit in
        - update labels
        qmask - 0,1 male or female
        umask - list of 1's len labels
        label - int label
        '''
        vid = self.keys[index]
   
        # this is for IEMOCAP
        # return torch.FloatTensor(self.videoText[vid]),\
        #        torch.FloatTensor(self.videoVisual[vid]),\
        #        torch.FloatTensor(self.videoAudio[vid]),\
        #        torch.FloatTensor([[1,0] if x=='M' else [0,1] for x in\
        #                           self.videoSpeakers[vid]]),\
        #        torch.FloatTensor([1]*len(self.videoLabels[vid])),\
        #        torch.LongTensor(self.videoLabels[vid]),\
        #        vid

        # this is for SPAFF
        return torch.FloatTensor(self.videoText[vid]),\
            torch.FloatTensor(self.videoSpeakers[vid]), \
            torch.FloatTensor([1]*len(self.videoLabels[vid])),\
            torch.LongTensor(self.videoLabels[vid]),\
            vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)

        # IEMOCAP
        # return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True) if i<6 else dat[i].tolist() for i in dat]
        # SPAFF
        return [pad_sequence(dat[i]) if i<2 else pad_sequence(dat[i], True) if i<4 else dat[i].tolist() for i in dat]


class AVECDataset(Dataset):

    def __init__(self, path, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
            self.videoAudio, self.videoVisual, self.videoSentence,\
            self.trainVid, self.testVid = pickle.load(open(path, 'rb'),encoding='latin1')

        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoText[vid]),\
               torch.FloatTensor(self.videoVisual[vid]),\
               torch.FloatTensor(self.videoAudio[vid]),\
               torch.FloatTensor([[1,0] if x=='user' else [0,1] for x in\
                                  self.videoSpeakers[vid]]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.FloatTensor(self.videoLabels[vid])

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True) for i in dat]


class MELDDataset(Dataset):

    def __init__(self, path, classify, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabelsEmotion, self.videoText,\
        self.videoAudio, self.videoSentence, self.trainVid,\
        self.testVid, self.videoLabelsSentiment = pickle.load(open(path, 'rb'))

        if classify == 'emotion':
            self.videoLabels = self.videoLabelsEmotion
        else:
            self.videoLabels = self.videoLabelsSentiment
        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''
        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoText[vid]),\
               torch.FloatTensor(self.videoAudio[vid]),\
               torch.FloatTensor(self.videoSpeakers[vid]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.LongTensor(self.videoLabels[vid]),\
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<3 else pad_sequence(dat[i], True) if i<5 else dat[i].tolist() for i in dat]


class DailyDialogueDataset(Dataset):

    def __init__(self, split, path):
        
        self.Speakers, self.Features, \
        self.ActLabels, self.EmotionLabels, self.trainId, self.testId, self.validId = pickle.load(open(path, 'rb'))
        
        if split == 'train':
            self.keys = [x for x in self.trainId]
        elif split == 'test':
            self.keys = [x for x in self.testId]
        elif split == 'valid':
            self.keys = [x for x in self.validId]

        self.len = len(self.keys)

    def __getitem__(self, index):
        conv = self.keys[index]
        
        return  torch.FloatTensor(self.Features[conv]), \
                torch.FloatTensor([[1,0] if x=='0' else [0,1] for x in self.Speakers[conv]]),\
                torch.FloatTensor([1]*len(self.EmotionLabels[conv])), \
                torch.LongTensor(self.EmotionLabels[conv]), \
                conv

    def __len__(self):
        return self.len
    
    def collate_fn(self, data):
        dat = pd.DataFrame(data)

        return [pad_sequence(dat[i]) if i<2 else pad_sequence(dat[i], True) if i<4 else dat[i].tolist() for i in dat]
