import numpy as np
import pandas as pd
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset

import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def get_vader_feature(df):
    sid = SentimentIntensityAnalyzer()
    df['neg'] = df['text'].apply(lambda review: sid.polarity_scores(review)['neg'])
    df['pos'] = df['text'].apply(lambda review: sid.polarity_scores(review)['pos'])
    df['neu'] = df['text'].apply(lambda review: sid.polarity_scores(review)['neu'])
    df['compound'] = df['text'].apply(lambda review: sid.polarity_scores(review)['compound'])

    
class StressDataset(Dataset):
    def __init__(self, file_path, mode):
        super().__init__()
        self.mode = mode
        df = pd.read_csv(file_path, sep='\t')
        dic = {'not stress': 0, 'stress': 1}
        if mode != 'test':
            self.labels = df['label'].tolist()
        self.data = {}
        get_vader_feature(df)
        for idx, row in df.iterrows():
            if mode != 'test':
                self.data[idx] = (row['text'], row['neg'], row['neu'], row['pos'], row['compound'], row['label'])
            else:
                self.data[idx] = (row['text'], row['neg'], row['neu'], row['pos'], row['compound'])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.mode != 'test':
            text, neg, neu, pos, compound, label = self.data[idx]
            vad_score = [neg, neu, pos, compound]
            return (text, torch.tensor(vad_score), torch.tensor(label, dtype=torch.long))
        else:
            text, neg, neu, pos, compound = self.data[idx]
            vad_score = [neg, neu, pos, compound]
            return (text, torch.tensor(vad_score))

    def get_labels(self):
        return self.labels
