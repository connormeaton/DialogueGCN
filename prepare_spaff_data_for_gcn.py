
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from ast import literal_eval


data = np.loadtxt('/Users/asi/connor_asi/cnn_for_100d_utterance_representation/100d_utterance_representations.csv', delimiter=',')      
df_path = glob.glob('/Users/asi/connor_asi/spaff_data/utterance_level_transcripts/*')

df_list = []
for path in df_path:
    d = pd.read_csv(path)
    df_list.append(d)

def get_indices(df_list):
    """
    Returns the split indices inside the array.
    """
    indices = [0]
    for df in df_list:
        indices.append(len(df) + indices[-1])
    return indices[1:]

# split the given arr into multiple sections.
sections = np.split(data, get_indices(df_list))
count = 0
for d, s in zip(df_list, sections):
    if len(d) == len(s):
        d['100d'] = s.tolist() # append the section of array to dataframe
        d.to_csv(f'/Users/asi/connor_asi/spaff_data/utterance_with_100d/df_{count}.csv')
        count += 1
    
new_df_path = glob.glob('/Users/asi/connor_asi/spaff_data/utterance_with_100d/*')
new_keys = []
for i in new_df_path:
    new_keys.append(i.split('/')[-1][:-4])

dict_text = {}
dict_labels = {}
dict_speaker = {}
dict_umask = {}
for i in new_keys:
    for j in new_df_path:
        if i in j:
            d = pd.read_csv(j)
            d = d[d.speaker_spaff != 7]
            d = d[d.speaker_spaff != 8]
            d = d[d.speaker_spaff != 13]
            d = d[d.speaker_spaff != 14]
            d['speaker_spaff_'] = d.speaker_spaff.astype('category').cat.codes
            min_speaker = min(d.speaker_label.values)
            def label(df, min_speaker):
                x = 0
                if df.speaker_label == min_speaker:
                    x = 0
                else:
                    x = 1
                return x
            d['speaker_label'] = d.apply(label, axis=1, args=(min_speaker,))
            
            def format(df):
                if df.speaker_label == 0:
                    x = [0,1]
                else:
                    x = [1, 0]
                return x
            d['formated_speaker_label'] = d.apply(format, axis=1)

            d['100d'] = d['100d'].apply(literal_eval)
            d['100d'] = d['100d'].apply(np.array)

            # d['speaker_spaff'] = d.speaker_spaff.apply(literal_eval)
            # d['formated_speaker_label'] = d['formated_speaker_label'].apply(literal_eval)
            dict_text[i] = d['100d'].to_list()
            dict_labels[i] = d.speaker_spaff_.values
            dict_speaker[i] = d.formated_speaker_label.to_list()
            dict_umask[i] = [1] * len(d)

list_of_dicts = [dict_text, dict_labels, dict_speaker, new_keys]


pickle.dump(list_of_dicts, open( "SPAFF_features/spaff_features.pkl", "wb" ) )
