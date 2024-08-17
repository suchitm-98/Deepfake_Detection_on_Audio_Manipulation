# Program to convert train flac files into spectrogram images

import pandas as pd
import librosa as librosa
import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
import librosa.display
import os
import soundfile as sf

#######################################################

def mel_spect(audio_file,Path):
    file_name = audio_file.split('.')[0]
    #extension = audio_file.split('.')[1]
    pos = file_name.rfind('/')
    file_name = file_name[(pos+1) : ]
    n_fft=2048
    hop_length=512
    y,sr = sf.read(audio_file)
    spectrogram_librosa = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window='hann')) ** 2
    spectrogram_librosa_db = librosa.power_to_db(spectrogram_librosa, ref=np.max)
    plt.figure(figsize=(8, 4))
    plt.axis('off')
    librosa.display.specshow(spectrogram_librosa_db, sr=sr, y_axis='mel', x_axis='time', hop_length=hop_length,fmax=8000)
    plt.gray()
    plt.tight_layout()
    plt.savefig('%s%s.png'%(Path, file_name ))
    plt.show()

#######################################################
    
metadata = pd.read_csv('Dataset/Metadata_train_PA.csv')
metadata = metadata.drop(columns = 'Unnamed: 0')

real_df = metadata[metadata['label'] == 'bonafide']
real_df = real_df.reset_index()
real_final = real_df.iloc[ : , 1]

fake_df = metadata[metadata['label'] == 'spoof']
fake_df = fake_df.reset_index()
fake_final = fake_df.iloc[ : , 1 ]

del metadata,fake_df, real_df

########################################################

DATA_FOLDER = 'Dataset/'
TRAIN_FOLDER = 'train_flac/'
OUTPUT_TRAIN_REAL_IMG = 'train_img/real/'
OUTPUT_TRAIN_FAKE_IMG = 'train_img/fake/'

########################################################

train_img_real = os.path.join(DATA_FOLDER, OUTPUT_TRAIN_REAL_IMG)

count = 0
real_list_of_flacs = []
for flac in real_final:
    count+=1
    flac = flac+'.flac'
    flac_train = os.path.join(DATA_FOLDER,TRAIN_FOLDER)+flac
    real_list_of_flacs.append(flac_train) 
    if (count== 5000):
        break
del flac, flac_train, count

for file in real_list_of_flacs:    
    mel_spect(file,train_img_real)    
del file

########################################################

train_img_fake = os.path.join(DATA_FOLDER, OUTPUT_TRAIN_FAKE_IMG)

count = 0
fake_list_of_flacs =[]
for flac in fake_final:
    count+=1
    flac = flac + '.flac'
    flac_train = os.path.join(DATA_FOLDER,TRAIN_FOLDER)+flac
    fake_list_of_flacs.append(flac_train)
    if (count== 5000):
        break
del flac, flac_train, count

for file in fake_list_of_flacs:    
    mel_spect(file, train_img_fake)    
del file

##########################################################